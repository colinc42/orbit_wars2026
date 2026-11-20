"""Turn a raw Orbit Wars observation into fixed-size numpy arrays.

The raw observation has a variable number of planets (20-40, plus comets) and
fleets. Neural networks want fixed shapes, so planets are placed into
MAX_PLANETS slots (padding marked by `planet_mask`) and fleets are summarised
per planet and globally.
"""

import math

import numpy as np

BOARD_SIZE = 100.0
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
EPISODE_STEPS = 500

# 40 planets max + up to 4 comets at a time; leave headroom for overlap.
MAX_PLANETS = 48

# Raw list layouts from the environment.
P_ID, P_OWNER, P_X, P_Y, P_RADIUS, P_SHIPS, P_PROD = range(7)
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)

PLANET_FEATURES = [
    "is_mine",
    "is_enemy",
    "is_neutral",
    "x",
    "y",
    "radius",
    "log_ships",
    "production",
    "is_comet",
    "is_orbiting",
    "my_fleets_heading_here",
    "enemy_fleets_heading_here",
]
GLOBAL_FEATURES = [
    "step",
    "angular_velocity",
    "my_planet_ships",
    "enemy_planet_ships",
    "my_fleet_ships",
    "enemy_fleet_ships",
    "my_production",
    "enemy_production",
]
N_PLANET_FEATURES = len(PLANET_FEATURES)
N_GLOBAL_FEATURES = len(GLOBAL_FEATURES)


def _get(obs, key, default=None):
    # Observations are kaggle `Struct`s (dict subclasses) or plain dicts.
    return obs.get(key, default) if hasattr(obs, "get") else getattr(obs, key, default)


def _log_ships(n):
    return math.log1p(max(n, 0)) / math.log1p(1000)


def is_orbiting(planet):
    dx, dy = planet[P_X] - CENTER, planet[P_Y] - CENTER
    return math.hypot(dx, dy) + planet[P_RADIUS] < ROTATION_RADIUS_LIMIT


def _fleet_target(fleet, planets):
    """Best guess at which planet a fleet is heading for: the planet closest to
    its line of travel, in front of it. Ignores orbital motion."""
    fx, fy, ang = fleet[F_X], fleet[F_Y], fleet[F_ANGLE]
    dx, dy = math.cos(ang), math.sin(ang)
    best, best_d = None, float("inf")
    for p in planets:
        px, py = p[P_X] - fx, p[P_Y] - fy
        along = px * dx + py * dy
        if along <= 0:
            continue
        perp = abs(px * dy - py * dx)
        if perp < p[P_RADIUS] + 2.0 and along < best_d:
            best, best_d = p[P_ID], along
    return best


def encode(obs):
    """Encode an observation.

    Returns a dict of numpy arrays:
      planets      (MAX_PLANETS, N_PLANET_FEATURES) float32
      planet_mask  (MAX_PLANETS,) bool   - slot holds a real planet
      source_mask  (MAX_PLANETS,) bool   - slot is a planet we own with ships
      globals      (N_GLOBAL_FEATURES,)  float32
      planet_ids   (MAX_PLANETS,) int64  - env planet id per slot, -1 = padding
    """
    me = _get(obs, "player", 0)
    planets = list(_get(obs, "planets", []) or [])[:MAX_PLANETS]
    fleets = list(_get(obs, "fleets", []) or [])
    comet_ids = set(_get(obs, "comet_planet_ids", []) or [])
    step = _get(obs, "step", 0) or 0

    feats = np.zeros((MAX_PLANETS, N_PLANET_FEATURES), dtype=np.float32)
    planet_mask = np.zeros(MAX_PLANETS, dtype=bool)
    source_mask = np.zeros(MAX_PLANETS, dtype=bool)
    planet_ids = np.full(MAX_PLANETS, -1, dtype=np.int64)
    slot_of = {}

    for i, p in enumerate(planets):
        owner = p[P_OWNER]
        feats[i, 0] = owner == me
        feats[i, 1] = owner not in (me, -1)
        feats[i, 2] = owner == -1
        feats[i, 3] = p[P_X] / BOARD_SIZE
        feats[i, 4] = p[P_Y] / BOARD_SIZE
        feats[i, 5] = p[P_RADIUS] / 3.0
        feats[i, 6] = _log_ships(p[P_SHIPS])
        feats[i, 7] = p[P_PROD] / 5.0
        feats[i, 8] = p[P_ID] in comet_ids
        feats[i, 9] = is_orbiting(p) and p[P_ID] not in comet_ids
        planet_mask[i] = True
        source_mask[i] = owner == me and p[P_SHIPS] > 0
        planet_ids[i] = p[P_ID]
        slot_of[p[P_ID]] = i

    g = np.zeros(N_GLOBAL_FEATURES, dtype=np.float32)
    g[0] = step / EPISODE_STEPS
    g[1] = (_get(obs, "angular_velocity", 0.0) or 0.0) / 0.05
    for p in planets:
        if p[P_OWNER] == me:
            g[2] += p[P_SHIPS]
            g[6] += p[P_PROD]
        elif p[P_OWNER] != -1:
            g[3] += p[P_SHIPS]
            g[7] += p[P_PROD]

    for f in fleets:
        mine = f[F_OWNER] == me
        g[4 if mine else 5] += f[F_SHIPS]
        target = _fleet_target(f, planets)
        if target in slot_of:
            feats[slot_of[target], 10 if mine else 11] += f[F_SHIPS]

    feats[:, 10] = np.log1p(feats[:, 10]) / math.log1p(1000)
    feats[:, 11] = np.log1p(feats[:, 11]) / math.log1p(1000)
    g[2:6] = np.log1p(g[2:6]) / math.log1p(10000)
    g[6:8] = g[6:8] / 50.0

    return {
        "planets": feats,
        "planet_mask": planet_mask,
        "source_mask": source_mask,
        "globals": g,
        "planet_ids": planet_ids,
    }


def ship_totals(obs):
    """(my_total, enemy_total) ships on planets + in fleets — the game's score."""
    me = _get(obs, "player", 0)
    mine = enemy = 0
    for p in _get(obs, "planets", []) or []:
        if p[P_OWNER] == me:
            mine += p[P_SHIPS]
        elif p[P_OWNER] != -1:
            enemy += p[P_SHIPS]
    for f in _get(obs, "fleets", []) or []:
        if f[F_OWNER] == me:
            mine += f[F_SHIPS]
        else:
            enemy += f[F_SHIPS]
    return mine, enemy
