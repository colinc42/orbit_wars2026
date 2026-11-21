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


# =============================================================================================
# Planet tokens with the incoming-ships forecast (PLANET_FORECAST_SPEC.md, step 4)
#
# Everything above this line is the older observation encoder used by PolicyNet.
# Below: one player's view of a recorded state, as 44 planet tokens plus one
# global vector, built from a Replay and its precomputed forecast
# (precompute.py). Fleets are not inputs: their effect is in the forecast blocks.
#
#     from precompute import forecast_path, load_forecast
#     fc = load_forecast(forecast_path("replay_memory", "games/000/g0000042.npz"))
#     planets, pmask, pids = planet_tokens(game, fc, t, player)    # [44, 146], [44], [44]
#     g = global_token(game, fc, t, player)                        # [20]
#
# Point of view: the board is rotated so the player's starting planet is in the
# x > 50, y > 50 quarter, and owners become classes (see owner_class). Raw player
# ids are never features.
# =============================================================================================

from forecast import BUCKETS, N_BUCKETS  # noqa: E402  (kept next to the code that uses it)
from sim import MAX_COMET_PATH, copy_state, make_state, planet_end_positions  # noqa: E402

N_SLOTS = 44           # planet slots per state: up to 40 planets + 4 comets
N_CLASSES = 5          # owner classes, see owner_class
N_PLAYER_CLASSES = 4   # the classes that are players: me, enemy +1, +2, +3
SHIP_SCALE = 8.0       # ship counts enter as log1p(ships) / SHIP_SCALE
POSITION_SCALE = 50.0  # positions as (x - 50) / 50, so the board is [-1, 1]
VELOCITY_SCALE = 4.0   # board units per turn; comets move 4 per turn, the fastest of anything
RADIUS_SCALE = 3.0
PRODUCTION_SCALE = 5.0
TOTAL_PRODUCTION_SCALE = 50.0
STEP_SCALE = 500.0     # episode length
ANGULAR_VELOCITY_SCALE = 0.05
COMET_SPAWN_STATES = (50, 150, 250, 350, 450)   # first state of each comet group
COMET_WAIT_SCALE = 100.0

# A player's starting planet is planet `4 * g + HOME_OFFSET[n_players][player]` of
# some group g of four symmetric planets (see the engine's interpreter).
HOME_OFFSET = {2: (0, 3), 4: (0, 1, 2, 3)}

# Planet token blocks, in order.
CURRENT_FEATS = 11 + N_CLASSES                    # see current_features
INCOMING_FEATS = N_BUCKETS * N_PLAYER_CLASSES     # log1p(ships)/8 arriving per bucket and class
PROJ_OWNER_FEATS = N_BUCKETS * N_CLASSES          # one-hot owner class at the end of each bucket
PROJ_SHIPS_FEATS = N_BUCKETS                      # log1p(garrison)/8 at the end of each bucket
PLANET_FEATS = CURRENT_FEATS + INCOMING_FEATS + PROJ_OWNER_FEATS + PROJ_SHIPS_FEATS   # 146

GLOBAL_BASE_FEATS = 12                            # see global_token
GLOBAL_FEATS = GLOBAL_BASE_FEATS + 2 * N_PLAYER_CLASSES   # + lost and unresolved per class = 20

CLASS_NAMES = ("neutral", "me", "enemy+1", "enemy+2", "enemy+3")


def rotate_quarter_turns(dx, dy, k):
    """Rotate the vector (dx, dy) by k quarter turns anticlockwise. Works on arrays too."""
    for _ in range(int(k) % 4):
        dx, dy = -dy, dx
    return dx, dy


def start_turns(planets0, n_players):
    """(n_players,) int: quarter turns (anticlockwise) that bring each player's starting
    planet into the x > 50, y > 50 quarter of the board.

    planets0: planet rows of state 0. The starting planets are the group of four
    symmetric planets in which every player owns its seat's member (HOME_OFFSET). In a
    scrambled start other planets are handed out too, so "planets owned at state 0" is
    not enough on its own; any group that fits gives the same answer.
    """
    owner_of = {int(p[P_ID]): int(p[P_OWNER]) for p in planets0}
    xy_of = {int(p[P_ID]): (p[P_X], p[P_Y]) for p in planets0}
    offsets = HOME_OFFSET[n_players]
    for base in range(0, len(planets0) - 3, 4):
        if all(owner_of.get(base + off) == player for player, off in enumerate(offsets)):
            break
    else:
        raise ValueError("no group of starting planets found at state 0")

    turns = np.zeros(n_players, dtype=np.int64)
    for player, off in enumerate(offsets):
        x, y = xy_of[base + off]
        for k in range(4):
            dx, dy = rotate_quarter_turns(x - CENTER, y - CENTER, k)
            if dx > 0 and dy > 0:
                turns[player] = k
                break
    return turns


def owner_class(owner, player, turns):
    """Owner id -> class from `player`'s point of view.

    0 neutral, 1 me, and for enemies, where their start sits relative to mine once
    the board is turned my way: 2 one quarter turn ahead (anticlockwise, the way
    planets orbit), 3 opposite, 4 one quarter turn behind. In a 2-player game the
    enemy is always 3.
    """
    owner = int(owner)
    if owner == -1:
        return 0
    if owner == player:
        return 1
    return 1 + int((turns[player] - turns[owner]) % 4)


def class_of_players(player, turns):
    """(n_players,) int: owner_class of every player id."""
    return np.array([owner_class(q, player, turns) for q in range(len(turns))], dtype=np.int64)


def per_class(values, player, turns):
    """(n_players,) values per player id -> (4,) per class: me, enemy +1, +2, +3."""
    out = np.zeros(N_PLAYER_CLASSES)
    for q, c in enumerate(class_of_players(player, turns)):
        out[c - 1] += values[q]
    return out


def current_features(state, player, turns):
    """(n, CURRENT_FEATS) float32 for the planets of a sim state, in their row order:

    x, y, vx, vy, orbit_r, is_orbiting, is_comet, radius, production, log1p(ships)/8,
    comet_turns_left, owner one-hot (5). Positions and velocities are rotated to the
    player's view; the velocity is the planet's move during the next turn.
    """
    planets = state["planets"]
    n = len(planets)
    k = turns[player]
    new_xy, _ = planet_end_positions(copy_state(state))   # on a copy: it advances comets
    x, y = rotate_quarter_turns(planets[:, P_X] - CENTER, planets[:, P_Y] - CENTER, k)
    vx, vy = rotate_quarter_turns(new_xy[:, 0] - planets[:, P_X], new_xy[:, 1] - planets[:, P_Y], k)
    is_comet = state["comet_len"] > 0
    start_r = np.hypot(state["initial_xy"][:, 0] - CENTER, state["initial_xy"][:, 1] - CENTER)
    is_orbit = ~is_comet & (np.nan_to_num(start_r, nan=np.inf) + planets[:, P_RADIUS] < ROTATION_RADIUS_LIMIT)
    turns_left = np.where(is_comet, state["comet_len"] - 1 - state["path_index"], 0)

    out = np.zeros((n, CURRENT_FEATS), dtype=np.float32)
    out[:, 0] = x / POSITION_SCALE
    out[:, 1] = y / POSITION_SCALE
    out[:, 2] = vx / VELOCITY_SCALE
    out[:, 3] = vy / VELOCITY_SCALE
    out[:, 4] = np.hypot(x, y) / POSITION_SCALE
    out[:, 5] = is_orbit
    out[:, 6] = is_comet
    out[:, 7] = planets[:, P_RADIUS] / RADIUS_SCALE
    out[:, 8] = planets[:, P_PROD] / PRODUCTION_SCALE
    out[:, 9] = np.log1p(planets[:, P_SHIPS]) / SHIP_SCALE
    out[:, 10] = turns_left / MAX_COMET_PATH
    for i in range(n):
        out[i, 11 + owner_class(planets[i, P_OWNER], player, turns)] = 1.0
    return out


def forecast_features(incoming, proj_owner, proj_ships, player, turns):
    """(n, INCOMING_FEATS + PROJ_OWNER_FEATS + PROJ_SHIPS_FEATS) float32 from one state's
    forecast rows: incoming (n, 13, n_players), proj_owner (n, 13), proj_ships (n, 13)."""
    n = len(incoming)
    classes = class_of_players(player, turns)   # player id -> 1..4

    by_class = np.zeros((n, N_BUCKETS, N_PLAYER_CLASSES))
    for q, c in enumerate(classes):
        by_class[:, :, c - 1] += incoming[:, :, q]

    owner_onehot = np.zeros((n, N_BUCKETS, N_CLASSES))
    for i in range(n):
        for b in range(N_BUCKETS):
            owner_onehot[i, b, owner_class(proj_owner[i, b], player, turns)] = 1.0

    return np.concatenate([
        (np.log1p(by_class) / SHIP_SCALE).reshape(n, INCOMING_FEATS),
        owner_onehot.reshape(n, PROJ_OWNER_FEATS),
        np.log1p(proj_ships) / SHIP_SCALE,
    ], axis=1).astype(np.float32)


def planet_tokens(game, fc, t, player, state=None):
    """One player's view of state t of a Replay.

    fc: the game's forecast arrays (precompute.load_forecast). state: make_state(game, t),
    if you already have it. Returns
      planets  [N_SLOTS, PLANET_FEATS] float32  planets sorted by id, zero padding after them
      pmask    [N_SLOTS] bool                   True for real planets
      pids     [N_SLOTS] int32                  planet id per slot, -1 for padding
    """
    state = make_state(game, t) if state is None else state
    turns = start_turns(game.planets(0), game.n_players)
    rows = slice(game._p[t], game._p[t + 1])
    feats = np.concatenate([
        current_features(state, player, turns),
        forecast_features(fc["fc_incoming"][rows], fc["fc_proj_owner"][rows], fc["fc_proj_ships"][rows],
                          player, turns),
    ], axis=1)

    order = np.argsort(state["planets"][:, P_ID], kind="stable")
    n = len(order)
    if n > N_SLOTS:
        raise ValueError(f"{n} planets do not fit in {N_SLOTS} slots")
    planets = np.zeros((N_SLOTS, PLANET_FEATS), dtype=np.float32)
    pmask = np.zeros(N_SLOTS, dtype=bool)
    pids = np.full(N_SLOTS, -1, dtype=np.int32)
    planets[:n] = feats[order]
    pmask[:n] = True
    pids[:n] = state["planets"][order, P_ID]
    return planets, pmask, pids


def global_token(game, fc, t, player):
    """[GLOBAL_FEATS] float32 for `player` at state t of a Replay:

      0      step / 500
      1      angular velocity / 0.05
      2      1 in a 4-player game
      3-6    log1p(ships on planets and in fleets) / 8 for me, enemy +1, +2, +3
      7-10   total production / 50 for me, enemy +1, +2, +3
      11     turns until the next comet group appears / 100, at most 1 (1 if there are no more)
      12-15  log1p(forecast ships lost to the sun, the edge or a leaving comet) / 8 per class
      16-19  log1p(forecast ships still flying after the horizon) / 8 per class
    """
    n_players = game.n_players
    turns = start_turns(game.planets(0), n_players)
    ships = np.zeros(n_players)
    production = np.zeros(n_players)
    for p in game.planets(t):
        if p[P_OWNER] >= 0:
            ships[int(p[P_OWNER])] += p[P_SHIPS]
            production[int(p[P_OWNER])] += p[P_PROD]
    for f in game.fleets(t):
        ships[int(f[F_OWNER])] += f[F_SHIPS]
    upcoming = [s - t for s in COMET_SPAWN_STATES if s > t]

    g = np.zeros(GLOBAL_FEATS, dtype=np.float32)
    g[0] = t / STEP_SCALE
    g[1] = game.meta["angular_velocity"] / ANGULAR_VELOCITY_SCALE
    g[2] = n_players == 4
    g[3:7] = np.log1p(per_class(ships, player, turns)) / SHIP_SCALE
    g[7:11] = per_class(production, player, turns) / TOTAL_PRODUCTION_SCALE
    g[11] = min(upcoming[0], COMET_WAIT_SCALE) / COMET_WAIT_SCALE if upcoming else 1.0
    g[12:16] = np.log1p(per_class(fc["fc_lost"][t], player, turns)) / SHIP_SCALE
    g[16:20] = np.log1p(per_class(fc["fc_unresolved"][t], player, turns)) / SHIP_SCALE
    return g


def forecast_table(game, fc, t, player, planet_id):
    """One planet's forecast from `player`'s point of view, as a readable table."""
    turns = start_turns(game.planets(0), game.n_players)
    planets = game.planets(t)
    i = int(np.flatnonzero(planets[:, P_ID] == planet_id)[0])
    row = game._p[t] + i
    classes = class_of_players(player, turns)
    now = CLASS_NAMES[owner_class(planets[i, P_OWNER], player, turns)]
    lines = [
        f"planet {planet_id} at state {t}, seen by player {player}: {now}, {int(planets[i, P_SHIPS])} ships",
        f"{'turns':>7} | {'me':>5} {'en+1':>5} {'en+2':>5} {'en+3':>5} | {'owner':>8} {'ships':>6}",
    ]
    for b, (lo, hi) in enumerate(BUCKETS):
        arriving = np.zeros(N_PLAYER_CLASSES, dtype=np.int64)
        for q, c in enumerate(classes):
            arriving[c - 1] += fc["fc_incoming"][row, b, q]
        owner = CLASS_NAMES[owner_class(fc["fc_proj_owner"][row, b], player, turns)]
        span = f"{lo}" if lo == hi else f"{lo}-{hi}"
        lines.append(f"{span:>7} | " + " ".join(f"{a:5d}" for a in arriving)
                     + f" | {owner:>8} {int(fc['fc_proj_ships'][row, b]):6d}")
    return "\n".join(lines)
