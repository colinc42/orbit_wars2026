"""Map a discrete policy decision onto Orbit Wars moves.

The env wants `[[from_planet_id, angle_radians, num_ships], ...]` — a
continuous angle and an integer ship count. Learning raw angles is hard, so
the policy instead picks, for every planet slot it owns:

  target[i]   in [0, MAX_PLANETS]   a planet slot to attack/reinforce,
                                    or MAX_PLANETS = HOLD (send nothing)
  fraction[i] in [0, len(FRACTIONS)) which share of the garrison to send

and `decode` converts that into a move, aiming ahead of orbiting planets.
"""

import math

from features import (
    CENTER,
    MAX_PLANETS,
    P_ID,
    P_RADIUS,
    P_SHIPS,
    P_X,
    P_Y,
    _get,
    is_orbiting,
)

HOLD = MAX_PLANETS
N_TARGETS = MAX_PLANETS + 1
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
N_FRACTIONS = len(FRACTIONS)
MAX_SHIP_SPEED = 6.0


def fleet_speed(ships, max_speed=MAX_SHIP_SPEED):
    if ships <= 1:
        return 1.0
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(speed, max_speed)


def predict_position(planet, turns, angular_velocity, comet_ids=()):
    """Where `planet` will be after `turns` steps. Comets are treated as static
    (their exact paths are in obs['comets'] if you want to do better)."""
    if planet[P_ID] in comet_ids or not is_orbiting(planet):
        return planet[P_X], planet[P_Y]
    dx, dy = planet[P_X] - CENTER, planet[P_Y] - CENTER
    r = math.hypot(dx, dy)
    ang = math.atan2(dy, dx) + angular_velocity * turns
    return CENTER + r * math.cos(ang), CENTER + r * math.sin(ang)


def aim(source, target, ships, angular_velocity, comet_ids=(), iters=3):
    """Angle from source to where target will be when the fleet arrives.
    Does not avoid the sun — a good first thing for a smarter agent to add."""
    speed = fleet_speed(ships)
    tx, ty = target[P_X], target[P_Y]
    for _ in range(iters):
        dist = math.hypot(tx - source[P_X], ty - source[P_Y]) - source[P_RADIUS]
        tx, ty = predict_position(target, max(dist, 0.0) / speed, angular_velocity, comet_ids)
    return math.atan2(ty - source[P_Y], tx - source[P_X])


def decode(obs, targets, fractions, min_ships=1):
    """Convert per-slot (target, fraction) choices into env moves.

    `targets` / `fractions` are length-MAX_PLANETS sequences of ints indexed by
    planet slot (the order of obs['planets']). Slots we don't own are ignored.
    """
    me = _get(obs, "player", 0)
    planets = list(_get(obs, "planets", []) or [])[:MAX_PLANETS]
    angular_velocity = _get(obs, "angular_velocity", 0.0) or 0.0
    comet_ids = set(_get(obs, "comet_planet_ids", []) or [])

    moves = []
    for i, src in enumerate(planets):
        if src[1] != me or src[P_SHIPS] <= 0:
            continue
        t = int(targets[i])
        if t == HOLD or t == i or t >= len(planets):
            continue
        ships = int(src[P_SHIPS] * FRACTIONS[int(fractions[i])])
        if ships < min_ships:
            continue
        angle = aim(src, planets[t], ships, angular_velocity, comet_ids)
        moves.append([int(src[P_ID]), float(angle), ships])
    return moves
