"""One Orbit Wars turn, reproduced with NumPy arrays.

`simulate_turn(state, moves)` does exactly what the official engine's
`interpreter` (kaggle_environments/envs/orbit_wars/orbit_wars.py) does for one
turn, in the same order, so that a recorded game can be replayed and fleets
already in flight can be run forward to see where they land.

Answers to the four questions checked against the code and the data
-------------------------------------------------------------------
1. `Replay.observation(t, player)` returns a dict with player, step (= t),
   angular_velocity, planets, fleets, comets (groups with planet_ids, paths and
   path_index) and comet_planet_ids. It does not include initial_planets
   (those are the positions in `planets(0)`) or next_fleet_id.
2. The per-game angular velocity is `game.meta["angular_velocity"]`.
3. `comet_first_step[g]` is the first state the group is on the board. At state t
   comet k of group g is at `comet_paths[g, k, t - comet_first_step[g]]`, so the
   engine's `path_index` at state t is `t - comet_first_step[g]`. The group is on
   the board for `comet_path_len[g]` states and is gone at state first + len.
4. Orbit offset c = -1: an orbiting planet's position at state s >= 1 equals
   `initial_angle + angular_velocity * (s - 1)`. The turn that goes from state s
   to state s + 1 runs with step = s and moves planets to the angle for step s.
   State 0 holds the initial positions, and state 1 has the same positions,
   because the first turn runs with step = 0. Checked: largest error 7e-6 over
   a whole game (float32 recordings).

State
-----
A state is a dict of NumPy arrays (n planets, m fleets):

  planets          (n, 7) float64  [id, owner, x, y, radius, ships, production]
  fleets           (m, 7) float64  [id, owner, x, y, angle, from_planet_id, ships]
  step             int             the engine's step for the next turn (= state index t)
  angular_velocity float           radians per turn
  next_fleet_id    int             id the next launched fleet gets
  initial_xy       (n, 2) float64  each planet's position at the start of the game
                                   (NaN for comets: they follow their path instead)
  comet_path       (n, MAX_COMET_PATH, 2) float64  a comet's path (NaN padded; all NaN for planets)
  comet_len        (n,) int        number of points on the path (0 = not a comet)
  path_index       (n,) int        index of the comet's current point on its path

The last four are "row aligned": row i describes planets[i]. When a comet is
removed, its row is removed from all of them together (`keep_planet_rows`).
Positions are board units (the board is 100 x 100); ship counts are whole numbers
stored as float64.
"""

import math

import numpy as np
from numba import njit

# Board and rules (same names and values as the engine)
BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0   # orbit radius + planet radius below this: the planet orbits
MAX_SHIP_SPEED = 6.0           # configuration.shipSpeed
LAUNCH_CLEARANCE = 0.1         # a new fleet starts this far outside its planet's edge
MAX_COMET_PATH = 40            # the engine only accepts comet paths of 5..40 points

# Column numbers
P_ID, P_OWNER, P_X, P_Y, P_RADIUS, P_SHIPS, P_PROD = range(7)
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)

# Ways a fleet can disappear (the `kind` in events)
HIT_PLANET, HIT_SUN, LEFT_BOARD = "planet", "sun", "edge"
# move_fleets codes a fleet's fate as the planet row it hit (>= 0) or one of these
STILL_FLYING, HIT_SUN_CODE, LEFT_BOARD_CODE = -1, -2, -3

ROW_ALIGNED = ("planets", "initial_xy", "comet_path", "comet_len", "path_index")


# ---------------------------------------------------------------------------------------------
# Geometry, copied from the engine
#
# These and move_fleets are compiled with numba's @njit: the code is plain Python
# loops, numba just runs them much faster. They can still be called from Python
# as usual. Set the environment variable NUMBA_DISABLE_JIT=1 to run them as
# ordinary Python (for example to step through with a debugger).
# ---------------------------------------------------------------------------------------------

@njit(cache=True)
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


@njit(cache=True)
def point_to_segment_distance(p, v, w):
    """Minimum distance from point p to line segment v-w."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return distance(p, v)
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)


@njit(cache=True)
def swept_pair_hit(A, B, P0, P1, r):
    """True iff a fleet moving A->B and a planet moving P0->P1 come within r
    of each other for some t in [0, 1]. Treats both segments as linear over
    the tick (planet rotation is linearised to its chord)."""
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


@njit(cache=True)
def fleet_speed(ships):
    """Board units per turn: 1 for a single ship, rising to MAX_SHIP_SPEED at 1000 ships."""
    speed = 1.0 + (MAX_SHIP_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(speed, MAX_SHIP_SPEED)


# ---------------------------------------------------------------------------------------------
# Building a state
# ---------------------------------------------------------------------------------------------

def make_state(game, t):
    """State t of a recorded game (a `replay.Replay`) as a simulator state dict."""
    planets = game.planets(t)
    n = len(planets)
    a = game.arrays

    # Starting positions of the base planets, looked up by id.
    start = {int(p[P_ID]): (p[P_X], p[P_Y]) for p in game.planets(0)}
    initial_xy = np.full((n, 2), np.nan)
    for i, pid in enumerate(planets[:, P_ID].astype(int)):
        if pid in start and not a["planet_is_comet"][game._p[t] + i]:
            initial_xy[i] = start[pid]

    # Comets on the board at state t: their paths and where along them they are.
    comet_path = np.full((n, MAX_COMET_PATH, 2), np.nan)
    comet_len = np.zeros(n, dtype=np.int64)
    path_index = np.zeros(n, dtype=np.int64)
    row_of = {int(pid): i for i, pid in enumerate(planets[:, P_ID])}
    for g, first in enumerate(a["comet_first_step"]):
        length = int(a["comet_path_len"][g])
        if not first <= t < first + length:
            continue
        for k, pid in enumerate(a["comet_ids"][g]):
            i = row_of.get(int(pid))
            if i is None:
                continue
            comet_path[i, :length] = a["comet_paths"][g, k, :length]
            comet_len[i] = length
            path_index[i] = t - first

    # Fleet ids are handed out in order, one per accepted order.
    next_fleet_id = int(np.sum(a["action_valid"][: game.action_rows(t).start]))

    return {
        "planets": planets,
        "fleets": game.fleets(t),
        "step": t,
        "angular_velocity": float(game.meta["angular_velocity"]),
        "next_fleet_id": next_fleet_id,
        "initial_xy": initial_xy,
        "comet_path": comet_path,
        "comet_len": comet_len,
        "path_index": path_index,
    }


def copy_state(state):
    """A copy that simulate_turn can change without touching the original."""
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}


def keep_planet_rows(state, keep):
    """Keep only the planet rows where `keep` (n,) bool is True, in every row-aligned array."""
    for key in ROW_ALIGNED:
        state[key] = state[key][keep]


def expired_comets(state):
    """(n,) bool: comets that have moved past the last point of their path."""
    return (state["comet_len"] > 0) & (state["path_index"] >= state["comet_len"])


# ---------------------------------------------------------------------------------------------
# The steps of one turn
# ---------------------------------------------------------------------------------------------

def launch_fleets(state, moves):
    """Step 1. Turn the players' orders into new fleets (changes `state` in place).

    moves: list over players (seat order) of [[planet_id, angle, ships], ...], or None.
    Returns the new fleet id for every order, in order (-1 = the game ignored it).
    """
    planets = state["planets"]
    new_fleets, launched = [], []
    for player, action in enumerate(moves or []):
        if not action or not isinstance(action, list):
            continue
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)
            rows = np.flatnonzero(planets[:, P_ID] == from_id)
            valid = len(rows) == 1 and planets[rows[0], P_OWNER] == player and 0 < ships <= planets[rows[0], P_SHIPS]
            if not valid:
                launched.append(-1)
                continue
            p = planets[rows[0]]
            p[P_SHIPS] -= ships
            x = p[P_X] + math.cos(angle) * (p[P_RADIUS] + LAUNCH_CLEARANCE)
            y = p[P_Y] + math.sin(angle) * (p[P_RADIUS] + LAUNCH_CLEARANCE)
            new_fleets.append([state["next_fleet_id"], player, x, y, angle, from_id, ships])
            launched.append(state["next_fleet_id"])
            state["next_fleet_id"] += 1
    if new_fleets:
        state["fleets"] = np.concatenate([state["fleets"], np.array(new_fleets, dtype=np.float64)])
    return launched


def produce(planets):
    """Step 2. Every owned planet gains its production (neutral planets do not). In place."""
    owned = planets[:, P_OWNER] != -1
    planets[owned, P_SHIPS] += planets[owned, P_PROD]


def planet_end_positions(state):
    """Step 3. Where every planet will be at the end of this turn.

    Returns new_xy (n, 2) and check (n,) bool. check is False for a comet's first
    placement (it appears mid-turn and is not tested against fleets this turn).
    Advances comets' path_index in place.
    """
    planets = state["planets"]
    n = len(planets)
    new_xy = planets[:, P_X:P_Y + 1].copy()
    check = np.ones(n, dtype=bool)
    w, step = state["angular_velocity"], state["step"]

    for i in range(n):
        if state["comet_len"][i] > 0:
            # Comets move one point along their path; past the end they stay put
            # this turn and are removed after movement.
            state["path_index"][i] += 1
            idx = state["path_index"][i]
            if idx < state["comet_len"][i]:
                new_xy[i] = state["comet_path"][i, idx]
                check[i] = planets[i, P_X] >= 0   # off-board placeholder = first placement
            continue
        dx = state["initial_xy"][i, 0] - CENTER
        dy = state["initial_xy"][i, 1] - CENTER
        r = math.sqrt(dx ** 2 + dy ** 2)
        if r + planets[i, P_RADIUS] < ROTATION_RADIUS_LIMIT:
            # Orbiting: absolute angle from the starting position. Static planets don't move.
            angle = math.atan2(dy, dx) + w * step
            new_xy[i] = (CENTER + r * math.cos(angle), CENTER + r * math.sin(angle))
    return new_xy, check


@njit(cache=True)
def move_fleets(fleets, planets, new_xy, check):
    """Step 4. Move every fleet one turn and find what it hit (changes fleet x, y in place).

    Returns fate (m,) int: the planet row the fleet hit, or HIT_SUN_CODE / LEFT_BOARD_CODE,
    or STILL_FLYING.
    """
    fate = np.full(len(fleets), STILL_FLYING, dtype=np.int64)
    for f in range(len(fleets)):
        angle, ships = fleets[f, F_ANGLE], fleets[f, F_SHIPS]
        speed = fleet_speed(ships)
        old_pos = (fleets[f, F_X], fleets[f, F_Y])
        fleets[f, F_X] += math.cos(angle) * speed
        fleets[f, F_Y] += math.sin(angle) * speed
        new_pos = (fleets[f, F_X], fleets[f, F_Y])

        # Planets first, in list order: the first one hit wins.
        for i in range(len(planets)):
            if not check[i]:
                continue
            p_old = (planets[i, P_X], planets[i, P_Y])
            if swept_pair_hit(old_pos, new_pos, p_old, new_xy[i], planets[i, P_RADIUS]):
                fate[f] = i
                break
        if fate[f] != STILL_FLYING:
            continue

        if not (0 <= new_pos[0] <= BOARD_SIZE and 0 <= new_pos[1] <= BOARD_SIZE):
            fate[f] = LEFT_BOARD_CODE
        elif point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            fate[f] = HIT_SUN_CODE
    return fate



def combat(garrison_owner, garrison_ships, arriving):
    """Step 6 for one planet. Returns its (owner, ships) after the fight.

    arriving: dict owner -> total ships arriving this turn (at least one entry).
    The biggest force minus the second biggest survives (an exact tie leaves
    nobody); a lone owner keeps all its ships. A survivor of the same owner
    reinforces; otherwise it fights the garrison and captures only if the
    garrison goes below zero.
    """
    ranked = sorted(arriving.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = ranked[0]
    if len(ranked) > 1:
        survivor_ships = top_ships - ranked[1][1]
        survivor_owner = top_owner if survivor_ships > 0 else -1
    else:
        survivor_owner, survivor_ships = top_owner, top_ships

    if survivor_ships > 0:
        if garrison_owner == survivor_owner:
            garrison_ships += survivor_ships
        else:
            garrison_ships -= survivor_ships
            if garrison_ships < 0:
                garrison_owner, garrison_ships = survivor_owner, abs(garrison_ships)
    return garrison_owner, garrison_ships


# ---------------------------------------------------------------------------------------------
# One turn
# ---------------------------------------------------------------------------------------------

def simulate_turn(state, moves=None):
    """Play one turn. Does not change `state`.

    moves: list over players of [[planet_id, angle, ships], ...]; None = nobody launches.
    Returns (new_state, events):
      new_state  the state after the turn; new_state["launched"] lists the fleet id
                 each order created (-1 if ignored), in the order they were given
      events     one tuple per fleet removed this turn:
                 (fleet_id, owner, ships, kind, planet_id or -1), kind in "planet" / "sun" / "edge".
                 A fleet that hits a comet on the turn it leaves the board is listed
                 as "planet" with that comet's id, but its ships simply vanish.
    New comet spawns and the end of the game are not simulated.
    """
    state = copy_state(state)

    # 0. Remove comets that are already past the end of their path (before launches).
    keep_planet_rows(state, ~expired_comets(state))

    # 1. Launch.
    state["launched"] = launch_fleets(state, moves)
    planets, fleets = state["planets"], state["fleets"]

    # 2. Production.
    produce(planets)

    # 3. Planet end-of-turn positions.
    new_xy, check = planet_end_positions(state)

    # 4. Fleet movement and collisions.
    fate = move_fleets(fleets, planets, new_xy, check)
    events = []
    arrivals = {}   # planet id -> {owner: ships}
    for f, where in enumerate(fate):
        if where == STILL_FLYING:
            continue
        fid, owner, ships = int(fleets[f, F_ID]), int(fleets[f, F_OWNER]), int(fleets[f, F_SHIPS])
        if where >= 0:
            pid = int(planets[where, P_ID])
            events.append((fid, owner, ships, HIT_PLANET, pid))
            by_owner = arrivals.setdefault(pid, {})
            by_owner[owner] = by_owner.get(owner, 0) + ships
        else:
            events.append((fid, owner, ships, HIT_SUN if where == HIT_SUN_CODE else LEFT_BOARD, -1))

    # 5. Apply planet positions, remove expired comets and the fleets that were removed.
    planets[:, P_X:P_Y + 1] = new_xy
    keep_planet_rows(state, ~expired_comets(state))
    state["fleets"] = fleets[fate == STILL_FLYING]
    planets = state["planets"]

    # 6. Combat at every planet that fleets reached (a comet that just left has no row: ships lost).
    for i in range(len(planets)):
        pid = int(planets[i, P_ID])
        if pid in arrivals:
            owner, ships = combat(int(planets[i, P_OWNER]), int(planets[i, P_SHIPS]), arrivals[pid])
            planets[i, P_OWNER], planets[i, P_SHIPS] = owner, ships

    state["step"] += 1
    return state, events
