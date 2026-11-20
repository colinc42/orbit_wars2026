"""Condensed summaries of an Orbit Wars game state, and the fleets launched each turn.

A state is given either as a Kaggle observation dict (``obs["planets"]``, ``obs["fleets"]``) or as
the arrays a ``Replay`` returns (``game.planets(t)``, ``game.fleets(t)``). Rows are

    planets: [id, owner, x, y, radius, ships, production]     owner -1 = neutral
    fleets:  [id, owner, x, y, angle, from_planet_id, ships]

Everything here is plain numpy with no Python loops over planets or fleets.

    from orbit_rl.replay import Replay
    from orbit_rl.utils import summarise, launched

    game = Replay("replay_memory/games/000/g0000042.npz")
    s = summarise(game.planets(10), game.fleets(10), player=0, n=3)
    s["balance"], s["production"], s["grid"].shape          # float, (4,), (4, 3, 3)
    new = launched(game.fleets(10), game.fleets(11))       # fleets sent after seeing state 10

For training, ``game_launches`` finds the launches for every turn of a game in one pass.
"""

import numpy as np

BOARD = 100.0          # the board is 100 x 100
MAX_PLAYERS = 4

# Column numbers
P_ID, P_OWNER, P_X, P_Y, P_RADIUS, P_SHIPS, P_PROD = range(7)
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)


def _rows(x):
    """A list of rows or an array -> float64 array of shape (k, 7), also when there are no rows."""
    return np.asarray(x, dtype=np.float64).reshape(-1, 7)


def _state(state, fleets):
    if isinstance(state, dict):
        return _rows(state.get("planets", [])), _rows(state.get("fleets", []))
    return _rows(state), _rows(fleets if fleets is not None else [])


# ---------------------------------------------------------------- single-state summaries

def ships_per_player(planets, fleets=None, n_players=MAX_PLAYERS):
    """(n_players,) ships on planets plus ships in flight. This equals the game's score."""
    planets, fleets = _state(planets, fleets)
    total = np.zeros(n_players)
    for rows, owner, ships in ((planets, P_OWNER, P_SHIPS), (fleets, F_OWNER, F_SHIPS)):
        o = rows[:, owner].astype(np.int64)
        keep = o >= 0                                       # neutral planets belong to nobody
        total += np.bincount(o[keep], weights=rows[keep, ships], minlength=n_players)[:n_players]
    return total


def balance_of_power(planets, fleets=None, player=0, n_players=MAX_PLAYERS):
    """Player's ships divided by the ships of all other players together.

    1.0 is level in a two-player game; in a four-player game 1/3 means an equal share. Returns
    inf when every opponent has been reduced to zero ships.
    """
    total = ships_per_player(planets, fleets, n_players)
    mine = total[player]
    others = total.sum() - mine
    return mine / others if others > 0 else np.inf


def production_per_player(planets, fleets=None, n_players=MAX_PLAYERS):
    """(n_players,) ships per turn each player gets from the planets (and comets) it owns."""
    planets, _ = _state(planets, fleets)
    o = planets[:, P_OWNER].astype(np.int64)
    keep = o >= 0
    return np.bincount(o[keep], weights=planets[keep, P_PROD], minlength=n_players)[:n_players]


def ship_grid(planets, fleets=None, n=3, n_players=MAX_PLAYERS, include_fleets=True):
    """(n_players, n, n) ships each player has in each of n*n equal squares of the board.

    ``grid[p, row, col]``: row 0 is the bottom strip (small y) and col 0 the left strip (small x),
    so ``grid[p]`` lines up with the board when drawn with ``imshow(..., origin="lower")``. Garrisons
    count at the planet's centre and fleets at their current position. Neutral planets are left out.
    Channels for seats that are not in the game are zero.
    """
    planets, fleets = _state(planets, fleets)
    parts = [planets[:, [P_OWNER, P_X, P_Y, P_SHIPS]]]
    if include_fleets:
        parts.append(fleets[:, [F_OWNER, F_X, F_Y, F_SHIPS]])
    rows = np.concatenate(parts)
    owner = rows[:, 0].astype(np.int64)
    rows, owner = rows[owner >= 0], owner[owner >= 0]

    col = np.clip((rows[:, 1] * (n / BOARD)).astype(np.int64), 0, n - 1)   # x = 100 goes in the last square
    row = np.clip((rows[:, 2] * (n / BOARD)).astype(np.int64), 0, n - 1)
    flat = np.bincount((owner * n + row) * n + col, weights=rows[:, 3], minlength=n_players * n * n)
    return flat[: n_players * n * n].reshape(n_players, n, n)


def summarise(planets, fleets=None, player=0, n=3, n_players=MAX_PLAYERS):
    """All three summaries of one state at once.

    ``planets`` may be an observation dict, in which case ``fleets`` is ignored and ``player``
    defaults to ``obs["player"]``.

    Returns a dict with
      balance     float           player's ships / all other players' ships
      production  (n_players,)    ships per turn from owned planets
      grid        (n_players, n, n)  ships per player per square
      ships       (n_players,)    total ships per player (the score; free once the rest is done)
    """
    if isinstance(planets, dict):
        player = planets.get("player", player)
    planets, fleets = _state(planets, fleets)
    grid = ship_grid(planets, fleets, n, n_players)
    ships = grid.sum(axis=(1, 2))                           # every owned ship lands in some square
    others = ships.sum() - ships[player]
    return {
        "balance": ships[player] / others if others > 0 else np.inf,
        "production": production_per_player(planets, None, n_players),
        "grid": grid,
        "ships": ships,
    }


# ---------------------------------------------------------------- launches

def launched(fleets_before, fleets_after):
    """Fleets in ``fleets_after`` that were launched this turn: (k, 7) fleet rows.

    The game numbers fleets from one counter that only goes up, so a new fleet is any fleet whose
    id is higher than every id in the previous state. That is one comparison per fleet, with no
    matching of ids. ``from_planet_id`` gives the planet it left and ``ships`` its size; its
    position has already moved one tick along its path.

    Not included, because they never appear in a state: launches that hit something or leave the
    board on their first tick (about 1% of launches). The order table (``action_fleet`` in a
    replay) has those.
    """
    before, after = _rows(fleets_before), _rows(fleets_after)
    if len(before) == 0:
        return after
    return after[after[:, F_ID] > before[:, F_ID].max()]


def launched_per_player(fleets_before, fleets_after, n_players=MAX_PLAYERS):
    """(n_players,) ships each player launched this turn (visible launches only)."""
    new = launched(fleets_before, fleets_after)
    return np.bincount(new[:, F_OWNER].astype(np.int64), weights=new[:, F_SHIPS],
                       minlength=n_players)[:n_players]


def game_launches(replay):
    """Every visible launch in a whole game in one vectorised pass over the replay's flat arrays.

    Returns (step, rows): ``rows`` are fleet rows [id, owner, x, y, angle, from, ships] and
    ``step[i]`` is the state they first appear in, so they were launched by the orders given at
    ``step[i] - 1``. Same result as calling ``launched`` on every pair of consecutive states.
    """
    a = replay.arrays
    counts = a["fleet_n"].astype(np.int64)
    ids = a["fleet_id"].astype(np.int64)
    step = np.repeat(np.arange(len(counts)), counts)

    # Highest fleet id seen in each state, carried forward through states with no fleets.
    top = np.full(len(counts), -1, dtype=np.int64)
    np.maximum.at(top, step, ids)
    top = np.maximum.accumulate(top)
    prev_top = np.concatenate(([-1], top[:-1]))

    new = ids > prev_top[step]
    new &= step > 0                                         # fleets in the first state have no "before"
    rows = np.stack([a[k][new] for k in ("fleet_id", "fleet_owner", "fleet_x", "fleet_y",
                                         "fleet_angle", "fleet_from", "fleet_ships")], axis=1)
    return step[new], rows.astype(np.float64)
