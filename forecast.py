"""Per-planet forecast of incoming ships, from the fleets already in flight.

Fleets never change course, so every fleet in flight at a state can be run
forward with the exact game rules (`sim.simulate_turn`, nobody launching
anything new) to see which planet it hits and when. `forecast` does that for
HORIZON turns and sums the result per planet, per time bucket and per player.

    from forecast import forecast
    from sim import make_state
    fc = forecast(make_state(game, t), game.n_players)
    fc["incoming"][row, bucket, player]    # ships `player` lands on planet `row` in `bucket`

Owners are absolute player ids here (-1 = neutral); `features.py` turns them
into "me / enemies" from one player's point of view.
"""

import numpy as np

from sim import HIT_PLANET, P_ID, P_OWNER, P_PROD, P_SHIPS, F_OWNER, F_SHIPS, simulate_turn

HORIZON = 50   # turns simulated ahead
# Inclusive ranges of "turns ahead"; 13 buckets.
BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10),
           (11, 15), (16, 25), (26, 50)]
N_BUCKETS = len(BUCKETS)

BUCKET_OF_TURN = np.full(HORIZON + 1, -1, dtype=np.int64)   # turns ahead -> bucket (index 0 unused)
for _b, (_lo, _hi) in enumerate(BUCKETS):
    BUCKET_OF_TURN[_lo:_hi + 1] = _b
BUCKET_END = [hi for lo, hi in BUCKETS]                      # last turn of each bucket


def forecast(state, n_players):
    """Run the fleets in flight forward, assuming nobody launches anything new.

    state: a sim state (see sim.py); it is not changed.
    Returns a dict (rows follow the order of state["planets"]; n = number of planets):
      incoming    (n, 13, n_players) int  ships of each player that hit the planet in each bucket
      proj_owner  (n, 13) int             owner at the end of each bucket's last turn (-1 neutral)
      proj_ships  (n, 13) int             garrison at the end of each bucket's last turn
      lost        (n_players,) int        ships that hit the sun, leave the board, or hit a
                                          comet on the turn it leaves, within HORIZON
      unresolved  (n_players,) int        ships still flying after HORIZON turns
      events      list of (fleet_id, owner, ships, kind, planet_id, turns_ahead), one per fleet
                  that disappears within HORIZON (kind as in sim.simulate_turn)
    A comet that leaves the board keeps its row, with its last owner and ships.
    """
    start = state["planets"]
    n = len(start)
    row_of = {int(pid): i for i, pid in enumerate(start[:, P_ID])}

    incoming = np.zeros((n, N_BUCKETS, n_players), dtype=np.int64)
    proj_owner = np.zeros((n, N_BUCKETS), dtype=np.int64)
    proj_ships = np.zeros((n, N_BUCKETS), dtype=np.int64)
    lost = np.zeros(n_players, dtype=np.int64)
    unresolved = np.zeros(n_players, dtype=np.int64)
    owner = start[:, P_OWNER].astype(np.int64)   # latest known owner and garrison of each row
    ships = start[:, P_SHIPS].astype(np.int64)
    events = []

    turns_done = 0
    for k in range(1, HORIZON + 1):
        if len(state["fleets"]) == 0:
            break   # nothing left in flight: only production changes garrisons from here
        state, turn_events = simulate_turn(state)
        turns_done = k

        on_board = set(state["planets"][:, P_ID].astype(int).tolist())
        for fleet_id, who, n_ships, kind, pid in turn_events:
            events.append((fleet_id, who, n_ships, kind, pid, k))
            if kind == HIT_PLANET and pid in on_board:
                incoming[row_of[pid], BUCKET_OF_TURN[k], who] += n_ships
            else:
                lost[who] += n_ships   # sun, edge, or a comet that left this turn

        rows = [row_of[int(pid)] for pid in state["planets"][:, P_ID]]
        owner[rows] = state["planets"][:, P_OWNER]
        ships[rows] = state["planets"][:, P_SHIPS]

        b = BUCKET_OF_TURN[k]
        if k == BUCKET_END[b]:
            proj_owner[:, b] = owner
            proj_ships[:, b] = ships

    for f in state["fleets"]:
        unresolved[int(f[F_OWNER])] += int(f[F_SHIPS])

    # Buckets after the last fleet landed: owners stay, owned planets keep producing.
    for b, end in enumerate(BUCKET_END):
        if end > turns_done:
            proj_owner[:, b], proj_ships[:, b] = coast(state, row_of, owner, ships, end - turns_done)

    return {
        "incoming": incoming,
        "proj_owner": proj_owner,
        "proj_ships": proj_ships,
        "lost": lost,
        "unresolved": unresolved,
        "events": events,
    }


def coast(state, row_of, owner, ships, turns):
    """Owner and garrison of every start row `turns` turns after `state`, with no fleets in flight.

    owner, ships: (n,) the latest values per start row (rows that already left keep theirs).
    A comet only produces while it is still on the board.
    Returns (owner, ships), each (n,) int.
    """
    owner, ships = owner.copy(), ships.copy()
    planets = state["planets"]
    for i in range(len(planets)):
        row = row_of[int(planets[i, P_ID])]
        if owner[row] == -1:
            continue   # neutral planets don't produce
        producing = turns
        if state["comet_len"][i] > 0:
            producing = min(turns, state["comet_len"][i] - 1 - state["path_index"][i])
        ships[row] += int(planets[i, P_PROD]) * producing
    return owner, ships
