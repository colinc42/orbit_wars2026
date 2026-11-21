"""Test 1: simulate_turn reproduces recorded games turn by turn.

For every sampled game and every turn t: start from recorded state t, apply the
recorded orders, and compare with recorded state t + 1. Turns where a new comet
group appears are skipped (the simulator does not spawn comets).

    python -m pytest tests/test_sim.py -s       # prints the report
"""

import numpy as np
import pytest

from conftest import sample_games
from replay import Replay
from sim import F_FROM, F_ID, F_OWNER, F_SHIPS, F_X, F_Y, P_ID, P_OWNER, P_SHIPS, P_X, P_Y, make_state, simulate_turn

POSITION_TOLERANCE = 1e-3   # recordings are float32


def spawn_turns(game):
    """Turns t after which a new comet group is on the board (state t + 1 is its first)."""
    return {int(first) - 1 for first in game.arrays["comet_first_step"]}


def compare_turn(game, t):
    """Simulate turn t of `game`; return a list of mismatch descriptions (empty = all match)."""
    moves = [game.moves(t, player) for player in range(game.n_players)]
    new, _ = simulate_turn(make_state(game, t), moves)
    want_p, want_f = game.planets(t + 1), game.fleets(t + 1)
    got_p, got_f = new["planets"], new["fleets"]
    problems = []

    if not np.array_equal(got_p[:, P_ID], want_p[:, P_ID]):
        problems.append(f"t={t} planet ids {got_p[:, P_ID]} != {want_p[:, P_ID]}")
    else:
        for col, name in ((P_OWNER, "owner"), (P_SHIPS, "ships")):
            bad = np.flatnonzero(got_p[:, col] != want_p[:, col])
            for i in bad:
                problems.append(f"t={t} planet {int(want_p[i, P_ID])} {name} {got_p[i, col]} != {want_p[i, col]}")
        err = np.abs(got_p[:, P_X:P_Y + 1] - want_p[:, P_X:P_Y + 1]).max(initial=0)
        if err > POSITION_TOLERANCE:
            problems.append(f"t={t} planet position error {err:.2e}")

    if not np.array_equal(got_f[:, F_ID], want_f[:, F_ID]):
        missing = sorted(set(want_f[:, F_ID].astype(int)) - set(got_f[:, F_ID].astype(int)))
        extra = sorted(set(got_f[:, F_ID].astype(int)) - set(want_f[:, F_ID].astype(int)))
        problems.append(f"t={t} fleet ids differ: recorded only {missing}, simulated only {extra}")
    else:
        for col, name in ((F_OWNER, "owner"), (F_SHIPS, "ships"), (F_FROM, "from")):
            bad = np.flatnonzero(got_f[:, col] != want_f[:, col])
            for i in bad:
                problems.append(f"t={t} fleet {int(want_f[i, F_ID])} {name} {got_f[i, col]} != {want_f[i, col]}")
        err = np.abs(got_f[:, F_X:F_Y + 1] - want_f[:, F_X:F_Y + 1]).max(initial=0)
        if err > POSITION_TOLERANCE:
            problems.append(f"t={t} fleet position error {err:.2e}")

    rows = game.action_rows(t)
    want_ids = game.arrays["action_fleet"][rows]
    got_ids = np.array(new["launched"], dtype=np.int64)
    if not np.array_equal(got_ids, want_ids):
        problems.append(f"t={t} launched fleet ids {got_ids} != {want_ids}")
    if not np.array_equal(got_ids >= 0, game.arrays["action_valid"][rows]):
        problems.append(f"t={t} order validity differs")
    return problems


def check_game(path):
    """Returns (turns checked, turns skipped, mismatch descriptions) for one game."""
    game = Replay(path)
    skip = spawn_turns(game)
    checked, problems = 0, []
    for t in range(len(game) - 1):
        if t in skip:
            continue
        checked += 1
        problems += [f"{path.name} {p}" for p in compare_turn(game, t)]
    skipped = len([t for t in skip if 0 <= t < len(game) - 1])
    return checked, skipped, problems


@pytest.mark.replays
def test_simulator_reproduces_recorded_games():
    games = sample_games()
    checked = skipped = 0
    problems = []
    for path in games:
        c, s, p = check_game(path)
        checked, skipped, problems = checked + c, skipped + s, problems + p
    print(f"\nTest 1: {len(games)} games, {checked} turns compared, {skipped} comet-spawn turns skipped, "
          f"{len(problems)} mismatches")
    for p in problems[:50]:
        print("  ", p)
    assert len(games) >= 200
    assert not problems
