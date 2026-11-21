"""Test 2: the forecast agrees with what actually happened.

Fleets never change course, so for every fleet alive at state t, the turn the
forecast says it disappears, and how, must match the recording, whatever was
launched later. The one exception is a comet spawned after t that gets in the
way: mismatches where the recording shows the fleet hitting such a comet are
reported separately and don't count against the 99.9% requirement.

How a fleet disappeared is read off the recording independently of the forecast:
its last recorded position and heading, the planets' recorded positions on that
turn, the board edge and the sun.

    python -m pytest tests/test_forecast.py -s       # prints the report
"""

import math

import numpy as np
import pytest

from conftest import sample_games
from forecast import HORIZON, forecast
from replay import Replay
from sim import (BOARD_SIZE, CENTER, F_ANGLE, F_ID, F_OWNER, F_SHIPS, F_X, F_Y, P_ID, P_RADIUS, P_X, P_Y,
                 SUN_RADIUS, fleet_speed, make_state, point_to_segment_distance, swept_pair_hit)

GAMES_PER_GROUP = 5        # 8 groups -> 40 games, every state of each
TRUTH_MARGIN = 1e-3        # extra reach when reading collisions off float32 recordings (error ~1e-5)
REQUIRED_MATCH = 0.999


def last_seen(game):
    """fleet id -> (last state it appears in, its row in the game's fleet_* arrays)."""
    state_of_row = np.repeat(np.arange(len(game)), game.arrays["fleet_n"])
    out = {}
    for row, (fid, t) in enumerate(zip(game.arrays["fleet_id"], state_of_row)):
        out[int(fid)] = (int(t), row)
    return out


def recorded_fate(game, t_last, fleet):
    """How the fleet (a row of game.fleets(t_last)) disappeared on the turn after t_last.

    Returns (kind, planet ids it could have hit). Checks planets first, then the edge,
    then the sun, like the engine.
    """
    old = (fleet[F_X], fleet[F_Y])
    speed = fleet_speed(fleet[F_SHIPS])
    new = (old[0] + math.cos(fleet[F_ANGLE]) * speed, old[1] + math.sin(fleet[F_ANGLE]) * speed)
    after_xy = {int(p[P_ID]): (p[P_X], p[P_Y]) for p in game.planets(t_last + 1)}
    hit = []
    for p in game.planets(t_last):
        p0 = (p[P_X], p[P_Y])
        p1 = after_xy.get(int(p[P_ID]), p0)   # a comet that just left stayed put
        if swept_pair_hit(old, new, p0, p1, p[P_RADIUS] + TRUTH_MARGIN):
            hit.append(int(p[P_ID]))
    if hit:
        return "planet", hit
    if not (0 <= new[0] <= BOARD_SIZE and 0 <= new[1] <= BOARD_SIZE):
        return "edge", []
    if point_to_segment_distance((CENTER, CENTER), old, new) < SUN_RADIUS + TRUTH_MARGIN:
        return "sun", []
    return "unknown", []


def check_game(path):
    """Compare forecast and recording for every fleet at every state of one game.

    Returns (pairs compared, mismatches), each mismatch being
    (hit_new_comet, game name, fleet id, t, description).
    """
    game = Replay(path)
    T = len(game)
    a = game.arrays
    seen = last_seen(game)
    fleet_rows = np.stack([a[f"fleet_{c}"] for c in ("id", "owner", "x", "y", "angle", "from", "ships")], 1)
    fleet_rows = fleet_rows.astype(np.float64)
    compared, problems = 0, []

    for t in range(T - 1):
        state = make_state(game, t)
        if len(state["fleets"]) == 0:
            continue
        fc = forecast(state, game.n_players)
        predicted = {e[0]: e for e in fc["events"]}   # fleet id -> (id, owner, ships, kind, pid, k)
        new_comets = set(a["comet_ids"][a["comet_first_step"] > t].ravel().tolist())   # spawned after t

        # Every fleet's ships end up in exactly one of incoming / lost / unresolved.
        total = fc["incoming"].sum((0, 1)) + fc["lost"] + fc["unresolved"]
        expected = np.bincount(state["fleets"][:, F_OWNER].astype(int), state["fleets"][:, F_SHIPS],
                               minlength=game.n_players)
        assert np.array_equal(total, expected), f"{path.name} t={t}: ships not conserved"

        for fleet in state["fleets"]:
            fid = int(fleet[F_ID])
            t_last, row = seen[fid]
            event = predicted.get(fid)
            k_pred = event[5] if event else None
            # A fleet still flying at the end of the game: we only know it lasted that long.
            if t_last == T - 1 and (k_pred is None or t + k_pred > T - 1):
                continue
            compared += 1
            k_true = t_last + 1 - t
            kind, candidates = "?", []
            if k_true <= HORIZON:
                kind, candidates = recorded_fate(game, t_last, fleet_rows[row])

            if k_true > HORIZON:
                ok, why = event is None, f"forecast {event and event[3]} after {k_pred} turns, really {k_true}"
            elif event is None:
                ok, why = False, f"forecast still flying, recording {kind} {candidates} after {k_true} turns"
            else:
                ok = k_pred == k_true and kind == event[3] and (kind != "planet" or event[4] in candidates)
                why = (f"forecast {event[3]} {event[4]} after {k_pred} turns, "
                       f"recording {kind} {candidates} after {k_true} turns")
            if not ok:
                hit_new_comet = kind == "planet" and bool(new_comets & set(candidates))
                problems.append((hit_new_comet, path.name, fid, t, why))
    return compared, problems


def by_fleet(problems):
    """One line per mismatched fleet: the states it was wrong at, and the last description."""
    fleets = {}
    for _, game, fid, t, why in problems:
        fleets.setdefault((game, fid), []).append((t, why))
    return [f"{game} fleet {fid} at t={rows[0][0]}..{rows[-1][0]} ({len(rows)} states): {rows[-1][1]}"
            for (game, fid), rows in fleets.items()]


@pytest.mark.replays
def test_forecast_matches_recording():
    games = sample_games(GAMES_PER_GROUP)
    compared, problems = 0, []
    for path in games:
        c, p = check_game(path)
        compared += c
        problems += p
    comet = [p for p in problems if p[0]]
    other = [p for p in problems if not p[0]]
    match = 1 - len(other) / max(compared, 1)
    print(f"\nTest 2: {len(games)} games, {compared} (state, fleet) pairs compared, {len(problems)} mismatches: "
          f"{len(comet)} where the fleet hit a comet spawned after t, {len(other)} others "
          f"({match:.5%} match leaving out the new-comet hits)")
    for label, group in (("OTHER", other), ("new comet", comet)):
        for line in by_fleet(group):
            print(f"   {label}: {line}")
    assert match >= REQUIRED_MATCH
