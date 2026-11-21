"""Test 4: planet tokens and the global token, from each player's point of view.

Needs forecast files for a few games: `python precompute.py --games 20`.

    python -m pytest tests/test_features.py -s     # also prints one planet's forecast table
"""

import numpy as np
import pytest

from conftest import REPLAY_ROOT, read_index
from features import (CURRENT_FEATS, GLOBAL_FEATS, INCOMING_FEATS, N_BUCKETS, N_PLAYER_CLASSES, N_SLOTS,
                      PLANET_FEATS, forecast_table, global_token, planet_tokens, start_turns)
from precompute import forecast_path, load_forecast
from replay import Replay

ME, ENEMY_OPPOSITE = 0, 2   # columns of the incoming block: me, enemy+1, enemy+2 (opposite), enemy+3


def games_with_forecast(n_players=None, start=None):
    rows = [r for r in read_index() if forecast_path(REPLAY_ROOT, r["file"]).exists()]
    if n_players:
        rows = [r for r in rows if len(r["agents"]) == n_players]
    if start:
        rows = [r for r in rows if r["start"] == start]
    if not rows:
        pytest.skip("no forecast files yet: run python precompute.py --games 20")
    return rows


def load(row):
    return Replay(REPLAY_ROOT / row["file"]), load_forecast(forecast_path(REPLAY_ROOT, row["file"]))


def incoming_block(planets):
    """[44, 13, 4] incoming block of the planet tokens (log-scaled ships per bucket and class)."""
    return planets[:, CURRENT_FEATS:CURRENT_FEATS + INCOMING_FEATS].reshape(-1, N_BUCKETS, N_PLAYER_CLASSES)


@pytest.mark.replays
def test_shapes_no_nan_and_zero_padding():
    for row in games_with_forecast()[:5]:
        game, fc = load(row)
        for t in range(0, len(game), 17):
            for player in range(game.n_players):
                planets, pmask, pids = planet_tokens(game, fc, t, player)
                g = global_token(game, fc, t, player)
                assert planets.shape == (N_SLOTS, PLANET_FEATS) == (44, 146)
                assert g.shape == (GLOBAL_FEATS,) == (20,)
                assert pmask.shape == pids.shape == (44,)
                assert np.isfinite(planets).all() and np.isfinite(g).all()
                n = len(game.planets(t))
                assert pmask.sum() == n and pmask[:n].all()
                assert (planets[~pmask] == 0).all() and (pids[~pmask] == -1).all()
                assert (np.diff(pids[:n]) > 0).all()   # sorted by id


@pytest.mark.replays
def test_two_player_views_mirror_each_other():
    row = games_with_forecast(n_players=2, start="standard")[0]
    game, fc = load(row)
    turns = start_turns(game.planets(0), 2)
    assert (turns[1] - turns[0]) % 4 == 2   # the two starts are opposite each other

    for t in range(0, len(game), 7):
        p0, m0, ids0 = planet_tokens(game, fc, t, 0)
        p1, m1, ids1 = planet_tokens(game, fc, t, 1)
        assert np.array_equal(ids0, ids1)

        # Rotated views: the same set of planet positions (the map is symmetric).
        xy0 = np.round(p0[m0, :2], 3)
        xy1 = np.round(p1[m1, :2], 3)
        assert np.array_equal(xy0[np.lexsort(xy0.T)], xy1[np.lexsort(xy1.T)]), f"t={t}"

        # My incoming ships are the other player's "opposite enemy" incoming ships, planet by planet.
        in0, in1 = incoming_block(p0), incoming_block(p1)
        assert np.array_equal(in0[:, :, ME], in1[:, :, ENEMY_OPPOSITE]), f"t={t}"
        assert np.array_equal(in1[:, :, ME], in0[:, :, ENEMY_OPPOSITE]), f"t={t}"
        assert not in0[:, :, [1, 3]].any()   # nobody in the quarter-turn seats in a 2-player game


@pytest.mark.replays
def test_print_one_forecast_table():
    row = games_with_forecast(n_players=4)[0]
    game, fc = load(row)
    # The state and planet with the most ships on the way, for an interesting table.
    best = max(
        ((fc["fc_incoming"][game._p[t] + i].sum(), t, int(pid))
         for t in range(len(game)) for i, pid in enumerate(game.planets(t)[:, 0])),
    )
    _, t, pid = best
    print(f"\n{row['file']} ({row['agents']})")
    print(forecast_table(game, fc, t, 0, pid))
