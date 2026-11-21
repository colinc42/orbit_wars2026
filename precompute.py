"""Run the forecast once for every state of every recorded game, and save it.

    python precompute.py --games 20             # the first 20 games, with timing
    python precompute.py                        # every game in replay_memory/index.jsonl
    python precompute.py --workers 12           # several games at once

The forecast does not depend on whose point of view it is seen from, so one run
per state is enough. Results go next to the games:

    replay_memory/forecast/000/g0000042.npz     for replay_memory/games/000/g0000042.npz

holding, stacked like the game's planet_* arrays (state t owns rows
planet_n[:t].sum() .. planet_n[:t+1].sum(), in the order of game.planets(t)):

    fc_incoming    (total_planet_rows, 13, n_players) int32
    fc_proj_owner  (total_planet_rows, 13) int8
    fc_proj_ships  (total_planet_rows, 13) int32
    fc_lost        (T, n_players) int32
    fc_unresolved  (T, n_players) int32

Games that already have a forecast file are skipped, so a run can be resumed.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from forecast import N_BUCKETS, forecast
from replay import Replay
from sim import make_state

ROOT = Path(__file__).resolve().parent
DEFAULT_REPLAYS = ROOT / "replay_memory"
REPORT_EVERY = 10   # seconds between progress lines


def forecast_path(replay_root, game_file):
    """replay_memory/games/000/g0000042.npz -> replay_memory/forecast/000/g0000042.npz"""
    game_file = Path(game_file)
    return Path(replay_root) / "forecast" / game_file.parent.name / game_file.name


def load_forecast(path):
    """The arrays saved by `forecast_game`, as a dict."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def forecast_game(game):
    """Forecast every state of a Replay. Returns the dict of arrays described above."""
    n_rows = int(game.arrays["planet_n"].sum())
    n_players = game.n_players
    out = {
        "fc_incoming": np.zeros((n_rows, N_BUCKETS, n_players), dtype=np.int32),
        "fc_proj_owner": np.zeros((n_rows, N_BUCKETS), dtype=np.int8),
        "fc_proj_ships": np.zeros((n_rows, N_BUCKETS), dtype=np.int32),
        "fc_lost": np.zeros((len(game), n_players), dtype=np.int32),
        "fc_unresolved": np.zeros((len(game), n_players), dtype=np.int32),
    }
    for t in range(len(game)):
        fc = forecast(make_state(game, t), n_players)
        rows = slice(game._p[t], game._p[t + 1])
        out["fc_incoming"][rows] = fc["incoming"]
        out["fc_proj_owner"][rows] = fc["proj_owner"]
        out["fc_proj_ships"][rows] = fc["proj_ships"]
        out["fc_lost"][t] = fc["lost"]
        out["fc_unresolved"][t] = fc["unresolved"]
    return out


def run_one(job):
    """Forecast one game file and save it. Returns (number of states, seconds)."""
    replay_root, game_file = job
    start = time.perf_counter()
    game = Replay(Path(replay_root) / game_file)
    arrays = forecast_game(game)
    path = forecast_path(replay_root, game_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)   # written atomically: an interrupted run never leaves half a file
    return len(game), time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_REPLAYS), help="replay memory folder")
    ap.add_argument("--games", type=int, help="only the first N games (by index)")
    ap.add_argument("--workers", type=int, default=1, help="games run in parallel (1 = plain loop)")
    args = ap.parse_args()

    root = Path(args.root)
    rows = [json.loads(line) for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda r: r["index"])
    if args.games:
        rows = rows[: args.games]
    todo = [(str(root), r["file"]) for r in rows if not forecast_path(root, r["file"]).exists()]
    total_states = sum(r["n_states"] for r in rows)
    print(f"{len(rows)} games ({total_states:,} states), {len(rows) - len(todo)} already done, "
          f"{len(todo)} to forecast on {args.workers} worker(s)")

    started = last_report = time.time()
    done_games = done_states = 0
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.imap_unordered(run_one, todo)
            for n_states, _ in results:
                done_games, done_states = done_games + 1, done_states + n_states
                if time.time() - last_report > REPORT_EVERY:
                    last_report = time.time()
                    print(f"  {done_games}/{len(todo)} games, {done_states / (last_report - started):.0f} states/s",
                          flush=True)
    else:
        for job in todo:
            n_states, seconds = run_one(job)
            done_games, done_states = done_games + 1, done_states + n_states
            print(f"  {done_games}/{len(todo)} {job[1]}: {n_states} states in {seconds:.1f} s", flush=True)

    elapsed = time.time() - started
    if done_states:
        print(f"Done: {done_games} games, {done_states:,} states in {elapsed:.1f} s "
              f"({1000 * elapsed / done_states:.2f} ms per state)")


if __name__ == "__main__":
    main()
