"""Generate a replay memory: thousands of fully recorded Orbit Wars games.


    python generate_replays.py                         # 15,000 games or 2 GB, whichever comes first
    python generate_replays.py --games 200 --out replay_memory_small
    python generate_replays.py --max-gb 1 --workers 8
    python generate_replays.py --plan 20               # just print the first 20 planned games


Games are played by the scripted agents in orbit_rl/scripted.py plus Kaggle's
built-in `random` and `starter`, in many different mixtures:


  players   40% two-player (15% of those an agent against itself),
            60% four-player (four different agents / two pairs / any four)
  noise     35% of games are played exactly as scripted; in the rest each
            player independently plays as written (40%), with a few random
            mistakes (40%), or with many (20%) - see scripted.noisy
  start     80% the normal opening, 20% a random made-up mid-game position
            (planets already handed out with random garrisons), so the data
            covers big, contested positions and not only openings


Game i uses map seed `--seed + i` and its plan depends only on (--seed, i), so
a run can be interrupted and resumed, and any single game can be regenerated
exactly. Output:


  <out>/games/000/g0000042.npz   one game per file (format: orbit_rl/replay.py)
  <out>/index.jsonl              one line per game: agents, noise, result, size
  <out>/summary.json             totals, written at the end


The --max-gb limit is strict: a game that would take the folder over it is
deleted and the run stops.
"""


import argparse
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path


from replay import play_recorded, save_replay


ROOT = Path(__file__).resolve().parent


# How often each agent is drawn into a game (relative weights).
AGENT_WEIGHTS = {
    "nearest-sniper": 3.0,
    "lead-sniper": 3.0,
    "weakest-first": 2.0,
    "economist": 2.0,
    "hunter": 2.0,
    "doomstack": 2.0,
    "gang-up": 2.0,
    "comet-chaser": 2.0,
    "turtle": 1.5,
    "trickler": 1.5,
    "sweeper": 1.0,
    "starter": 1.0,
    "random": 0.7,
    "passive": 0.5,
}




def plan_game(index, base_seed=0):
    """Decide who plays game `index`, and how. Depends only on its arguments."""
    rng = random.Random(f"plan-{base_seed}-{index}")
    names, weights = list(AGENT_WEIGHTS), list(AGENT_WEIGHTS.values())


    def distinct(k):
        chosen = []
        while len(chosen) < k:
            name = rng.choices(names, weights)[0]
            if name not in chosen:
                chosen.append(name)
        return chosen


    r = rng.random()
    if rng.random() < 0.4:
        agents = distinct(1) * 2 if r < 0.15 else distinct(2)
    elif r < 0.5:
        agents = distinct(4)
    elif r < 0.7:
        agents = distinct(2) * 2
        rng.shuffle(agents)
    else:
        agents = rng.choices(names, weights, k=4)


    clean_game = rng.random() < 0.35
    noise = []
    for _ in agents:
        r = rng.random()
        if clean_game or r < 0.4:
            noise.append(0.0)
        elif r < 0.8:
            noise.append(round(rng.uniform(0.02, 0.15), 3))
        else:
            noise.append(round(rng.uniform(0.15, 0.5), 3))


    return {
        "index": index,
        "seed": base_seed + index,
        "agents": agents,
        "noise": noise,
        "start": "scrambled" if rng.random() < 0.2 else "standard",
    }




def game_path(out, index):
    return Path(out) / "games" / f"{index // 1000:03d}" / f"g{index:07d}.npz"




def run_game(job):
    """Worker: play and save one game, return its index line."""
    plan, out = job
    arrays = play_recorded(plan["agents"], plan["seed"], plan["noise"], plan["start"])
    path = game_path(out, plan["index"])
    size = save_replay(path, arrays)
    meta = json.loads(str(arrays["meta"]))
    return {
        **plan,
        "file": path.relative_to(out).as_posix(),
        "bytes": size,
        "n_states": meta["n_states"],
        "orders": int(len(arrays["action_step"])),
        "rewards": meta["rewards"],
        "final_scores": meta["final_scores"],
        "winner": meta["winner"],
    }




def read_index(out):
    """Games already finished in `out` (for resuming), keyed by index."""
    done = {}
    index_file = out / "index.jsonl"
    if index_file.exists():
        for line in index_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if (out / row["file"]).exists():
                    done[row["index"]] = row
    return done




def summarise(rows):
    appearances, wins, noisy_seats = Counter(), Counter(), Counter()
    for row in rows:
        for seat, (agent, eps) in enumerate(zip(row["agents"], row["noise"])):
            appearances[agent] += 1
            wins[agent] += row["winner"] == seat
            noisy_seats[agent] += eps > 0
    return {
        "games": len(rows),
        "states": sum(r["n_states"] for r in rows),
        "orders": sum(r["orders"] for r in rows),
        "bytes": sum(r["bytes"] for r in rows),
        "two_player_games": sum(len(r["agents"]) == 2 for r in rows),
        "four_player_games": sum(len(r["agents"]) == 4 for r in rows),
        "scrambled_start_games": sum(r["start"] == "scrambled" for r in rows),
        "games_with_noise": sum(any(r["noise"]) for r in rows),
        "games_to_step_limit": sum(r["n_states"] >= 500 for r in rows),
        "agents": {
            name: {"seats": appearances[name], "wins": wins[name], "noisy_seats": noisy_seats[name]}
            for name in sorted(appearances)
        },
    }




def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=15000, help="number of games to have in the folder")
    ap.add_argument("--max-gb", type=float, default=2.0, help="hard limit on the folder's size")
    ap.add_argument("--out", default="replay_memory")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=0, help="base seed; game i uses map seed SEED + i")
    ap.add_argument("--plan", type=int, metavar="N", help="print the first N planned games and exit")
    args = ap.parse_args()


    if args.plan:
        for i in range(args.plan):
            p = plan_game(i, args.seed)
            players = "  ".join(f"{a}{'~' + str(e) if e else ''}" for a, e in zip(p["agents"], p["noise"]))
            print(f"{i:6d}  {p['start']:9s}  {players}")
        return


    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.rglob("*.tmp"):
        stale.unlink()
    budget = int(args.max_gb * 1024**3)
    done = read_index(out)
    used = sum(r["bytes"] for r in done.values())
    todo = [i for i in range(args.games) if i not in done]
    print(f"{len(done)} games ({used / 1024**2:.0f} MB) already in {out}; {len(todo)} to play "
          f"on {args.workers} workers, limit {args.max_gb:g} GB")


    largest = max([r["bytes"] for r in done.values()] + [2 * 1024**2])
    failures, full = 0, False
    started = last_report = time.time()
    new_states = 0
    queue = iter(todo)
    with ProcessPoolExecutor(args.workers) as pool, open(out / "index.jsonl", "a", encoding="utf-8") as index_file:
        pending = {}
        try:
            while True:
                # Only start a game if every game in flight could be the largest seen and still fit.
                while not full and len(pending) < 2 * args.workers:
                    if used + (len(pending) + 1) * largest > budget:
                        full = True
                        break
                    index = next(queue, None)
                    if index is None:
                        break
                    plan = plan_game(index, args.seed)
                    pending[pool.submit(run_game, (plan, out))] = plan
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    plan = pending.pop(future)
                    try:
                        row = future.result()
                    except Exception as e:  # one bad game shouldn't end the run
                        failures += 1
                        print(f"  game {plan['index']} failed: {e!r}")
                        continue
                    if used + row["bytes"] > budget:
                        (out / row["file"]).unlink()
                        full = True
                        continue
                    used += row["bytes"]
                    largest = max(largest, row["bytes"])
                    new_states += row["n_states"]
                    done[row["index"]] = row
                    index_file.write(json.dumps(row) + "\n")
                    index_file.flush()
                if time.time() - last_report > 10:
                    last_report = time.time()
                    rate = (len(done) - (args.games - len(todo))) / (last_report - started)
                    print(f"  {len(done):6d}/{args.games} games  {used / 1024**2:7.0f} MB  "
                          f"{rate:5.1f} games/s  {new_states / (last_report - started):6.0f} states/s", flush=True)
        except KeyboardInterrupt:
            print("Interrupted - finished games are kept; run again to resume.")
            for future in pending:
                future.cancel()


    summary = summarise([done[i] for i in sorted(done)])
    summary.update(seed=args.seed, max_gb=args.max_gb, failures=failures, seconds=round(time.time() - started, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


    print(f"\n{summary['games']} games, {summary['states']:,} states, {summary['orders']:,} orders, "
          f"{summary['bytes'] / 1024**3:.2f} GB in {out}")
    print(f"  2-player {summary['two_player_games']}, 4-player {summary['four_player_games']}, "
          f"scrambled start {summary['scrambled_start_games']}, with noise {summary['games_with_noise']}, "
          f"reached step limit {summary['games_to_step_limit']}")
    if full:
        print(f"  Stopped at the {args.max_gb:g} GB limit before reaching {args.games} games.")
    if failures:
        print(f"  {failures} games failed (see messages above).")
    print(f"  {'agent':16s} seats   wins")
    for name, s in summary["agents"].items():
        print(f"  {name:16s} {s['seats']:5d}  {s['wins']:5d}")




if __name__ == "__main__":
    main()
