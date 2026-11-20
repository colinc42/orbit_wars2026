"""Record full games to disk and read them back.

One game = one compressed .npz file holding every state, every order every
player gave, and the comet paths. Nothing is summarised or encoded: this is
the raw game, so you can build whatever model inputs you like from it later.

A game with T states has indices t = 0 .. T-1. The orders recorded at t are
what the players sent after seeing state t; they produce state t+1.

Arrays in the file (rows for all steps are concatenated; `planet_n[t]` and
`fleet_n[t]` say how many rows belong to state t):

  planet_n       (T,)    planets on the board at each state (comets come and go)
  planet_id, planet_owner (-1 = neutral), planet_x, planet_y, planet_radius,
  planet_ships, planet_prod, planet_is_comet
  fleet_n        (T,)    fleets in flight at each state
  fleet_id, fleet_owner, fleet_x, fleet_y, fleet_angle, fleet_from, fleet_ships
  score          (T, players)  ships on planets + in fleets (the game's score)

  action_step, action_player, action_planet, action_angle, action_ships
                 one row per order, sorted by step then player
  action_valid   False if the game ignored the order (not your planet, or more
                 ships than it holds)
  action_fleet   id of the fleet the order launched (-1 if ignored); matches
                 fleet_id, though a fleet that lands on its first tick never
                 appears in a state

  comet_first_step (G,)       first state each comet group is on the board
  comet_ids        (G, 4)     planet ids of the group's four comets
  comet_path_len   (G,)       number of positions in each path
  comet_paths      (G, 4, L, 2)  x, y of every position, NaN-padded. At state t
                              a comet is at paths[g, k, t - comet_first_step[g]]

  meta           JSON string: seed, agents, noise, start, angular_velocity,
                 rewards, final_scores, ... (see `play_recorded`)

Positions are stored as float32; ids, owners and ship counts are exact. To get
the exact float64 game back, re-run it: `play_recorded` with the same
arguments reproduces the game bit for bit.

Reading:

    from orbit_rl.replay import Replay, iter_replays
    game = Replay("replay_memory/games/000/g0000042.npz")
    game.meta["agents"], len(game)
    game.planets(t)             # (n, 7) [id, owner, x, y, radius, ships, production]
    game.fleets(t)              # (m, 7) [id, owner, x, y, angle, from_planet_id, ships]
    game.moves(t, player)       # [[planet_id, angle, ships], ...] as the agent sent them
    game.observation(t, player) # Kaggle-style observation dict (works with features.encode)
    game.arrays                 # the raw arrays above, for vectorised use
"""

import json
import os
import random
from pathlib import Path

import numpy as np

from scripted import noisy

FORMAT_VERSION = 1
PLANET_COLUMNS = ("id", "owner", "x", "y", "radius", "ships", "prod")
FLEET_COLUMNS = ("id", "owner", "x", "y", "angle", "from", "ships")
PLANET_DTYPES = (np.int16, np.int8, np.float32, np.float32, np.float32, np.int32, np.int8)
FLEET_DTYPES = (np.int32, np.int8, np.float32, np.float32, np.float32, np.int16, np.int32)


class Recorder:
    """Collects the states and orders of one FastGame as it is played."""

    def __init__(self, game):
        self.game = game
        self.n_base_planets = len(game.planets)
        self.planet_rows, self.planet_n = [], []
        self.fleet_rows, self.fleet_n = [], []
        self.scores = []
        self.action_rows = []
        self.comet_groups = []
        self._next_fleet_id = None

    def snapshot(self):
        """Record the current state. Call once per state, including the last."""
        game = self.game
        self.planet_rows.extend(p[:] for p in game.planets)
        self.planet_n.append(len(game.planets))
        self.fleet_rows.extend(f[:] for f in game.fleets)
        self.fleet_n.append(len(game.fleets))
        self.scores.append(game.scores())
        for group in game.shared.comets:
            if group["path_index"] == 0:  # a group is first visible on its first path position
                self.comet_groups.append(
                    {"first_step": game.step_count, "ids": list(group["planet_ids"]), "paths": group["paths"]}
                )

    def record_actions(self, actions):
        """Record the orders about to be passed to game.step, working out
        which ones the game will accept the same way the game does."""
        game = self.game
        holdings = {p[0]: [p[1], p[5]] for p in game.planets}
        next_id = game.shared.next_fleet_id
        for player, moves in enumerate(actions):
            for move in moves or []:
                if len(move) != 3:
                    continue
                pid, angle, ships = move[0], float(move[1]), int(move[2])
                slot = holdings.get(pid)
                valid = slot is not None and slot[0] == player and 0 < ships <= slot[1]
                fleet_id = -1
                if valid:
                    slot[1] -= ships
                    fleet_id = next_id
                    next_id += 1
                self.action_rows.append((game.step_count, player, pid, angle, ships, valid, fleet_id))
        self._next_fleet_id = next_id

    def check(self):
        """After game.step: confirm the game launched exactly the fleets
        record_actions predicted."""
        if self.game.shared.next_fleet_id != self._next_fleet_id:
            raise RuntimeError("Recorder and game disagree about which orders were valid")

    def arrays(self, meta):
        out = {"meta": np.array(json.dumps(meta))}
        planets = np.array(self.planet_rows, dtype=np.float64).reshape(-1, 7)
        fleets = np.array(self.fleet_rows, dtype=np.float64).reshape(-1, 7)
        for i, (name, dtype) in enumerate(zip(PLANET_COLUMNS, PLANET_DTYPES)):
            out[f"planet_{name}"] = planets[:, i].astype(dtype)
        out["planet_is_comet"] = out["planet_id"] >= self.n_base_planets
        out["planet_n"] = np.array(self.planet_n, dtype=np.int16)
        for i, (name, dtype) in enumerate(zip(FLEET_COLUMNS, FLEET_DTYPES)):
            out[f"fleet_{name}"] = fleets[:, i].astype(dtype)
        out["fleet_n"] = np.array(self.fleet_n, dtype=np.int32)
        out["score"] = np.array(self.scores, dtype=np.int32)

        a = self.action_rows
        out["action_step"] = np.array([r[0] for r in a], dtype=np.int16)
        out["action_player"] = np.array([r[1] for r in a], dtype=np.int8)
        out["action_planet"] = np.array([r[2] for r in a], dtype=np.int16)
        out["action_angle"] = np.array([r[3] for r in a], dtype=np.float64)
        out["action_ships"] = np.array([r[4] for r in a], dtype=np.int32)
        out["action_valid"] = np.array([r[5] for r in a], dtype=bool)
        out["action_fleet"] = np.array([r[6] for r in a], dtype=np.int32)

        g = self.comet_groups
        longest = max((len(c["paths"][0]) for c in g), default=0)
        paths = np.full((len(g), 4, longest, 2), np.nan, dtype=np.float32)
        for i, c in enumerate(g):
            for k, path in enumerate(c["paths"]):
                paths[i, k, : len(path)] = path
        out["comet_first_step"] = np.array([c["first_step"] for c in g], dtype=np.int16)
        out["comet_ids"] = np.array([c["ids"] for c in g], dtype=np.int16).reshape(-1, 4)
        out["comet_path_len"] = np.array([len(c["paths"][0]) for c in g], dtype=np.int16)
        out["comet_paths"] = paths
        return out


def scramble_start(game, rng):
    """Replace the opening position with a made-up mid-game one: a random
    share of the neutral planets is handed out to the players (unevenly) with
    random garrisons. Must be called before the first step."""
    n = game.n_players
    developed = rng.uniform(0.15, 0.9)
    strength = [rng.uniform(0.3, 1.0) for _ in range(n)]
    for p in game.planets:
        if p[1] != -1:
            p[5] = rng.randint(5, 150)
        elif rng.random() < developed:
            p[1] = rng.choices(range(n), weights=strength)[0]
            p[5] = max(1, min(500, int(rng.lognormvariate(3.0, 1.0) * strength[p[1]])))


def play_recorded(agents, seed, noise=None, start="standard"):
    """Play one full game and return its arrays (see the module docstring).

    agents: 2 or 4 agent specs (scripted names, 'random', 'starter', ...)
    seed:   map seed; with the other arguments it fully determines the game
    noise:  per-player mistake rate for scripted.noisy (0 = play as written)
    start:  'standard', or 'scrambled' for a random mid-game position
    """
    from fastsim import FastGame, as_callable  # imported here so reading replays doesn't need kaggle

    noise = list(noise) if noise is not None else [0.0] * len(agents)
    random.seed(seed)  # kaggle's built-in 'random' agent draws from the global RNG
    game = FastGame(len(agents), seed=seed)
    if start == "scrambled":
        scramble_start(game, random.Random(f"scramble-{seed}"))
    elif start != "standard":
        raise ValueError(f"Unknown start {start!r}")
    players = [
        noisy(as_callable(spec), eps, f"noise-{seed}-{i}") if eps > 0 else as_callable(spec)
        for i, (spec, eps) in enumerate(zip(agents, noise))
    ]

    recorder = Recorder(game)
    config = game.env.configuration
    while True:
        recorder.snapshot()
        if game.done:
            break
        actions = [player(game.observation(i), config) for i, player in enumerate(players)]
        recorder.record_actions(actions)
        game.step(actions)
        recorder.check()

    scores = game.scores()
    best = max(scores)
    meta = {
        "format_version": FORMAT_VERSION,
        "seed": seed,
        "n_players": game.n_players,
        "agents": list(agents),
        "noise": noise,
        "start": start,
        "n_states": game.step_count + 1,
        "n_base_planets": recorder.n_base_planets,
        "angular_velocity": game.shared.angular_velocity,
        "episode_steps": config.episodeSteps,
        "ship_speed": config.shipSpeed,
        "comet_speed": config.cometSpeed,
        "rewards": game.rewards,
        "final_scores": scores,
        "winner": scores.index(best) if best > 0 and scores.count(best) == 1 else -1,
    }
    return recorder.arrays(meta)


def save_replay(path, arrays):
    """Write atomically, so an interrupted run never leaves a half-written game."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return path.stat().st_size


class Replay:
    """One recorded game, loaded into memory."""

    def __init__(self, path):
        with np.load(path) as z:
            self.arrays = {k: z[k] for k in z.files}
        self.meta = json.loads(str(self.arrays.pop("meta")))
        self.n_players = self.meta["n_players"]
        a = self.arrays
        self._p = np.concatenate([[0], np.cumsum(a["planet_n"], dtype=np.int64)])
        self._f = np.concatenate([[0], np.cumsum(a["fleet_n"], dtype=np.int64)])
        self._a = np.searchsorted(a["action_step"], np.arange(len(a["planet_n"]) + 1))

    def __len__(self):
        """Number of states (one more than the number of turns played)."""
        return len(self.arrays["planet_n"])

    def _table(self, prefix, columns, lo, hi):
        return np.stack([self.arrays[f"{prefix}_{c}"][lo:hi].astype(np.float64) for c in columns], axis=1)

    def planets(self, t):
        """(n, 7) float64 rows [id, owner, x, y, radius, ships, production]."""
        return self._table("planet", PLANET_COLUMNS, self._p[t], self._p[t + 1])

    def fleets(self, t):
        """(m, 7) float64 rows [id, owner, x, y, angle, from_planet_id, ships]."""
        return self._table("fleet", FLEET_COLUMNS, self._f[t], self._f[t + 1])

    def action_rows(self, t):
        """Slice into the action_* arrays covering the orders given at state t."""
        return slice(self._a[t], self._a[t + 1])

    def moves(self, t, player):
        """What `player` sent at state t: [[planet_id, angle, ships], ...]."""
        a, rows = self.arrays, self.action_rows(t)
        return [
            [int(pid), float(angle), int(ships)]
            for who, pid, angle, ships in zip(
                a["action_player"][rows], a["action_planet"][rows], a["action_angle"][rows], a["action_ships"][rows]
            )
            if who == player
        ]

    def comets(self, t):
        """Comet groups on the board at state t, in the observation's format."""
        a = self.arrays
        groups = []
        for g, first in enumerate(a["comet_first_step"]):
            length = int(a["comet_path_len"][g])
            if first <= t < first + length:
                groups.append(
                    {
                        "planet_ids": a["comet_ids"][g].tolist(),
                        "paths": a["comet_paths"][g, :, :length].astype(np.float64).tolist(),
                        "path_index": int(t - first),
                    }
                )
        return groups

    def observation(self, t, player=0):
        """State t as the Kaggle observation dict `player` would have received.
        (`initial_planets` is omitted: it only repeats the starting positions,
        which are `planets(0)`.)"""
        a = self.arrays
        lo, hi = self._p[t], self._p[t + 1]
        planets = [
            [int(i), int(o), float(x), float(y), float(r), int(s), int(p)]
            for i, o, x, y, r, s, p in zip(*(a[f"planet_{c}"][lo:hi] for c in PLANET_COLUMNS))
        ]
        lo, hi = self._f[t], self._f[t + 1]
        fleets = [
            [int(i), int(o), float(x), float(y), float(ang), int(src), int(s)]
            for i, o, x, y, ang, src, s in zip(*(a[f"fleet_{c}"][lo:hi] for c in FLEET_COLUMNS))
        ]
        comets = self.comets(t)
        return {
            "player": player,
            "step": t,
            "angular_velocity": self.meta["angular_velocity"],
            "planets": planets,
            "fleets": fleets,
            "comets": comets,
            "comet_planet_ids": [pid for group in comets for pid in group["planet_ids"]],
        }


def iter_replays(root):
    """Yield every Replay under `root`, in game order."""
    for path in sorted(Path(root).rglob("g*.npz")):
        yield Replay(path)
