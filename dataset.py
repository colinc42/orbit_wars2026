"""Training samples from the replay memory: one player's view of one state.

A sample is (game, t, player) turned into fixed-size arrays:

  planets       [44, 146] float32   planet tokens (features.planet_tokens)
  pmask         [44] bool           True for real planets
  global        [20] float32        global token (features.global_token)
  win           float32             1 if this player won the game, else 0
  future_share  float32             this player's share of all ships SHARE_AHEAD turns
                                    later (or at the last state, if the game ends sooner)

Every sample has the same shapes, so a batch is just np.stack; there is no
padding of fleets and no fleet mask. Games are split by index: index % 10 == 0
is validation, the rest is training.

    from torch.utils.data import DataLoader
    loader = DataLoader(ForecastDataset("replay_memory", "train"), batch_size=256,
                        collate_fn=collate, num_workers=4)
    batch = next(iter(loader))     # dict of tensors, each with a leading batch dimension
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from features import global_token, planet_tokens
from precompute import forecast_path, load_forecast
from replay import Replay
from sim import make_state

VALIDATION_EVERY = 10    # index % 10 == 0 -> validation
STATES_PER_GAME = 16     # (t, player) samples drawn from each game read
BUFFER_SIZE = 4096       # shuffle buffer: samples from many games are mixed before they come out
SHARE_AHEAD = 50         # turns ahead for the future_share target


def split_games(replay_root, split):
    """Game files of a split ('train' or 'validation') that have a forecast file."""
    root = Path(replay_root)
    rows = [json.loads(line) for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    want_validation = split == "validation"
    return [
        r["file"] for r in sorted(rows, key=lambda r: r["index"])
        if (r["index"] % VALIDATION_EVERY == 0) == want_validation and forecast_path(root, r["file"]).exists()
    ]


def make_sample(game, fc, t, player):
    """The arrays of one sample (see the module docstring)."""
    planets, pmask, _ = planet_tokens(game, fc, t, player, state=make_state(game, t))
    score = game.arrays["score"][min(t + SHARE_AHEAD, len(game) - 1)]
    return {
        "planets": planets,
        "pmask": pmask,
        "global": global_token(game, fc, t, player),
        "win": np.float32(game.meta["rewards"][player] == 1),
        "future_share": np.float32(score[player] / max(score.sum(), 1)),
    }


def collate(samples):
    """List of samples -> dict of tensors, stacked along a new first (batch) axis."""
    return {key: torch.from_numpy(np.stack([s[key] for s in samples])) for key in samples[0]}


class ForecastDataset(IterableDataset):
    """An endless stream of shuffled samples.

    Reads a random game, draws STATES_PER_GAME random (t, player) pairs from it, and
    puts them in a shuffle buffer; once the buffer is full, every new sample pushes
    out a random old one. With several DataLoader workers, each worker reads games
    from its own share of the list.
    """

    def __init__(self, replay_root, split="train", states_per_game=STATES_PER_GAME,
                 buffer_size=BUFFER_SIZE, seed=0):
        self.root = Path(replay_root)
        self.files = split_games(self.root, split)
        if not self.files:
            raise ValueError(f"no {split} games with a forecast under {self.root} (run precompute.py)")
        self.states_per_game = states_per_game
        self.buffer_size = buffer_size
        self.seed = seed

    def samples_from_one_game(self, rng, files):
        game_file = rng.choice(files)
        game = Replay(self.root / game_file)
        fc = load_forecast(forecast_path(self.root, game_file))
        for _ in range(self.states_per_game):
            t = rng.randrange(len(game))
            player = rng.randrange(game.n_players)
            yield make_sample(game, fc, t, player)

    def __iter__(self):
        worker = get_worker_info()
        worker_id, n_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        rng = random.Random(f"{self.seed}-{worker_id}")
        files = self.files[worker_id::n_workers] or self.files

        buffer = []
        while True:
            for sample in self.samples_from_one_game(rng, files):
                if len(buffer) < self.buffer_size:
                    buffer.append(sample)
                    continue
                i = rng.randrange(len(buffer))
                yield buffer[i]
                buffer[i] = sample
