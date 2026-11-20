"""Gymnasium wrapper: one learning agent vs a fixed opponent.

Uses kaggle_environments' own Orbit Wars simulator via `env.train(...)`, so
the training game is exactly the competition game.
"""

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make

from actions import N_FRACTIONS, N_TARGETS, decode
from agents import resolve_agent
from features import MAX_PLANETS, N_GLOBAL_FEATURES, N_PLANET_FEATURES, encode, ship_totals


class OrbitWarsEnv(gym.Env):
    """Observation: the dict from `features.encode` (minus planet_ids).
    Action: dict(targets=int[MAX_PLANETS], fractions=int[MAX_PLANETS]); see
    actions.py. The raw kaggle observation is in info["raw_obs"].

    Reward: +1 / -1 / 0 at game end (the real objective), plus optional
    shaping `shaping * change in (my share of all player ships)` per step to
    give the agent a denser signal early in training. Set shaping=0 to train
    on the true reward only.
    """

    metadata = {"render_modes": ["html"]}

    def __init__(self, opponent="random", shaping=1.0, random_seat=True, debug=False):
        super().__init__()
        self.opponent_spec = opponent
        self.shaping = shaping
        self.random_seat = random_seat
        self.kaggle_env = make("orbit_wars", debug=debug)
        self.trainer = None
        self.seat = 0

        self.observation_space = spaces.Dict(
            {
                "planets": spaces.Box(-np.inf, np.inf, (MAX_PLANETS, N_PLANET_FEATURES), np.float32),
                "planet_mask": spaces.MultiBinary(MAX_PLANETS),
                "source_mask": spaces.MultiBinary(MAX_PLANETS),
                "globals": spaces.Box(-np.inf, np.inf, (N_GLOBAL_FEATURES,), np.float32),
            }
        )
        self.action_space = spaces.Dict(
            {
                "targets": spaces.MultiDiscrete([N_TARGETS] * MAX_PLANETS),
                "fractions": spaces.MultiDiscrete([N_FRACTIONS] * MAX_PLANETS),
            }
        )

    def _share(self, obs):
        mine, enemy = ship_totals(obs)
        return mine / max(mine + enemy, 1)

    def _wrap(self, obs):
        enc = encode(obs)
        enc.pop("planet_ids")
        return enc

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        self.seat = random.randint(0, 1) if self.random_seat else 0
        opponent = resolve_agent(self.opponent_spec)
        agents = [None, opponent] if self.seat == 0 else [opponent, None]
        self.trainer = self.kaggle_env.train(agents)
        self.raw_obs = self.trainer.reset()
        self.prev_share = self._share(self.raw_obs)
        return self._wrap(self.raw_obs), {"raw_obs": self.raw_obs, "seat": self.seat}

    def step(self, action):
        moves = decode(self.raw_obs, action["targets"], action["fractions"])
        return self.step_moves(moves)

    def step_moves(self, moves):
        """Step with raw env moves (useful for scripted or imitation agents)."""
        obs, reward, done, info = self.trainer.step(moves)
        self.raw_obs = obs
        share = self._share(obs)
        shaped = self.shaping * (share - self.prev_share)
        self.prev_share = share

        terminal = float(reward or 0.0) if done else 0.0
        info = dict(info or {})
        info.update(raw_obs=obs, seat=self.seat, terminal_reward=terminal)
        return self._wrap(obs), terminal + shaped, bool(done), False, info

    def render(self):
        return self.kaggle_env.render(mode="html")
