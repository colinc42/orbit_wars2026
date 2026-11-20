"""Run Orbit Wars games without the Kaggle wrapper's per-step overhead.

`FastGame` calls the official `interpreter` function from kaggle_environments
directly, so the rules are the competition's own code. It skips what the
wrapper adds around it (schema validation, deep copies of the whole state
every step, stdout capture), which is most of the time a game takes.
tests/test_replays.py checks that a FastGame and a Kaggle `env.run` with the
same seed and agents produce identical states.

Agents here must be callables `agent(obs, config=None)`; `as_callable`
converts the specs the rest of the project uses. The planet and fleet lists in
an observation are the live game state, so agents must not modify them.
"""

import contextlib
import io
import random

with contextlib.redirect_stdout(io.StringIO()):  # hide "Loading environment ... failed" lines
    from kaggle_environments.envs.orbit_wars import orbit_wars as ow
    from kaggle_environments.utils import Struct

from agents import resolve_agent

DEFAULT_CONFIG = {"episodeSteps": 500, "actTimeout": 1, "shipSpeed": 6.0, "cometSpeed": 4.0}


def as_callable(spec, device="cpu"):
    """Like agents.resolve_agent, but always returns `agent(obs, config=None)`."""
    if spec in ow.agents:
        builtin = ow.agents[spec]
        return lambda obs, config=None: builtin(obs)
    agent = resolve_agent(spec, device=device)
    if not callable(agent):
        raise ValueError(f"FastGame cannot run agent spec {spec!r} (agent files need the Kaggle wrapper)")
    return agent


class _Env:
    """The parts of kaggle's Environment object the interpreter reads."""

    def __init__(self, configuration):
        self.configuration = Struct(**configuration)
        self.info = {}
        self.done = True


class FastGame:
    def __init__(self, n_players=2, seed=None, **config):
        if n_players not in (2, 4):
            raise ValueError("Orbit Wars is a 2 or 4 player game")
        if seed is None:
            seed = random.randrange(2**31)
        self.n_players = n_players
        self.seed = seed
        self.env = _Env({**DEFAULT_CONFIG, **config, "seed": seed})
        self.state = [
            Struct(
                action=[],
                reward=0,
                info=Struct(),
                status="ACTIVE",
                observation=Struct(remainingOverageTime=60, step=0, player=i, planets=[], fleets=[]),
            )
            for i in range(n_players)
        ]
        self.state = ow.interpreter(self.state, self.env)  # builds the map
        self.env.done = False
        self.step_count = 0
        self.done = False

    @property
    def shared(self):
        """Player 0's observation: holds the state every player sees."""
        return self.state[0].observation

    @property
    def planets(self):
        return self.shared.planets

    @property
    def fleets(self):
        return self.shared.fleets

    @property
    def rewards(self):
        return [s.reward for s in self.state]

    def observation(self, player):
        return self.state[player].observation

    def scores(self):
        """Ships on planets + in fleets, per player (the game's final score)."""
        scores = [0] * self.n_players
        for p in self.planets:
            if p[1] != -1:
                scores[p[1]] += p[5]
        for f in self.fleets:
            scores[f[1]] += f[6]
        return scores

    def step(self, actions):
        """Advance one turn. `actions[i]` is player i's list of moves."""
        if self.done:
            raise RuntimeError("Game is over")
        for s, action in zip(self.state, actions):
            s.action = action
        self.state = ow.interpreter(self.state, self.env)
        self.step_count += 1
        for s in self.state:
            s.observation.step = self.step_count
        self.done = (
            any(s.status != "ACTIVE" for s in self.state)
            or self.step_count >= self.env.configuration.episodeSteps - 1
        )
        self.env.done = self.done
        return self.done

    def run(self, agents):
        """Play to the end with `agents` (callables). Returns final rewards."""
        while not self.done:
            self.step([agent(self.observation(i), self.env.configuration) for i, agent in enumerate(agents)])
        return self.rewards
