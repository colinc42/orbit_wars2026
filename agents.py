"""Turn short agent specs into kaggle agent callables.

Specs accepted everywhere (play.py, evaluate.py, training opponent):
  random           kaggle's built-in random agent
  starter          kaggle's built-in starter agent (greedy toward static planets)
  policy-random    our RandomPolicy placeholder
  nearest-sniper, lead-sniper, ...   the scripted agents in scripted.py
  model:PATH       a PolicyNet checkpoint, e.g. model:checkpoints/latest.pt
  path/to/main.py  any Kaggle-style agent file (e.g. submission/main.py)
"""

from model import PolicyNet, RandomPolicy
from scripted import SCRIPTED

BUILTIN = ("random", "starter")


def policy_agent(policy):
    """Wrap anything with `.act(obs)` as a kaggle agent function."""

    def agent(obs, config=None):
        return policy.act(obs)

    return agent


def resolve_agent(spec, device="cpu"):
    if callable(spec) or spec is None:
        return spec
    if spec in BUILTIN or spec.endswith(".py"):
        return spec  # kaggle_environments resolves these itself
    if spec in SCRIPTED:
        return SCRIPTED[spec]
    if spec == "policy-random":
        return policy_agent(RandomPolicy())
    if spec.startswith("model:"):
        return policy_agent(PolicyNet.load(spec[len("model:"):], device=device))
    raise ValueError(f"Unknown agent spec: {spec!r}")
