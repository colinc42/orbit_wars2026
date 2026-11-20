"""Policies. Every policy exposes `act(obs) -> moves` so it can be dropped into
a Kaggle agent, `play.py` or `evaluate.py` interchangeably.

- RandomPolicy: uniform random (target, fraction) choices. The placeholder.
- PolicyNet:    small attention network over planets with a pointer-style
                target head, a fraction head and a value head. Untrained it is
                also ~random; `train.py` improves it.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from actions import HOLD, N_FRACTIONS, decode
from features import MAX_PLANETS, N_GLOBAL_FEATURES, N_PLANET_FEATURES, encode


class RandomPolicy:
    def __init__(self, hold_prob=0.9, seed=None):
        # Mostly hold: firing every planet every turn just bleeds ships.
        self.hold_prob = hold_prob
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        enc = encode(obs)
        n = int(enc["planet_mask"].sum())
        targets = self.rng.integers(0, max(n, 1), size=MAX_PLANETS)
        targets[self.rng.random(MAX_PLANETS) < self.hold_prob] = HOLD
        fractions = self.rng.integers(0, N_FRACTIONS, size=MAX_PLANETS)
        return decode(obs, targets, fractions)


def to_tensors(enc, device="cpu"):
    """Batch a single encoded observation (adds a leading batch dim of 1)."""
    return {
        k: torch.as_tensor(v, device=device).unsqueeze(0)
        for k, v in enc.items()
        if k != "planet_ids"
    }


def stack_tensors(encs, device="cpu"):
    """Batch a list of encoded observations."""
    keys = [k for k in encs[0] if k != "planet_ids"]
    return {k: torch.as_tensor(np.stack([e[k] for e in encs]), device=device) for k in keys}


class PolicyNet(nn.Module):
    def __init__(self, hidden=128, heads=4, layers=2, hold_bias=2.0):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Sequential(
            nn.Linear(N_PLANET_FEATURES + N_GLOBAL_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=hidden * 2, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.hold = nn.Linear(hidden, 1)
        nn.init.constant_(self.hold.bias, hold_bias)
        self.fraction = nn.Linear(hidden, N_FRACTIONS)
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch):
        planets = batch["planets"].float()  # (B, N, F)
        mask = batch["planet_mask"].bool()  # (B, N)
        g = batch["globals"].float()  # (B, G)
        B, N, _ = planets.shape

        x = torch.cat([planets, g.unsqueeze(1).expand(B, N, -1)], dim=-1)
        h = self.encoder(self.embed(x), src_key_padding_mask=~mask)  # (B, N, H)

        # Pointer logits: source i -> target j, plus a HOLD option.
        logits = torch.einsum("bih,bjh->bij", self.query(h), self.key(h)) / math.sqrt(self.hidden)
        invalid = ~mask.unsqueeze(1).expand(B, N, N) | torch.eye(N, dtype=torch.bool, device=h.device)
        logits = logits.masked_fill(invalid, -1e9)
        target_logits = torch.cat([logits, self.hold(h)], dim=-1)  # (B, N, N+1)

        fraction_logits = self.fraction(h)  # (B, N, F)

        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        value = self.value(pooled).squeeze(-1)  # (B,)
        return target_logits, fraction_logits, value

    def evaluate(self, batch, targets=None, fractions=None, deterministic=False):
        """Sample (or score given) actions. Returns a dict with per-sample
        log_prob, entropy and value. Only planets we own (source_mask)
        contribute; the fraction only counts when the planet doesn't HOLD."""
        target_logits, fraction_logits, value = self(batch)
        t_dist = Categorical(logits=target_logits)
        f_dist = Categorical(logits=fraction_logits)
        if targets is None:
            if deterministic:
                targets, fractions = target_logits.argmax(-1), fraction_logits.argmax(-1)
            else:
                targets, fractions = t_dist.sample(), f_dist.sample()

        src = batch["source_mask"].float()
        sends = src * (targets != HOLD).float()
        log_prob = (t_dist.log_prob(targets) * src).sum(-1) + (f_dist.log_prob(fractions) * sends).sum(-1)
        entropy = (t_dist.entropy() * src).sum(-1) + (f_dist.entropy() * sends).sum(-1)
        return {
            "targets": targets,
            "fractions": fractions,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
        }

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        device = next(self.parameters()).device
        out = self.evaluate(to_tensors(encode(obs), device), deterministic=deterministic)
        return decode(obs, out["targets"][0].tolist(), out["fractions"][0].tolist())

    def save(self, path):
        torch.save({"config": {"hidden": self.hidden}, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        net = cls(hidden=ckpt["config"]["hidden"]).to(device)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net
