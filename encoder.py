"""State encoder over planet tokens (with the incoming-ships forecast) and one global token.

    planets [B, 44, 146] --planet MLP--> [B, 44, d] --+
                                                      +--> 45 tokens --L x (self-attention + MLP)--> pooling --> z [B, d]
    global  [B, 20]      --global MLP--> [B, 1, d] --+

Fleets are not inputs: what they will do is already in the planet tokens'
forecast blocks (features.py), so there is no fleet stream and no cross-attention.

Heads read z:
  win    logit that this player wins the game         (loss: binary cross-entropy)
  share  this player's share of all ships 50 turns on (loss: mean squared error)

    model = StateEncoder()
    out = model(batch)                  # batch from dataset.collate
    loss, parts = losses(out, batch)    # parts: {"win": ..., "share": ...} as floats
"""

import torch
from torch import nn
from torch.nn import functional as F

from features import GLOBAL_FEATS, PLANET_FEATS

D_MODEL = 128
N_LAYERS = 3
N_HEADS = 4
MLP_RATIO = 2          # hidden size of each block's MLP = MLP_RATIO * d


def mlp(in_size, hidden_size, out_size):
    """Two linear layers with a ReLU in between."""
    return nn.Sequential(nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_size))


class Block(nn.Module):
    """Pre-norm transformer block: x + attention(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, d, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = mlp(d, MLP_RATIO * d, d)

    def forward(self, x, padding):
        """x [B, N, d]; padding [B, N] bool, True = ignore this token. Returns [B, N, d]."""
        h = self.norm1(x)
        x = x + self.attn(h, h, h, key_padding_mask=padding, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class StateEncoder(nn.Module):
    def __init__(self, d=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS):
        super().__init__()
        self.planet_mlp = mlp(PLANET_FEATS, d, d)
        self.global_mlp = mlp(GLOBAL_FEATS, d, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.pool = nn.Linear(2 * d, d)       # [global token, mean of planet tokens] -> z
        self.win_head = nn.Linear(d, 1)
        self.share_head = nn.Linear(d, 1)

    def encode(self, batch):
        """batch: planets [B, 44, 146], pmask [B, 44], global [B, 20]. Returns z [B, d]."""
        planets = self.planet_mlp(batch["planets"].float())              # [B, 44, d]
        glob = self.global_mlp(batch["global"].float()).unsqueeze(1)     # [B, 1, d]
        x = torch.cat([glob, planets], dim=1)                            # [B, 45, d]
        pmask = batch["pmask"].bool()
        never_masked = torch.zeros_like(pmask[:, :1])                    # the global token
        padding = torch.cat([never_masked, ~pmask], dim=1)               # [B, 45]

        for block in self.blocks:
            x = block(x, padding)
        x = self.norm(x)

        # Pooling: the global token, and the mean over real planets.
        m = pmask.unsqueeze(-1).float()
        planet_mean = (x[:, 1:] * m).sum(1) / m.sum(1).clamp(min=1.0)    # [B, d]
        return self.pool(torch.cat([x[:, 0], planet_mean], dim=-1))      # [B, d]

    def forward(self, batch):
        """Returns {"z": [B, d], "win_logit": [B], "share": [B]}."""
        z = self.encode(batch)
        return {
            "z": z,
            "win_logit": self.win_head(z).squeeze(-1),
            "share": torch.sigmoid(self.share_head(z).squeeze(-1)),
        }


def losses(out, batch):
    """Total loss and its parts (as floats) for a batch from dataset.collate."""
    win = F.binary_cross_entropy_with_logits(out["win_logit"], batch["win"].float())
    share = F.mse_loss(out["share"], batch["future_share"].float())
    return win + share, {"win": win.item(), "share": share.item()}
