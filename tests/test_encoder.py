"""The dataset produces fixed-size batches and the encoder trains on them (a smoke test)."""

import pytest
import torch

from dataset import ForecastDataset, collate, split_games
from encoder import StateEncoder, losses
from conftest import REPLAY_ROOT

BATCH = 32


@pytest.mark.replays
def test_one_training_step():
    if not split_games(REPLAY_ROOT, "train"):
        pytest.skip("no forecast files yet: run python precompute.py --games 20")
    stream = iter(ForecastDataset(REPLAY_ROOT, "train", states_per_game=8, buffer_size=64))
    batch = collate([next(stream) for _ in range(BATCH)])
    assert batch["planets"].shape == (BATCH, 44, 146)
    assert batch["pmask"].shape == (BATCH, 44) and batch["global"].shape == (BATCH, 20)
    assert set(batch) == {"planets", "pmask", "global", "win", "future_share"}   # no fleets anywhere

    model = StateEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = model(batch)
    assert out["z"].shape == (BATCH, 128)
    loss, parts = losses(out, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss) and set(parts) == {"win", "share"}


def test_padding_does_not_change_the_output():
    """Whatever sits in padded planet slots must not affect z."""
    torch.manual_seed(0)
    model = StateEncoder().eval()
    batch = {"planets": torch.randn(2, 44, 146), "pmask": torch.zeros(2, 44, dtype=torch.bool),
             "global": torch.randn(2, 20)}
    batch["pmask"][:, :30] = True
    z1 = model(batch)["z"]
    batch["planets"][:, 30:] = torch.randn(2, 14, 146) * 100
    z2 = model(batch)["z"]
    assert torch.allclose(z1, z2, atol=1e-5)
