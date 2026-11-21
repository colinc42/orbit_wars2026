"""Shared test helpers: where the replays are and which games to test on."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPLAY_ROOT = Path(os.environ.get("ORBIT_REPLAYS", ROOT / "replay_memory"))
GAMES_PER_GROUP = 25   # 8 groups (2p/4p x standard/scrambled x clean/noisy) -> 200 games


def read_index():
    lines = (REPLAY_ROOT / "index.jsonl").read_text(encoding="utf-8").splitlines()
    return sorted((json.loads(line) for line in lines if line.strip()), key=lambda row: row["index"])


def sample_games(per_group=GAMES_PER_GROUP):
    """Paths of `per_group` games from each mix of player count, start and noise
    (the lowest-numbered ones, so the sample is the same every run)."""
    groups = {}
    for row in read_index():
        key = (len(row["agents"]), row["start"], any(row["noise"]))
        if len(groups.setdefault(key, [])) < per_group:
            groups[key].append(REPLAY_ROOT / row["file"])
    return [path for key in sorted(groups) for path in groups[key]]


def pytest_collection_modifyitems(config, items):
    if not (REPLAY_ROOT / "index.jsonl").exists():
        skip = pytest.mark.skip(reason=f"no replays at {REPLAY_ROOT} (set ORBIT_REPLAYS)")
        for item in items:
            if "replays" in item.keywords:
                item.add_marker(skip)
