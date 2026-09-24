"""The mixed table keeps six seats and reconciles the cumulative cash ledger."""

import json
from pathlib import Path

import pytest

from scripts.blueprint_mixed_table import PLAYER_IDS, main, run
from src.blueprint.artifact import save_training
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Table


def _trainer():
    table = Table(tuple(f"player-{i}" for i in range(6)), (10_000,) * 6)
    return BlueprintTrainer(table, PilotConfig())


def test_mixed_table_reproducible_and_zero_sum():
    trainer = _trainer()
    result = run(trainer, hands=12, seed=931, sample_every=3)
    assert result == run(trainer, hands=12, seed=931, sample_every=3)
    assert [row["hand"] for row in result["samples"]] == [3, 6, 9, 12]
    assert result["players"] == list(PLAYER_IDS)
    assert all(sum(row["net_bb"].values()) == 0 for row in result["samples"])
    assert result["stacks"].startswith("reset to 100 BB")


def test_mixed_table_requires_pinned_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.json.gz"
    digest = save_training(_trainer(), checkpoint)
    output = tmp_path / "result"
    with pytest.raises(SystemExit):
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--expected-sha256",
                "0" * 64,
                "--out",
                str(output),
                "--hands",
                "1",
            ]
        )
    assert not output.exists()
    assert (
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--expected-sha256",
                digest,
                "--out",
                str(output),
                "--hands",
                "1",
            ]
        )
        == 0
    )
    result = json.loads((output / "result.json").read_text())
    assert result["checkpoint_sha256"] == digest
    assert result["samples"][0]["hand"] == 1
