import json
from hashlib import sha256

import pytest

from scripts.compare_holdem_sampling import ARMS, measure_cell, sample_seed
from src.holdem.checkpoint import save_training
from src.holdem.training import HoldemTrainer
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import table
from tests.test_holdem_training import config


@pytest.mark.parametrize("arm", ARMS)
def test_complete_cells_reproduce_and_incomplete_cells_have_no_estimates(tmp_path, arm):
    with deterministic_cpu():
        trainer = HoldemTrainer(table(4, (6,) * 4), config())
        trainer.step()
    job = tmp_path / "saved"
    job.mkdir()
    checkpoint = job / "training-1.pt"
    manifest = {"fixture": "sampling"}
    digest = save_training(trainer, checkpoint, manifest=manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (job / "training-1.json").write_text(json.dumps({"sha256": digest}))
    args = (tmp_path, "saved", 1, 0, arm, 3)
    first = measure_cell(*args, 10000, 30, tmp_path / "first")
    repeated = measure_cell(*args, 10000, 30, tmp_path / "repeat")
    assert first["status"] == "completed"
    assert first["rows"] == repeated["rows"]
    assert first["value_bb"] == repeated["value_bb"]
    assert first["completed_replicates"] == first["requested_replicates"] == 3
    assert first["profile_sha256"] == trainer.current_profile().fingerprint
    assert first["table"]["button"] == 1
    assert (
        first["checkpoint_sha256"]
        == sha256(checkpoint.read_bytes()).hexdigest()
        == digest
    )
    limit = first["rows"][0]["nodes"] + 1
    failed = measure_cell(*args, limit, 30, tmp_path / "failed")
    assert failed["status"] == "invalid" and failed["completed_replicates"] == 1
    assert failed["attempted_replicates"] == 2
    assert "value_bb" not in failed and "root_action_values_bb" not in failed
    assert "CollectionLimitExceeded" in failed["error"]
    assert json.loads((tmp_path / "failed/report.json").read_text()) == json.loads(
        json.dumps(failed)
    )
    with pytest.raises(FileExistsError):
        measure_cell(*args, 10000, 30, tmp_path / "first")


def test_comparison_streams_are_distinct():
    seeds = {
        sample_seed(job, arm, r)
        for job in ("first", "second")
        for arm in ARMS
        for r in range(256)
    }
    assert len(seeds) == 2 * 3 * 256
