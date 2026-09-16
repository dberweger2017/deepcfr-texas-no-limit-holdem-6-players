import json
from hashlib import sha256

import numpy as np
import pytest
import torch

from scripts.compare_holdem_variance import GradientMoments, measure_variance
from src.holdem.checkpoint import save_training
from src.holdem.training import HoldemTrainer
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import table
from tests.test_holdem_training import config


def test_gradient_moments_include_empty_roots_and_report_outlier_sensitivity():
    vectors = torch.tensor([[0, 0], [1, 2], [-3, 4], [100, 50]], dtype=torch.float64)
    moments = GradientMoments(2)
    for v in vectors:
        moments.add(v)
    report = moments.report()
    torch.testing.assert_close(moments.mean, vectors.mean(0))
    assert report["trace_sample_covariance"] == pytest.approx(
        float(vectors.var(0).sum())
    )
    assert report["trace_without_largest_sample"] == pytest.approx(
        float(vectors[:-1].var(0).sum())
    )
    assert 0 < report["largest_sample_variance_fraction"] <= 1
    with pytest.raises(FloatingPointError):
        moments.add(torch.tensor([float("nan"), 1]))


def test_saved_comparison_pairs_paths_and_keeps_failed_cells_out_of_estimates(tmp_path):
    with deterministic_cpu():
        trainer = HoldemTrainer(table(4, (6,) * 4), config())
        trainer.step()
    job = tmp_path / "saved"
    job.mkdir()
    manifest = {"fixture": "variance"}
    checkpoint = job / "training-1.pt"
    digest = save_training(trainer, checkpoint, manifest=manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (job / "training-1.json").write_text(json.dumps({"sha256": digest}))
    args = (tmp_path, "saved", 0)
    raw = measure_variance(*args, "single-zero", 3, tmp_path / "zero")
    frozen = measure_variance(*args, "single-frozen", 3, tmp_path / "frozen")
    branched = measure_variance(*args, "first-frozen", 3, tmp_path / "branch")
    for result in (raw, frozen, branched):
        assert result["status"] == "completed"
        assert result["completed_replicates"] == 3
        assert result["gradient"]["trace_sample_covariance"] >= 0
    assert [r["path_sha256"] for r in raw["rows"]] == [
        r["path_sha256"] for r in frozen["rows"]
    ]
    assert [r["max_inverse_own_reach"] for r in raw["rows"]] == [
        r["max_inverse_own_reach"] for r in frozen["rows"]
    ]
    assert sha256(checkpoint.read_bytes()).hexdigest() == digest
    mean = np.load(tmp_path / "frozen/mean-gradient.npy", allow_pickle=False)
    assert np.linalg.norm(mean) == pytest.approx(frozen["gradient"]["mean_norm"])
    failed = measure_variance(
        *args,
        "single-zero",
        3,
        tmp_path / "failed",
        max_nodes=raw["rows"][0]["nodes"] + 1,
    )
    assert failed["status"] == "invalid" and failed["completed_replicates"] == 1
    assert "gradient" not in failed and "root_value_bb" not in failed
    assert not (tmp_path / "failed/mean-gradient.npy").exists()
    with pytest.raises(FileExistsError):
        measure_variance(*args, "single-zero", 3, tmp_path / "zero")
