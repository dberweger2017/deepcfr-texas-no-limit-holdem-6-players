from dataclasses import replace
from hashlib import sha256

import pytest

from src.solver.neural.campaign import REFERENCE, Campaign, assess, run_seed, summarize
from src.solver.neural.experiment import Plan
from src.solver.neural.solver import Config


def campaign():
    return Campaign(
        Plan(
            "kuhn",
            2,
            Config(
                hidden=8,
                traversals=8,
                advantage_steps=2,
                strategy_steps=3,
                capacity=16,
                batch_size=8,
            ),
            evaluation_interval=1,
        ),
        (11, 29, 47),
        2,
        2,
        sha256(REFERENCE.read_bytes()).hexdigest(),
    )


def test_campaign_requires_every_seed_and_retains_failed_results(tmp_path):
    spec = replace(campaign(), maximum_exploitability=0)
    paths = [tmp_path / str(seed) for seed in spec.seeds]
    for seed, path in zip(spec.seeds, paths):
        assert run_seed(spec, seed, path)["status"] == "failed"
    with pytest.raises(ValueError, match="Every declared seed"):
        summarize(spec, paths[:-1], tmp_path / "missing")
    with pytest.raises(ValueError, match="duplicate"):
        summarize(spec, paths + paths[:1], tmp_path / "duplicate")
    result = summarize(spec, paths, tmp_path / "summary")
    assert result["status"] == "failed"
    assert [row["seed"] for row in result["runs"]] == list(spec.seeds)
    assert result["exploitability_summary"]["count"] == 3
    path = paths[0] / "training/report.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        summarize(spec, paths, tmp_path / "tampered")


def test_campaign_resume_keeps_the_original_seed_and_plan(tmp_path):
    spec = campaign()
    assert run_seed(spec, 11, tmp_path / "paused", stop_after=1)["status"] == "paused"
    with pytest.raises(ValueError, match="seed"):
        run_seed(spec, 29, tmp_path / "wrong", resume=tmp_path / "paused")
    result = run_seed(spec, 11, tmp_path / "resumed", resume=tmp_path / "paused")
    assert result["status"] == "passed"
    assert result["completed_iterations"] == 2


def test_acceptance_uses_the_final_neural_policy_and_both_thresholds():
    spec = replace(campaign(), maximum_exploitability=0.03, maximum_value_error=0.02)
    report = {
        "status": "completed",
        "completed_iterations": 2,
        "advantage_fits": [],
        "evaluations": [
            {
                "iteration": 1,
                "neural_average": {"exploitability": 0.001, "value_player0": 0},
            },
            {
                "iteration": 2,
                "neural_average": {"exploitability": 0.04, "value_player0": 0},
            },
        ],
    }
    assert assess(spec, 11, report, {"value": 0})["status"] == "failed"
    report["evaluations"][-1]["neural_average"] = {
        "exploitability": 0.01,
        "value_player0": 0.03,
    }
    assert assess(spec, 11, report, {"value": 0})["status"] == "failed"
    report["evaluations"][-1]["neural_average"]["value_player0"] = 0.01
    assert assess(spec, 11, report, {"value": 0})["status"] == "passed"


def test_campaign_limits_cannot_be_omitted_or_changed_silently(tmp_path):
    spec = campaign()
    for changes in (
        {"seeds": (1, 1, 2)},
        {"seeds": (1, 2)},
        {"maximum_exploitability": float("nan")},
        {"maximum_value_error": -1},
    ):
        with pytest.raises(ValueError):
            replace(spec, **changes)
    with pytest.raises(ValueError, match="declared"):
        run_seed(spec, 7, tmp_path / "wrong-seed")
    with pytest.raises(ValueError, match="hash"):
        run_seed(
            replace(spec, reference_sha256="0" * 64), 11, tmp_path / "wrong-reference"
        )
