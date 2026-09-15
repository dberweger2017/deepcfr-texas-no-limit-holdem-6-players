import json
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
import torch

from src.solver.games import new_game
from src.solver.neural.artifact import load_policy
from src.solver.neural.experiment import Plan, check_refit, reproduce, run
from src.solver.neural.solver import Config


def small_plan(**kwargs):
    return Plan(
        "kuhn",
        2,
        Config(
            hidden=8,
            strategy_hidden=16,
            traversals=8,
            advantage_steps=3,
            strategy_steps=4,
            batch_size=8,
            capacity=16,
        ),
        evaluation_interval=1,
        **kwargs,
    )


def test_complete_neural_bundle_reproduces_and_policy_reloads(tmp_path):
    plan = small_plan()
    result = run(plan, tmp_path / "first")
    assert result["status"] == "completed"
    assert result["completed_iterations"] == 2
    assert len(result["advantage_fits"]) == 4
    assert len(result["evaluations"]) == 2
    replay = reproduce(tmp_path / "first", tmp_path / "second")
    assert result["policy_file_sha256"] == replay["policy_file_sha256"]
    assert (tmp_path / "first/policy.pt").read_bytes() == (
        tmp_path / "second/policy.pt"
    ).read_bytes()
    before = torch.random.get_rng_state().clone()
    policy = load_policy(
        tmp_path / "first/policy.pt", result["policy_file_sha256"], player=0, seed=3
    )
    assert torch.equal(before, torch.random.get_rng_state())
    first = new_game("kuhn").deal(0).deal(1)
    second = new_game("kuhn").deal(0).deal(2)
    assert policy.distribution(first.information_set()) == policy.distribution(
        second.information_set()
    )
    assert sum(
        p for _, p in policy.distribution(first.information_set())
    ) == pytest.approx(1)
    with pytest.raises(FileExistsError):
        run(plan, tmp_path / "first")
    with pytest.raises(ValueError, match="hash"):
        load_policy(tmp_path / "first/policy.pt", "0" * 64, player=0, seed=3)
    (tmp_path / "first/policy.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        reproduce(tmp_path / "first", tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_timeout_is_retained_and_never_exports_a_partial_policy(tmp_path):
    result = run(small_plan(maximum_seconds=1e-12), tmp_path / "timeout")
    assert result["status"] == "timed_out"
    assert result["completed_iterations"] == 0
    assert (tmp_path / "timeout/report.json").exists()
    assert not (tmp_path / "timeout/policy.pt").exists()
    with pytest.raises(ValueError, match="completed"):
        reproduce(tmp_path / "timeout", tmp_path / "retry")


def test_source_changes_prevent_a_reproduction_claim(tmp_path):
    run(small_plan(), tmp_path / "one")
    path = tmp_path / "one/manifest.json"
    data = json.loads(path.read_text())
    data["source_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source_sha256"):
        reproduce(tmp_path / "one", tmp_path / "bad")


def test_refit_reconstructs_the_pilot_without_changing_replay(tmp_path):
    plan = small_plan()
    pilot = run(plan, tmp_path / "pilot")
    result = check_refit(plan, tmp_path / "refit")
    assert result["baseline_fits"] == pilot["advantage_fits"][-2:]
    assert len(result["comparisons"]) == 2
    for baseline, comparison in zip(result["baseline_fits"], result["comparisons"]):
        assert comparison["refit"]["steps"] == 4000
        assert comparison["baseline_steps"] == plan.training.advantage_steps
        for field in ("sample_noise_mse", "seen_samples", "stored_samples"):
            assert comparison["refit"][field] == baseline[field]
        assert len(comparison["memory_sha256"]) == 64
    assert not (tmp_path / "refit/policy.pt").exists()


def test_loader_rejects_nonfinite_weights_even_with_a_matching_file_hash(tmp_path):
    run(small_plan(), tmp_path / "run")
    path = tmp_path / "run/policy.pt"
    data = torch.load(path, weights_only=True)
    next(iter(data["strategy"].values())).fill_(np.nan)
    torch.save(data, path)
    with pytest.raises(ValueError, match="finite"):
        load_policy(path, sha256(path.read_bytes()).hexdigest(), player=0, seed=3)


def test_invalid_plan_limits_fail_before_any_run():
    plan = small_plan()
    for changes in (
        {"maximum_seconds": 900},
        {"maximum_seconds": float("nan")},
        {"iterations": 0},
        {"evaluation_interval": 0},
    ):
        with pytest.raises(ValueError):
            replace(plan, **changes)
