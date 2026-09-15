import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256

import pytest

from scripts.run_neural_readiness import (
    execute,
    load_plan,
    rental_time_remaining,
    smoke_plan,
)
from src.solver.experiment import canonical
from src.solver.neural.campaign import Campaign, run_seed
from src.solver.neural.readiness import run_job


def test_protocol_keeps_all_reserved_seeds_and_original_absolute_limits():
    settings = load_plan()
    seeds = [401, 409, 419, 421, 431, 433, 439, 443]
    for game, spec in settings["campaigns"].items():
        campaign = Campaign.from_dict(spec)
        assert list(campaign.seeds) == seeds
        assert (campaign.maximum_exploitability, campaign.maximum_value_error) == (
            (0.03, 0.03) if game == "kuhn" else (0.15, 0.10)
        )
    assert settings["acceptance"] == {
        "final_only": True,
        "all_seeds_required": True,
        "paired_regressions_are_vetoes": False,
    }
    smoke = smoke_plan(settings)
    assert set(smoke["campaigns"]["leduc"]["seeds"]).isdisjoint(seeds)


def test_worker_pairs_fixed_fit_with_identical_end_to_end_collection(tmp_path):
    settings = smoke_plan(load_plan())
    spec = settings["campaigns"]["leduc"]
    job = {"campaign": spec, "seed": 101, "comparison_seconds": 30}
    result = run_job(job, tmp_path / "decay")
    campaign = Campaign.from_dict(spec)
    fixed = replace(
        campaign,
        training=replace(
            campaign.training,
            training=replace(
                campaign.training.training,
                strategy_learning_rate_schedule="constant",
                strategy_final_learning_rate=None,
            ),
        ),
    )
    control = run_seed(fixed, 101, tmp_path / "fixed")
    comparison = result["comparison"]
    assert comparison["fixed_policy_sha256"] == control["final"]["strategy_sha256"]
    assert comparison["fixed"] == control["final"]["neural_average"]
    assert comparison["strategy_fit"] == control["final"]["strategy_fit"]
    assert comparison["exploitability_delta"] == (
        result["result"]["final"]["neural_average"]["exploitability"]
        - control["final"]["neural_average"]["exploitability"]
    )
    assert (tmp_path / "decay/fixed-policy.pt").exists()
    training = tmp_path / "decay/campaign/training"
    descriptor = json.loads((training / "checkpoint.json").read_text())
    assert (
        sha256((training / descriptor["file"]).read_bytes()).hexdigest()
        == comparison["checkpoint_sha256"]
    )
    job["campaign"] = asdict(replace(campaign, maximum_exploitability=0))
    failed = run_job(job, tmp_path / "failed")
    assert failed["status"] == "completed"
    assert failed["result"]["status"] == "failed"
    assert "comparison" in failed


def test_runner_smoke_uses_fresh_processes_and_never_claims_readiness(tmp_path):
    plan = smoke_plan(load_plan())
    report = execute(plan, tmp_path / "smoke", seconds=300, smoke=True)
    assert report["status"] == "smoke_completed" and not report["scored"]
    assert len(report["execution"]["jobs"]) == 6
    assert [r["seed"] for r in report["paired_comparisons"]] == [101, 103, 107]
    for game in ("kuhn", "leduc"):
        assert len(report["games"][game]["runs"]) == 3
    manifest = json.loads((tmp_path / "smoke/manifest.json").read_text())
    assert "scripts/run_neural_readiness.py" in manifest["source_files"]
    assert canonical(manifest["plan"]) == canonical(plan)


def test_failed_worker_leaves_campaign_inconclusive(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("worker deadline")

    monkeypatch.setattr("scripts.run_neural_readiness.run_jobs", fail)
    with pytest.raises(TimeoutError):
        execute(smoke_plan(load_plan()), tmp_path / "failed", seconds=1)
    result = json.loads((tmp_path / "failed/report.json").read_text())
    assert result["status"] == "inconclusive"
    assert "worker deadline" in result["error"]


def test_rental_deadline_counts_setup_and_reserves_retrieval_time():
    resources = load_plan()["resources"]
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    assert rental_time_remaining(resources, 1.2, now, now) == 6600
    assert (
        rental_time_remaining(resources, 1.2, now - timedelta(minutes=15), now) == 5700
    )
    for rate in (0, -1, 1.21, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            rental_time_remaining(resources, rate, now, now)
    for started in (
        now + timedelta(seconds=1),
        now - timedelta(hours=2),
        now.replace(tzinfo=None),
    ):
        with pytest.raises(ValueError):
            rental_time_remaining(resources, 1, started, now)
