import json
from dataclasses import asdict, replace

import pytest

from scripts.check_solver import main
from src.solver.experiment import Plan, RunSpec, reproduce, run


def plan(**kwargs):
    return Plan(
        (RunSpec("kuhn", "external", 10, (7,), 1, 1),), evaluation_interval=5, **kwargs
    )


def test_saved_reference_run_reproduces_strategies_and_every_evaluation(tmp_path):
    spec = plan()
    assert Plan.from_dict(asdict(spec)) == spec
    result = run(spec, tmp_path / "first")
    assert result["status"] == "passed"
    assert [e["iteration"] for e in result["runs"][0]["evaluations"]] == [0, 5, 10]
    replay = reproduce(tmp_path / "first", tmp_path / "second")
    assert result["runs"][0]["strategy_sha256"] == replay["runs"][0]["strategy_sha256"]
    with pytest.raises(FileExistsError):
        run(spec, tmp_path / "first")
    manifest = tmp_path / "first/manifest.json"
    altered = json.loads(manifest.read_text())
    altered["source_sha256"] = "0" * 64
    manifest.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="source_sha256"):
        reproduce(tmp_path / "first", tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_failed_threshold_and_timeouts_remain_visible(tmp_path):
    spec = plan()
    failed = replace(spec, runs=(replace(spec.runs[0], maximum_exploitability=0),))
    result = run(failed, tmp_path / "failed")
    assert result["status"] == "failed"
    assert result["runs"][0]["completed_iterations"] == 10
    assert (tmp_path / "failed/kuhn-external-7-strategy.json").exists()
    with pytest.raises(ValueError, match="passing"):
        reproduce(tmp_path / "failed", tmp_path / "retry")
    exhausted = run(plan(maximum_total_seconds=1e-12), tmp_path / "budget")
    assert exhausted["status"] == "failed"
    assert exhausted["runs"][0]["status"] == "not_started_budget_exhausted"
    timed = run(plan(maximum_seconds_per_run=1e-12), tmp_path / "timeout")
    assert timed["status"] == "failed"
    assert timed["runs"][0]["status"] == "timed_out"
    assert timed["runs"][0]["completed_iterations"] == 0


def test_cli_returns_failure_for_unmet_acceptance_limits(tmp_path):
    spec = plan()
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(asdict(spec)))
    assert main(["--plan", str(source), "--out", str(tmp_path / "passing")]) == 0
    failed = replace(spec, runs=(replace(spec.runs[0], maximum_exploitability=0),))
    source.write_text(json.dumps(asdict(failed)))
    assert main(["--plan", str(source), "--out", str(tmp_path / "failed")]) == 1


def test_invalid_budgets_and_duplicate_runs_are_rejected():
    spec = plan()
    for kwargs in (
        {"maximum_total_seconds": 900},
        {"maximum_seconds_per_run": float("inf")},
        {"evaluation_interval": 0},
        {"oracle_tolerance": 0.1},
        {"runs": spec.runs * 2},
    ):
        with pytest.raises(ValueError):
            replace(spec, **kwargs)
    with pytest.raises(ValueError):
        replace(spec.runs[0], seeds=(7, 7))
