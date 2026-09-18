import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.local_fullgame import (
    GIB,
    adjusted_interval,
    final_plan,
    guarded_run,
    resource_failure,
)
from src.holdem.experiment import Experiment


def test_local_plan_preserves_recipe_and_doubles_total_work():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "configs/holdem/local-fullgame.json").read_text())
    old = json.loads((root / "configs/holdem/longer-05.json").read_text())
    plan = Experiment.from_dict(data)
    assert plan.seeds == (2026091802, 2026091803)
    assert plan.training.sampler == "first-decision"
    assert plan.training.fit.width == old["training"]["fit"]["width"]
    assert plan.training.capacity == old["training"]["capacity"]
    assert plan.training.exploration == old["training"]["exploration"]
    assert (
        plan.iterations * plan.training.traversals_per_player
        == 2 * old["iterations"] * old["training"]["traversals_per_player"]
    )
    assert (
        plan.iterations * plan.training.fit.steps
        == 2 * old["iterations"] * old["training"]["fit"]["steps"]
    )
    final = final_plan(plan)
    arena = final.arena(final.scenarios[0])
    assert arena.split == "test"
    assert arena.blocks == 4096
    assert arena.opponents == ("random",)
    assert arena.root_seed != plan.evaluation_seed
    assert final.benchmarks == ()


def test_resource_limits_fail_closed():
    assert resource_failure(10, 10, 0, 30 * GIB, 0) == "wall_time_limit"
    assert resource_failure(0, 10, 8 * GIB, 30 * GIB, 0) == "process_memory_limit"
    assert resource_failure(0, 10, 0, 11 * GIB, 0) == "free_disk_limit"
    assert resource_failure(0, 10, 0, 30 * GIB, 9 * GIB) == "output_size_limit"
    assert resource_failure(0, 10, 0, 30 * GIB, 0) is None


def test_timeout_terminates_worker_and_retains_failure(tmp_path, monkeypatch):
    import scripts.local_fullgame as runner

    # Test the actual deadline independently of the CI host's available disk.
    original = runner.resource_failure
    monkeypatch.setattr(
        runner,
        "resource_failure",
        lambda elapsed, limit, *args: original(elapsed, limit, 0, 30 * GIB, 0),
    )
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    with pytest.raises(RuntimeError, match="wall_time_limit"):
        guarded_run(command, tmp_path / "worker.log", seconds=0.15, poll_seconds=0.02)
    status = json.loads((tmp_path / "worker.json").read_text())
    assert status["status"] == "failed"
    with pytest.raises(ProcessLookupError):
        os.kill(status["pid"], 0)


def test_failed_worker_is_not_reported_complete(tmp_path, monkeypatch):
    import scripts.local_fullgame as runner

    monkeypatch.setattr(runner, "resource_failure", lambda *args: None)
    with pytest.raises(RuntimeError, match="worker_exit_7"):
        guarded_run(
            [sys.executable, "-c", "raise SystemExit(7)"],
            tmp_path / "worker.log",
            seconds=5,
            poll_seconds=0.02,
        )
    assert json.loads((tmp_path / "worker.json").read_text())["returncode"] == 7


def test_adjusted_intervals_are_wider_and_keep_inconclusive_cases():
    from scipy.stats import t

    half = float(t.ppf(0.975, 99))
    estimate = {"blocks": 100, "bb_per_100": 2.2, "ci95": [2.2 - half, 2.2 + half]}
    result = adjusted_interval(estimate)
    assert result[0] < 0 < estimate["ci95"][0]
    assert result[1] > estimate["ci95"][1]
    assert adjusted_interval({**estimate, "ci95": None}) is None


def test_help_cannot_start_a_campaign(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.local_fullgame", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--plan" in completed.stdout
    assert list(tmp_path.iterdir()) == []


def test_exhausted_budget_never_launches_worker(tmp_path, monkeypatch):
    import scripts.local_fullgame as runner

    def forbidden(*args, **kwargs):
        pytest.fail("Started a worker without remaining time")

    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="positive remaining time"):
        guarded_run([sys.executable], tmp_path / "worker.log", seconds=0)


def test_campaign_failure_prevents_second_seed(tmp_path, monkeypatch):
    import scripts.local_fullgame as runner
    from src.arena import artifacts

    monkeypatch.setattr(
        artifacts, "git", lambda *args: "" if args[0] == "status" else "revision"
    )
    monkeypatch.setattr(artifacts, "environment", dict)
    monkeypatch.setattr(artifacts, "source_fingerprint", lambda: "source")
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 30 * GIB})()
    )
    commands = []

    def fail(command, *args, **kwargs):
        commands.append(command)
        raise RuntimeError("worker_exit_1")

    monkeypatch.setattr(runner, "guarded_run", fail)
    plan = Path(__file__).resolve().parents[1] / "configs/holdem/local-fullgame.json"
    with pytest.raises(RuntimeError, match="worker_exit_1"):
        runner.campaign(plan, tmp_path / "campaign")
    assert len(commands) == 1
    assert "2026091802" in commands[0]
    status = json.loads((tmp_path / "campaign/status.json").read_text())
    assert status["state"] == "failed"
    assert status["seeds"] == []


def test_disk_monitor_tolerates_atomic_rename(tmp_path):
    from scripts.local_fullgame import used_bytes

    stable = tmp_path / "published.json"
    stable.write_bytes(b"12345")

    class RenamedFile:
        def is_file(self):
            return True

        def stat(self):
            raise FileNotFoundError("renamed before stat")

    class Directory:
        def rglob(self, pattern):
            return iter((stable, RenamedFile()))

    assert used_bytes(Directory()) == 5
