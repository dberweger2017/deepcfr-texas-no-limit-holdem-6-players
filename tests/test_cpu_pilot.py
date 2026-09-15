import json
import os
import sys
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from time import monotonic

import numpy as np
import pytest

from scripts import cpu_pilot
from scripts.pilot_replay import export_replay, load_replay
from scripts.pilot_worker import execute
from src.solver.experiment import write_json
from src.solver.neural.experiment import Plan, run
from src.solver.neural.solver import Config


def small_plan():
    return Plan(
        "kuhn",
        2,
        Config(
            hidden=8,
            traversals=8,
            advantage_steps=2,
            strategy_steps=3,
            capacity=32,
            batch_size=8,
            seed=11,
        ),
        evaluation_interval=2,
        maximum_seconds=30,
    )


@pytest.fixture
def replay(tmp_path):
    training, output = tmp_path / "training", tmp_path / "replay"
    run(small_plan(), training)
    export_replay(training, output)
    return output


def test_portable_replay_keeps_targets_and_origin_but_is_not_a_checkpoint(replay):
    solver, memory, manifest = load_replay(replay)
    assert memory.seen >= memory.size > 0
    assert solver.iterations == 0
    assert solver.strategy is None
    assert manifest["iteration"] == 2
    assert manifest["checkpoint_sha256"]
    original = manifest["training_manifest"]["environment"]
    manifest["training_manifest"]["environment"] = {"platform": "another host"}
    write_json(replay / "manifest.json", manifest)
    assert load_replay(replay)[1].seen == memory.seen
    assert original != manifest["training_manifest"]["environment"]
    manifest["training_manifest"]["source_sha256"] = "0" * 64
    write_json(replay / "manifest.json", manifest)
    with pytest.raises(ValueError, match="source mismatch"):
        load_replay(replay)


def test_replay_rejects_modified_bytes_and_illegal_targets(replay):
    data = (replay / "replay.npz").read_bytes()
    (replay / "replay.npz").write_bytes(data + b"modified")
    with pytest.raises(ValueError, match="hash"):
        load_replay(replay)
    with np.load(BytesIO(data)) as saved:
        arrays = dict(saved)
    arrays["targets"][0] = [-1, 1, 1]
    np.savez_compressed(replay / "replay.npz", **arrays)
    manifest = json.loads((replay / "manifest.json").read_text())
    manifest["replay_sha256"] = sha256((replay / "replay.npz").read_bytes()).hexdigest()
    write_json(replay / "manifest.json", manifest)
    with pytest.raises(ValueError, match="probabilities"):
        load_replay(replay)


def test_refits_are_repeatable_and_report_unobserved_information_sets(replay, tmp_path):
    job = {
        "kind": "refit",
        "replay": str(replay),
        "maximum_seconds": 30,
        "fit": {"hidden": 8, "steps": 5, "batch_size": 8, "learning_rate": 0.001},
    }
    first = execute(job, tmp_path / "first")
    second = execute(job, tmp_path / "second")
    assert first["status"] == second["status"] == "completed"
    for field in ("fit", "evaluation", "policy_sha256", "replay_sha256"):
        assert first[field] == second[field]
    rows = json.loads((tmp_path / "first/information_sets.json").read_text())
    assert sum(row["samples"] for row in rows) == first["fit"]["stored_samples"]
    assert all(row["target"] is None for row in rows if row["samples"] == 0)


def test_process_scheduling_preserves_training_results(tmp_path):
    jobs = [{"kind": "training", "plan": asdict(small_plan())}] * 2
    results = [
        cpu_pilot.run_jobs(
            jobs,
            tmp_path / str(workers),
            workers=workers,
            deadline=monotonic() + 60,
            job_seconds=30,
        )
        for workers in (1, 2)
    ]
    assert cpu_pilot.same_training(*results)
    assert all(row["report"]["peak_rss_bytes"] > 0 for row in results[1]["jobs"])
    results[1]["jobs"][0]["report"]["result"]["completed_iterations"] += 1
    assert not cpu_pilot.same_training(*results)


@pytest.mark.skipif(
    sys.platform == "win32", reason="Pilot requires POSIX process groups"
)
def test_timeout_reaps_worker_and_records_partial_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu_pilot, "ROOT", tmp_path)
    (tmp_path / "sleeper.py").write_text(
        "import os, pathlib, time\n"
        "pathlib.Path('worker.pid').write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(TimeoutError, match="Worker"):
        cpu_pilot.run_jobs(
            [{}],
            tmp_path / "jobs",
            workers=1,
            deadline=monotonic() + 5,
            job_seconds=0.5,
            module="sleeper",
        )
    pid = int((tmp_path / "worker.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert (tmp_path / "jobs/job-000.json").exists()


def test_invalid_limits_fail_before_creating_output(tmp_path):
    settings = json.loads(
        (cpu_pilot.ROOT / "configs/solver/cpu-pilot-v1.json").read_text()
    )
    for field, value in (
        ("workers", 0),
        ("job_seconds", 1000),
        ("maximum_seconds", 20000),
        ("benchmark_seeds", [1, 1]),
    ):
        with pytest.raises(ValueError):
            cpu_pilot.run_pilot({**settings, field: value}, tmp_path / field)
        assert not (tmp_path / field).exists()
