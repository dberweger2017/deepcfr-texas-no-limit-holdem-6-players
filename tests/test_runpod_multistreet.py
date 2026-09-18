"""Focused process and deadline checks for the Runpod operations wrapper."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest
from pathlib import Path

from scripts import runpod_multistreet
from scripts.runpod_multistreet import EXIT_DEADLINE, build_parser, run


def _args(tmp_path: Path, *extra: str):
    parser = build_parser()
    return parser.parse_args(
        [
            "run",
            "--out",
            str(tmp_path / "ops"),
            "--seeds",
            "11,29",
            "--workers",
            "2",
            "--max-runtime-seconds",
            "10",
            "--retrieval-reserve-seconds",
            "2",
            *extra,
        ]
    )


def test_parallel_workers_record_exit_and_provenance(tmp_path: Path):
    args = _args(tmp_path)
    command = [
        sys.executable,
        "-c",
        "import pathlib; pathlib.Path('{out}/result-{seed}').write_text('ok')",
    ]

    assert run(args, command) == 0
    status = json.loads((tmp_path / "ops/ops-status.json").read_text())
    assert status["state"] == "needs_retrieval"
    assert status["retrieval_ready"] is True
    assert status["thread_environment"]["OMP_NUM_THREADS"] == "1"
    assert status["host"]["git_revision"]
    telemetry = (tmp_path / "ops/resource-telemetry.jsonl").read_text().splitlines()
    assert telemetry
    assert "cgroup" in json.loads(telemetry[0])
    for worker, seed in enumerate((11, 29)):
        record = json.loads((tmp_path / f"ops/worker-{worker:03d}.json").read_text())
        assert record["returncode"] == 0
        assert record["state"] == "completed"
        assert (tmp_path / f"ops/worker-{worker:03d}/result-{seed}").read_text() == "ok"


@pytest.fixture
def cached_provenance(monkeypatch):
    # Deadline tests exercise process cleanup; network-mounted Git metadata can
    # consume their entire short budget before a worker has even started.
    provenance = runpod_multistreet.host_provenance()
    monkeypatch.setattr(runpod_multistreet, "host_provenance", lambda: provenance)


def test_deadline_terminates_workers_and_keeps_retrieval_ready(tmp_path: Path, cached_provenance):
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--out",
            str(tmp_path / "ops"),
            "--workers",
            "1",
            "--max-runtime-seconds",
            "3",
            "--retrieval-reserve-seconds",
            "1",
            "--term-grace-seconds",
            "0.1",
            "--poll-seconds",
            "0.02",
        ]
    )
    result = run(args, [sys.executable, "-c", "import time; time.sleep(10)"])
    assert result == EXIT_DEADLINE
    status = json.loads((tmp_path / "ops/ops-status.json").read_text())
    assert status["state"] == "deadline"
    assert status["retrieval_ready"] is True
    record = json.loads((tmp_path / "ops/worker-000.json").read_text())
    assert record["termination_reason"] == "work_deadline"
    assert record["returncode"] != 0


def test_deadline_kills_child_that_ignores_term(tmp_path: Path, cached_provenance):
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--out",
            str(tmp_path / "ops"),
            "--workers",
            "1",
            "--max-runtime-seconds",
            "3",
            "--retrieval-reserve-seconds",
            "1",
            "--term-grace-seconds",
            "0.1",
            "--poll-seconds",
            "0.02",
        ]
    )
    child_pid = tmp_path / "ops/child.pid"
    code = (
        "import pathlib,signal,subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c',\"import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(20)\"]); "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(p.pid)); "
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); time.sleep(20)"
    )
    assert run(args, [sys.executable, "-c", code]) == EXIT_DEADLINE
    pid = int(child_pid.read_text())
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        # Container PID 1 may leave a killed orphan waiting to be reaped.
        stat = Path(f"/proc/{pid}/stat")
        try:
            if stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                break
        except FileNotFoundError:
            if sys.platform == "linux":
                break
        time.sleep(0.05)
    else:
        raise AssertionError(f"detached child {pid} survived deadline cleanup")


def test_resume_skips_verified_completed_worker(tmp_path: Path):
    args = _args(tmp_path)
    command = [
        sys.executable,
        "-c",
        "import pathlib; p=pathlib.Path('{out}/count'); p.write_text(str(int(p.read_text())+1) if p.exists() else '1')",
    ]
    assert run(args, command) == 0
    args.resume = True
    assert run(args, command) == 0
    for worker in (0, 1):
        assert (tmp_path / f"ops/worker-{worker:03d}/count").read_text() == "1"
