"""Run bounded multi-street workers with durable, provider-agnostic status.

This module owns process supervision and provenance only.  The scientific
runner is supplied as an argv template after ``--``; each token may contain
``{worker}``, ``{seed}``, and ``{out}``.  No provider credential or API call is
made here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXIT_DEADLINE = 124
EXIT_INTERRUPTED = 130
_SECRET_FLAGS = ("--api-key", "--token", "--secret", "--password", "--credential")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def host_provenance() -> dict[str, Any]:
    """Return reproducibility metadata without copying environment values."""

    affinity: list[int] | None
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    return {
        "captured_at": utc_now(),
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version,
        "pid": os.getpid(),
        "cpu_affinity": affinity,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_worktree_clean_tracked": _git("status", "--porcelain", "--untracked-files=no") == "",
    }


def _safe_command(command: list[str]) -> list[str]:
    if not command:
        raise ValueError("scientific command is empty")
    # The wrapper never runs a shell.  Refuse explicit credential flags so a
    # secret cannot be copied into the durable argv record.  Ordinary paths
    # such as ``--checkpoint`` remain valid scientific arguments.
    for token in command:
        lowered = token.lower().split("=", 1)[0]
        if lowered in _SECRET_FLAGS or any(lowered.startswith(flag + "-") for flag in _SECRET_FLAGS):
            raise ValueError("credential flags are not supported by this wrapper")
    return command


def _expand(command: list[str], *, worker: int, seed: int | None, out: Path) -> list[str]:
    values = {"worker": str(worker), "seed": "" if seed is None else str(seed), "out": str(out)}
    try:
        expanded = [token.format_map(values) for token in command]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unsupported command placeholder: {error}") from error
    return _safe_command(expanded)


def _terminate(process: subprocess.Popen[str], *, grace_seconds: float) -> str:
    """Terminate a process group and return the action taken."""

    if process.poll() is not None:
        return "already_exited"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_exited"
    try:
        process.wait(timeout=max(0.0, grace_seconds))
        return "sigterm"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return "sigterm"
        process.wait()
        return "sigkill"


def _parse_seeds(value: str | None) -> list[int | None]:
    if value is None or not value.strip():
        return [None]
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--seeds must be a comma-separated list of integers") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain at least one unique integer")
    return seeds


def _write_status(out: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = utc_now()
    _atomic_json(out / "ops-status.json", status)


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_cpu_stat_usage(path: Path) -> int | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        name, _, value = line.partition(" ")
        if name == "usage_usec":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def resource_snapshot(active: dict[int, tuple[subprocess.Popen[str], dict[str, Any], Any]]) -> dict[str, Any]:
    """Capture cheap cgroup and per-worker process measurements."""

    cgroup: dict[str, int] = {}
    for name, path in {
        "memory_current_bytes": Path("/sys/fs/cgroup/memory.current"),
        "memory_peak_bytes": Path("/sys/fs/cgroup/memory.peak"),
        "memory_usage_bytes_v1": Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        "memory_peak_bytes_v1": Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
        "cpu_usage_ns_v1": Path("/sys/fs/cgroup/cpuacct/cpuacct.usage"),
    }.items():
        value = _read_int(path)
        if value is not None:
            cgroup[name] = value
    cpu_usage = _read_cpu_stat_usage(Path("/sys/fs/cgroup/cpu.stat"))
    if cpu_usage is not None:
        cgroup["cpu_usage_usec"] = cpu_usage
    workers: dict[str, dict[str, int]] = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    for worker, (process, _record, _log) in active.items():
        statm = Path(f"/proc/{process.pid}/statm")
        try:
            fields = statm.read_text(encoding="utf-8").split()
            workers[str(worker)] = {
                "pid": process.pid,
                "rss_bytes": int(fields[1]) * page_size,
            }
        except (OSError, IndexError, ValueError):
            workers[str(worker)] = {"pid": process.pid}
    return {"captured_at": utc_now(), "cgroup": cgroup, "workers": workers}


def _append_resource_snapshot(out: Path, active: dict[int, tuple[subprocess.Popen[str], dict[str, Any], Any]]) -> None:
    with (out / "resource-telemetry.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(resource_snapshot(active), sort_keys=True) + "\n")


def run(args: argparse.Namespace, command: list[str]) -> int:
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    command = _safe_command(command)
    seeds = _parse_seeds(args.seeds)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if len(seeds) > args.workers:
        raise ValueError("--workers must be at least the number of seeds")
    if args.max_runtime_seconds <= args.retrieval_reserve_seconds:
        raise ValueError("max runtime must leave a positive retrieval reserve")

    started = time.monotonic()
    started_wall = time.time()
    work_deadline = started + args.max_runtime_seconds - args.retrieval_reserve_seconds
    status: dict[str, Any] = {
        "format": "runpod-multistreet-ops-v1",
        "state": "running",
        "started_at": utc_now(),
        "max_runtime_seconds": args.max_runtime_seconds,
        "retrieval_reserve_seconds": args.retrieval_reserve_seconds,
        "work_deadline_at": datetime.fromtimestamp(started_wall + args.max_runtime_seconds - args.retrieval_reserve_seconds, timezone.utc).isoformat(),
        "total_deadline_at": datetime.fromtimestamp(started_wall + args.max_runtime_seconds, timezone.utc).isoformat(),
        "workers": {},
        "host": host_provenance(),
        "command_template": command,
        "provider_control": "external; this wrapper never reads or emits provider credentials",
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    }
    _atomic_json(out / "host-provenance.json", status["host"])
    (out / "resource-telemetry.jsonl").touch()
    _write_status(out, status)

    pending = list(enumerate(seeds))
    if args.resume:
        remaining = []
        for worker, seed in pending:
            prior_path = out / f"worker-{worker:03d}.json"
            if prior_path.exists():
                prior = _read_json(prior_path)
                if prior.get("state") == "completed" and prior.get("returncode") == 0:
                    status["workers"][str(worker)] = prior
                    continue
            remaining.append((worker, seed))
        pending = remaining

    active: dict[int, tuple[subprocess.Popen[str], dict[str, Any], Any]] = {}
    _append_resource_snapshot(out, active)
    interrupted = False
    deadline_hit = False

    def stop_active(action: str) -> None:
        for worker, (process, record, log) in list(active.items()):
            record["termination"] = _terminate(process, grace_seconds=args.term_grace_seconds)
            record["state"] = "terminated"
            record["termination_reason"] = action
            record["returncode"] = process.returncode
            record["finished_at"] = utc_now()
            _atomic_json(out / f"worker-{worker:03d}.json", record)
            status["workers"][str(worker)] = record
            log.close()
        active.clear()

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        status["signal"] = signum

    old_handlers = {signum: signal.signal(signum, handle_signal) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        while pending or active:
            if interrupted:
                stop_active("signal")
                break
            if time.monotonic() >= work_deadline:
                deadline_hit = True
                stop_active("work_deadline")
                break
            while pending and len(active) < args.workers and time.monotonic() < work_deadline:
                worker, seed = pending.pop(0)
                worker_out = out / f"worker-{worker:03d}"
                worker_out.mkdir(parents=True, exist_ok=True)
                expanded = _expand(command, worker=worker, seed=seed, out=worker_out)
                record = {
                    "worker": worker,
                    "seed": seed,
                    "state": "running",
                    "started_at": utc_now(),
                    "argv": expanded,
                    "out": str(worker_out),
                }
                log = (out / f"worker-{worker:03d}.log").open("w", encoding="utf-8")
                env = os.environ.copy()
                # Scientific workers do not need provider control credentials.
                # Remove the known pod-scoped names before spawning them.
                env.pop("RUNPOD_API_KEY", None)
                env.pop("RUNPOD_POD_ID", None)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    env[name] = "1"
                env.update({"RUNPOD_WORKER_INDEX": str(worker), "RUNPOD_WORKER_OUT": str(worker_out)})
                process = subprocess.Popen(
                    expanded,
                    cwd=args.cwd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                record["pid"] = process.pid
                record["host"] = status["host"]
                _atomic_json(out / f"worker-{worker:03d}.json", record)
                active[worker] = (process, record, log)
                status["workers"][str(worker)] = record
                _write_status(out, status)
            for worker, (process, record, log) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                record["returncode"] = returncode
                record["state"] = "completed" if returncode == 0 else "failed"
                record["finished_at"] = utc_now()
                _atomic_json(out / f"worker-{worker:03d}.json", record)
                status["workers"][str(worker)] = record
                log.close()
                del active[worker]
            _append_resource_snapshot(out, active)
            _write_status(out, status)
            if active:
                time.sleep(min(args.poll_seconds, max(0.01, work_deadline - time.monotonic())))
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        for worker, (process, _record, log) in active.items():
            _terminate(process, grace_seconds=args.term_grace_seconds)
            log.close()

    failed = [row for row in status["workers"].values() if row.get("returncode") not in (0, None)]
    status["finished_at"] = utc_now()
    status["state"] = "interrupted" if interrupted else "deadline" if deadline_hit else "failed" if failed else "needs_retrieval"
    # Failed, deadline-limited, and interrupted runs still need artifact
    # retrieval.  The operator decides whether the partial result is useful;
    # the wrapper must never make it disappear by declaring it unready.
    status["retrieval_ready"] = True
    status["retrieval_deadline_at"] = status["total_deadline_at"]
    _write_status(out, status)
    _atomic_json(out / "retrieval-ready.json", {"ready": status["retrieval_ready"], "status": status["state"], "created_at": utc_now()})
    if interrupted:
        return EXIT_INTERRUPTED
    if deadline_hit:
        return EXIT_DEADLINE
    return 1 if failed or pending else 0


def status(args: argparse.Namespace) -> int:
    value = _read_json(args.out.expanduser().resolve() / "ops-status.json")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run_parser = subparsers.add_parser("run", help="run bounded scientific workers")
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--cwd", type=Path, default=Path.cwd())
    run_parser.add_argument("--seeds", help="comma-separated seeds; omit for one unseeded worker")
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--max-runtime-seconds", type=float, required=True)
    run_parser.add_argument("--retrieval-reserve-seconds", type=float, required=True)
    run_parser.add_argument("--term-grace-seconds", type=float, default=30.0)
    run_parser.add_argument("--poll-seconds", type=float, default=2.0)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("command", nargs=argparse.REMAINDER, help="scientific argv after --")
    status_parser = subparsers.add_parser("status", help="print durable status")
    status_parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "status":
            return status(args)
        command = args.command
        if command and command[0] == "--":
            command = command[1:]
        return run(args, command)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
