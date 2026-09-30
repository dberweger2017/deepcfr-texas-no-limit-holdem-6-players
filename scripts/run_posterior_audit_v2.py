"""Detached, sequential M4 supervision under one immutable 9.5-hour clock."""

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import sleep, time
from zoneinfo import ZoneInfo

from scripts.run_exact_ranker_experiment import file_hash, guard, inputs, swap
from src.diagnostics.posterior_audit_v2 import atomic_json

BASE = "7d31a39c80cc72deb772336ce251f3c4bdcd46c9"
PARENT_HEAD = "b49c1d9b231bd3f298e89a89f7d848a26fef4de1"
REPORT_HASH = "ea9ea795e68a30e13bea81a9020347ed15380324af54b5ea43e91449db19f52c"
MANIFEST_HASH = "c82f6808163e23b7588f6753147d4933b56bb3b3249abec11356f41b18e70fe6"
COORDINATION = Path("/tmp/DR_RESEARCH_M4_COORDINATION.txt")


def parent_inputs(args):
    checked = inputs(args)
    publication = Path("/Users/dberweger/Local/hu20-exact-lbr-ranker-pr126/results/hu20-exact-ranker-m4-20260930-publication-final")
    for name, expected in (("report.json", REPORT_HASH), ("manifest.json", MANIFEST_HASH)):
        path = publication / name
        if file_hash(path) != expected:
            raise ValueError(f"Merged #126 {name} changed")
        checked["files"][str(path)] = expected
    parent = json.loads((publication / "manifest.json").read_text())
    for original, record in parent["files"].items():
        if file_hash(Path(original)) != record["sha256"]:
            raise ValueError(f"Merged #126 sealed file changed: {original}")
        checked["files"][original] = record["sha256"]
    checked["files"][str(args.plan)] = file_hash(args.plan)
    import pokers
    for path in Path(pokers.__file__).parent.iterdir():
        if path.suffix == ".so" or path.name == "__init__.py":
            checked["files"][str(path)] = file_hash(path)
    checked.update({"merged_base": BASE, "merged_pr_head": PARENT_HEAD,
                    "parent_report_sha256": REPORT_HASH, "parent_manifest_sha256": MANIFEST_HASH})
    return checked


def run(args):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.source_head or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Scientific source must be frozen and clean")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", BASE, head])
    args.root.mkdir(parents=True, exist_ok=args.resume)
    clock_path = args.root / "scientific-clock.json"
    if args.resume:
        clock = json.loads(clock_path.read_text())
        if clock["source_head"] != head or clock["deadline"] != clock["started"] + 34200 or clock["science_deadline"] != clock["started"] + 32400:
            raise ValueError("Original scientific clock/source changed")
        previous = json.loads((args.root / "coordinator.json").read_text())
        if previous["status"] != "running":
            raise ValueError("Completed/failed/stopped audit must not be restarted")
        if previous["owner_pid"] != os.getpid():
            try:
                os.kill(previous["owner_pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError("Coordinator is already alive")
        state = previous
        state["recoveries"].append({"time": time(), "previous_owner": state["owner_pid"], "owner": os.getpid()})
        state["owner_pid"] = os.getpid()
    else:
        started = time()
        clock = {"started": started, "deadline": started + 34200, "science_deadline": started + 32400,
                 "source_head": head, "merged_base": BASE, "window_seconds": 34200,
                 "swap_start_mib": swap(), "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                 "started_madrid": datetime.fromtimestamp(started, ZoneInfo("Europe/Madrid")).isoformat(),
                 "deadline_utc": datetime.fromtimestamp(started+34200, timezone.utc).isoformat(),
                 "deadline_madrid": datetime.fromtimestamp(started+34200, ZoneInfo("Europe/Madrid")).isoformat()}
        atomic_json(clock_path, clock)
        state = {"status": "running", "clock": clock, "owner_pid": os.getpid(), "phases": [],
                 "recoveries": [], "peak_owned_rss_bytes": 0, "minimum_free_disk_bytes": shutil.disk_usage(args.root).free,
                 "maximum_swap_growth_mib": 0, "environment": {"platform": platform.platform(),
                 "python": sys.version, "machine": platform.machine(), "processor": subprocess.check_output(
                     ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()}}
    atomic_json(args.root / "coordinator.json", state)
    with COORDINATION.open("a") as handle:
        handle.write(f"\nDr Research posterior audit v2 CLAIM: coordinator {os.getpid()}, source {head}, checkout {Path.cwd()}, start {clock['started']}, science cutoff {clock['science_deadline']}, hard deadline {clock['deadline']}; one heavy child, 10.5 GiB owned RSS, .5 GiB swap growth, 8 GiB free disk, AC/caffeinate.\n")
    common = ["--root", str(args.root), "--selection", str(args.selection), "--plan", str(args.plan),
              "--raw-dir", str(args.raw_dir), "--models", str(args.models), "--source-head", head]
    phases = [("focused-tests", [sys.executable, "-m", "pytest", "-q", "tests/test_posterior_audit_v2.py",
                                "tests/test_hu20_b100_posterior.py", "tests/test_hu20_b100_diagnosis.py"])]
    phases += [(phase, [sys.executable, "-m", "scripts.audit_hu20_posterior_v2", phase, *common])
               for phase in ("timer", "stability", "main", "suit-likelihood", "values", "river")]
    child = None
    try:
        guard(args.root, clock)
        if not (args.root / "verified-inputs.json").exists():
            atomic_json(args.root / "verified-inputs.json", parent_inputs(args))
        else:
            for filename, expected in json.loads((args.root / "verified-inputs.json").read_text())["files"].items():
                if file_hash(Path(filename)) != expected:
                    raise ValueError(f"Recovery input changed: {filename}")
        with (args.root / "resources.jsonl").open("a") as resources:
            for name, command in phases:
                completed = [p for p in state["phases"] if p["name"] == name and p.get("exit_code") == 0]
                if completed:
                    continue
                if name != "focused-tests":
                    attempt = args.root / f"{name}-attempt.json"
                    if attempt.exists() and json.loads(attempt.read_text())["status"] in ("scientific_stop", "failed"):
                        raise RuntimeError("Retained phase failure/stop; no blind retry")
                guard(args.root, dict(clock, deadline=clock["science_deadline"]))
                coordination = COORDINATION.read_text()
                phase = {"name": name, "command": command, "started": time(),
                         "coordination_sha256": sha256(coordination.encode()).hexdigest(),
                         "process_snapshot": subprocess.check_output(["ps", "-axo", "pid,ppid,%cpu,rss,command"], text=True)}
                state["phases"].append(phase)
                log_path = args.root / f"{name}-{len(state['phases']):02d}.log"
                with log_path.open("x") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    phase["pid"] = child.pid
                    with COORDINATION.open("a") as handle:
                        handle.write(f"Dr Research posterior audit phase {name}, child PID {child.pid}, start {phase['started']}, cutoff {clock['science_deadline']}, root {args.root}.\n")
                    atomic_json(args.root / "coordinator.json", state)
                    last_checkpoint = 0
                    while child.poll() is None:
                        sample = guard(args.root, dict(clock, deadline=clock["science_deadline"]), child.pid)
                        state["peak_owned_rss_bytes"] = max(state["peak_owned_rss_bytes"], sample["owned_rss_bytes"])
                        state["minimum_free_disk_bytes"] = min(state["minimum_free_disk_bytes"], sample["free_disk_bytes"])
                        state["maximum_swap_growth_mib"] = max(state["maximum_swap_growth_mib"], sample["swap_mib"]-clock["swap_start_mib"])
                        resources.write(json.dumps(sample, sort_keys=True)+"\n")
                        resources.flush()
                        if time()-last_checkpoint > 60:
                            atomic_json(args.root / "coordinator.json", state)
                            last_checkpoint = time()
                        sleep(2)
                    phase.update(finished=time(), exit_code=child.returncode)
                    child = None
                atomic_json(args.root / "coordinator.json", state)
                if phase["exit_code"]:
                    record = args.root / f"{name}-attempt.json"
                    detail = json.loads(record.read_text()) if record.exists() else {}
                    state.update(status=detail.get("status", "failed"), reason=detail.get("reason", f"{name} exited {phase['exit_code']}"))
                    break
            else:
                state["status"] = "complete"
    except Exception as exc:
        state.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        if child is not None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            state["phases"][-1].update(finished=time(), exit_code=child.returncode, external_guard_stop=True)
    state.update(finished=time(), swap_end_mib=swap(), free_disk_bytes=shutil.disk_usage(args.root).free)
    atomic_json(args.root / "coordinator.json", state)
    # Reporting uses the same clock, but no new scientific work. The external
    # wrapper seals once this coordinator and its logs have closed.
    with (args.root / "reporting.log").open("x") as log:
        code = subprocess.call([sys.executable, "-m", "scripts.report_posterior_audit_v2", "--root", str(args.root)],
                               stdout=log, stderr=subprocess.STDOUT)
    atomic_json(args.root / "reporting-status.json", {"exit_code": code, "finished": time()})
    with COORDINATION.open("a") as handle:
        handle.write(f"Dr Research posterior audit RELEASE: coordinator {os.getpid()} {state['status']}, reason {state.get('reason')}; no heavy child remains; root {args.root}.\n")
    return 0 if code == 0 else 1


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "corpus", "selection", "models", "raw-dir", "parent-manifest", "plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--resume", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
