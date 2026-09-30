"""One sequential M4 engineering window with external owned-job guards."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from time import sleep, time


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_hash(path):
    result = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def swap():
    output = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    return float(output.split("used = ", 1)[1].split("M", 1)[0])


def owned_rss(child):
    processes = []
    for line in subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True).splitlines():
        pid, parent, memory = map(int, line.split())
        processes.append((pid, parent, memory))
    owned = {os.getpid(), child}
    while True:
        expanded = owned | {pid for pid, parent, _ in processes if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    return (sum(memory * 1024 for pid, _, memory in processes if pid in owned),
            [{"pid": pid, "rss_bytes": memory * 1024} for pid, _, memory in processes
             if pid not in owned and memory * 1024 > 512 * 1024**2])


def guard(out, clock, child=0):
    (memory, foreign), swap_now, free = owned_rss(child), swap(), shutil.disk_usage(out).free
    sample = {"time": time(), "owned_rss_bytes": memory, "swap_mib": swap_now,
              "free_disk_bytes": free, "child_pid": child, "foreign_large_processes": foreign}
    if foreign:
        raise RuntimeError(f"Unexpected large unowned M4 process; coordinate before proceeding: {foreign}")
    if sample["time"] >= clock["deadline"]:
        raise TimeoutError("Original four-hour engineering deadline")
    if memory > 10.5 * 1024**3:
        raise MemoryError("Aggregate owned-job RSS exceeds 10.5 GiB")
    if swap_now - clock["swap_start_mib"] > 512:
        raise MemoryError("Swap growth exceeds 0.5 GiB")
    if free < 8 * 1024**3:
        raise OSError("Free disk below 8 GiB")
    if "AC Power" not in subprocess.check_output(["pmset", "-g", "batt"], text=True):
        raise RuntimeError("M4 is no longer on AC power")
    return sample


def inputs(args):
    checked = {}
    manifest = json.loads(args.parent_manifest.read_text())
    for original, record in manifest["files"].items():
        path = Path(original)
        actual = file_hash(path)
        if actual != record["sha256"]:
            raise ValueError(f"Sealed #121 input changed: {path}")
        checked[str(path)] = actual
    # #123 has a result/attempt hash seal in the published report, not a global manifest.
    profile = Path("/Users/dberweger/Local/hu20-lbr-kernel-profile-pr123/results/hu20-lbr-kernel-profile-m4-20260930-attempt-2")
    for name, expected in (("result.json", "24c19f9a704891144829940c2e1e98c198430839dae19e36d09d24b1f5eef545"),
                           ("attempts.jsonl", "5363a8ee329b556b4bcc96c923d877d6710c183579b7979db592c977207b9c11")):
        path = profile / name
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"Sealed #123 profile changed: {path}")
        checked[str(path)] = actual
    corpus = json.loads(args.corpus.read_text())
    for filename, expected in corpus["source_hashes"].items():
        path = args.raw_dir / filename
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"Wrong retained raw archive/hash: {path}")
        checked[str(path)] = actual
    models = [model for model in json.loads(args.models.read_text())
              if model.get("arm") == "B" and model.get("milestone") == 100000000]
    if {model["seed"] for model in models} != {2026093001, 2026093002, 2026093003} or len(models) != 3:
        raise ValueError("Saved B100M roster changed")
    if file_hash(args.models) != "d15c1fe9f42c0fbeffa7cb27d3bb4c5cb76226f1d812133dcedda9118d1d2402":
        raise ValueError("Sealed #121 model-spec byte hash changed")
    for model in models:
        for key, hash_key in (("path", "sha256"), ("checkpoint_path", "checkpoint_sha256")):
            path = Path(model[key])
            actual = file_hash(path)
            if actual != model[hash_key]:
                raise ValueError(f"Saved B100M model/checkpoint changed: {path}")
            checked[str(path)] = actual
    for path in (args.models, args.corpus, args.selection, args.parent_manifest):
        checked[str(path)] = file_hash(path)
    return {"files": checked, "models": models}


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "corpus", "selection", "models", "raw-dir", "parent-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.source_head or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Engineering source must be frozen and clean")
    start = time()
    clock = {"started": start, "deadline": start + 14400, "source_head": head,
             "swap_start_mib": swap(), "owner_pid": os.getpid(), "window_seconds": 14400}
    write(args.root / "engineering-clock.json", clock)
    state = {"status": "running", "clock": clock, "phases": [], "peak_owned_rss_bytes": 0}
    child = None
    coordination = Path("/tmp/DR_RESEARCH_M4_COORDINATION.txt")
    with coordination.open("a") as handle:
        handle.write(f"\nDr Research PR #126 CLAIM: task exact-ranker, supervisor PID {os.getpid()}, checkout {Path.cwd()}, start {start}, immutable deadline {clock['deadline']}, one heavy child.\n")
    try:
        guard(args.root, clock)
        write(args.root / "verified-inputs.json", inputs(args))
        common = ["--corpus", str(args.corpus), "--selection", str(args.selection),
                  "--raw-dir", str(args.raw_dir), "--models", str(args.models),
                  "--deadline", str(clock["deadline"])]
        phases = [("focused-tests", [sys.executable, "-m", "pytest", "-q", "tests/test_exact_lbr_ranker.py", "tests/test_cached_lbr.py"])]
        for phase, executor in (("ranks", "candidate"), ("validation", "candidate"),
                                ("bench336", "original"), ("bench336", "candidate"),
                                ("large", "original"), ("large", "candidate")):
            name = phase if phase in ("ranks", "validation") else phase + "-" + executor
            command = [sys.executable, "-m", "scripts.check_exact_lbr_ranker", phase,
                       "--executor", executor, "--out", str(args.root / name)] + common
            phases.append((name, command))
        with (args.root / "resources.jsonl").open("w") as resources:
            for name, command in phases:
                coordination_text = coordination.read_text()
                guard(args.root, clock)
                phase = {"name": name, "started": time(), "command": command,
                         "coordination_sha256": sha256(coordination_text.encode()).hexdigest()}
                state["phases"].append(phase)
                with (args.root / (name + ".log")).open("w") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    phase["pid"] = child.pid
                    with coordination.open("a") as handle:
                        handle.write(f"Dr Research PR #126 phase {name}, child PID {child.pid}, root {args.root}.\n")
                    write(args.root / "supervisor.json", state)
                    while child.poll() is None:
                        sample = guard(args.root, clock, child.pid)
                        resources.write(json.dumps(sample, sort_keys=True) + "\n"); resources.flush()
                        state["peak_owned_rss_bytes"] = max(state["peak_owned_rss_bytes"], sample["owned_rss_bytes"])
                        sleep(1)
                    phase.update({"finished": time(), "exit_code": child.returncode})
                    child = None
                write(args.root / "supervisor.json", state)
                if phase["exit_code"]:
                    raise RuntimeError(f"{name} failed; retained log and partials")
                if name not in ("focused-tests",):
                    result = json.loads((args.root / name / "result.json").read_text())
                    if result["status"] != "complete":
                        raise ValueError(f"{name} incomplete")
        state["status"] = "complete"
    except Exception as exc:
        state["status"] = "failed"
        state["failure"] = f"{type(exc).__name__}: {exc}"
        if child is not None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
    state.update({"finished": time(), "swap_end_mib": swap(),
                  "free_disk_bytes": shutil.disk_usage(args.root).free})
    write(args.root / "supervisor.json", state)
    with coordination.open("a") as handle:
        handle.write(f"Dr Research PR #126 RELEASE: supervisor {os.getpid()} {state['status']}; no owned heavy child remains. Root {args.root}.\n")
    print(json.dumps({key: state.get(key) for key in ("status", "failure", "finished")}))
    if state["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
