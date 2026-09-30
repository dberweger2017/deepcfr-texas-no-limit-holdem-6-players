"""One CPU-only pod's fixed mature workload; no provider credentials or rental."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_limit(paths):
    for path in paths:
        try:
            value = Path(path).read_text().strip()
            if value != "max" and 0 < int(value) < 2**60:
                return int(value)
        except (FileNotFoundError, ValueError, PermissionError):
            continue
    return None


def owned_rss(pid, proc=Path("/proc")):
    """Sum the child and its descendants, never all physical-host processes."""
    records = {}
    for path in proc.glob("[0-9]*/status"):
        try:
            fields = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
            records[int(path.parent.name)] = (int(fields["PPid"]), int(fields.get("VmRSS", "0 kB").split()[0])*1024)
        except (FileNotFoundError, ProcessLookupError, KeyError, ValueError, PermissionError):
            continue
    owned = {pid, os.getpid()}
    previous = set()
    while owned != previous:
        previous = set(owned)
        owned.update(child for child, (parent, _) in records.items() if parent in owned)
    return sum(size for child, (_, size) in records.items() if child in owned)


def guard_reason(rss, memory_limit, swap_growth, free, now, deadline, limits):
    rss_cap = min(limits["max_owned_rss_gib"]*2**30,
                  memory_limit*limits["max_rss_fraction_of_container_memory"])
    if rss >= rss_cap:
        return "Owned RSS / container headroom guard"
    if swap_growth > limits["max_swap_growth_gib"]*2**30:
        return "Swap growth guard"
    if free < limits["min_free_disk_gib"]*2**30:
        return "Free disk guard"
    if now >= deadline:
        return "Immutable rental/work deadline guard"
    return None


def stop_child(child):
    """Close our process group even when the resource monitor itself fails."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def execute(args):
    if sys.platform != "linux" or platform.python_version() != "3.11.14":
        raise ValueError("Pinned Linux Python 3.11.14 required")
    plan = json.loads(args.plan.read_text())
    if plan["budget_approval"].startswith("pending"):
        raise ValueError("Owner rental-budget approval is still pending")
    runtime = args.runtime.resolve()
    revision = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    if revision != plan["validated_trainer_source"]:
        raise ValueError("Unvalidated trainer source")
    memory = read_limit(("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    if memory is None:
        raise ValueError("Cannot verify actual container memory allocation")
    swap_paths = ("/sys/fs/cgroup/memory.swap.current",)
    swap_before = read_limit(swap_paths) or 0
    # v1 exposes combined memory+swap; subtract memory usage when needed.
    def swap_usage():
        v2 = Path(swap_paths[0])
        if v2.exists():
            return int(v2.read_text())
        combined = read_limit(("/sys/fs/cgroup/memory/memory.memsw.usage_in_bytes",))
        resident = read_limit(("/sys/fs/cgroup/memory/memory.usage_in_bytes",))
        if combined is not None and resident is not None:
            return max(0, combined-resident)
        raise ValueError("Cannot verify container swap usage")
    swap_before = swap_usage()
    args.out.mkdir(parents=True, exist_ok=False)
    root = args.out.resolve()
    state = {"status": "running", "started": time.time(), "deadline": args.deadline,
             "runtime_source": revision, "plan_sha256": sha256(args.plan.read_bytes()).hexdigest(),
             "driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
             "memory_limit_bytes": memory, "kernel": platform.uname()._asdict(), "affinity_logical_cpus": sorted(os.sched_getaffinity(0)),
             "attempts": []}
    (root / "lscpu.txt").write_text(subprocess.check_output(["lscpu"], text=True))
    (root / "cpu-topology.txt").write_text(subprocess.check_output(["lscpu", "-e=CPU,CORE,SOCKET,ONLINE"], text=True))
    (root / "cpuinfo.txt").write_text(Path("/proc/cpuinfo").read_text())
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                 "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
                 "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
                 "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us",
                 "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.swap.max"):
        p = Path(path)
        if p.exists():
            (root / (p.name + ".txt")).write_text(p.read_text())
    write(root / "worker.json", state)
    direct = [sys.executable, "-m", "scripts.benchmark_observation_reuse", "--variant", plan["variant"],
              "--nodes", str(plan["added_complete_nodes"]), "--deadline", str(args.deadline),
              "--out", str(root / "direct"), "--parent", str(args.parent.resolve()),
              "--parent-sha256", plan["parent"]["checkpoint_sha256"]]
    resumed = [sys.executable, "-m", "scripts.benchmark_observation_reuse", "--variant", plan["variant"],
               "--nodes", str(plan["added_complete_nodes"]), "--deadline", str(args.deadline),
               "--out", str(root / "resumed"), "--resume", str(root / "direct")]
    check = [sys.executable, "-m", "scripts.run_observation_reuse_benchmark", "verify", "--resumed",
             "--left", str(root / "direct"), "--right", str(root / "resumed"),
             "--out", str(root / "resume-verification.json")]
    for name, command in (("direct", direct), ("resumed", resumed), ("verify-resume", check)):
        attempt = {"name": name, "command": command, "started": time.time(), "status": "running"}
        state["attempts"].append(attempt)
        with (root / (name + ".log")).open("w") as log:
            if time.time() >= args.deadline:
                attempt.update(finished=time.time(), exit_code=None,
                               guard_failure="Immutable rental/work deadline guard", status="failed")
                write(root / "worker.json", state)
                break
            child = subprocess.Popen(command, cwd=runtime, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            attempt["pid"] = child.pid
            write(root / "worker.json", state)
            reason = None
            try:
                while child.poll() is None:
                    sample = {"time": time.time(), "phase": name, "owned_rss_bytes": owned_rss(child.pid),
                              "swap_growth_bytes": swap_usage()-swap_before,
                              "free_disk_bytes": shutil.disk_usage(root).free,
                              "cgroup_memory_current_bytes": read_limit(("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")),
                              "cgroup_memory_peak_bytes": read_limit(("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"))}
                    with (root / "resources.jsonl").open("a") as stream:
                        stream.write(json.dumps(sample, sort_keys=True)+"\n")
                    reason = guard_reason(sample["owned_rss_bytes"], memory, sample["swap_growth_bytes"],
                                          sample["free_disk_bytes"], sample["time"], args.deadline, plan["limits"])
                    if reason:
                        stop_child(child)
                        break
                    time.sleep(1)
            except Exception as error:
                reason = f"Resource-monitor failure: {type(error).__name__}: {error}"
                stop_child(child)
            attempt.update(finished=time.time(), exit_code=child.wait(), guard_failure=reason,
                           status="complete" if child.returncode == 0 and reason is None else "failed")
        write(root / "worker.json", state)
        if attempt["status"] != "complete":
            break
    state.update(finished=time.time(), status="complete" if len(state["attempts"]) == 3
                 and all(a["status"] == "complete" for a in state["attempts"]) else "incomplete")
    write(root / "worker.json", state)
    manifest = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            manifest[str(path.relative_to(root))] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    write(root.with_name(root.name+"-manifest.json"), manifest)
    return state["status"] == "complete"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("plan", "runtime", "out", "parent"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    raise SystemExit(0 if execute(parser.parse_args()) else 1)
