"""Owned Linux process guards; no provisioning or paid compute authorization."""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from time import monotonic, sleep

from src.diagnostics.flop_check import atomic_json
from src.diagnostics.flop_check_runtime import append, rss_for_tree
from src.diagnostics.saved_hu20 import file_hash


def linux_snapshot():
    if sys.platform != "linux":
        raise ValueError("Production runtime requires native Linux")
    fields = {name: int(value) * 1024 for name, value in
              re.findall(r"^(\w+):\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M)}
    group = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                 if line.startswith("0::"))
    base = Path("/sys/fs/cgroup") / group.lstrip("/")
    # A cgroup namespace mounts the current group at /sys/fs/cgroup.
    if not (base / "memory.max").exists():
        base = Path("/sys/fs/cgroup")
    cap = (base / "memory.max").read_text().strip()
    limit = fields["MemTotal"] if cap == "max" else min(int(cap), fields["MemTotal"])
    used = int((base / "memory.current").read_text())
    cpu = (base / "cpu.max").read_text().split()
    affinity = len(os.sched_getaffinity(0))
    cores = affinity if cpu[0] == "max" else min(affinity, int(cpu[0]) / int(cpu[1]))
    return {"platform": sys.platform, "physical_bytes": fields["MemTotal"],
            "memory_limit_bytes": limit, "cgroup_used_bytes": used,
            "available_bytes": min(fields["MemAvailable"], max(0, limit - used)),
            "swap_used_bytes": fields["SwapTotal"] - fields["SwapFree"],
            "memory_events": (base / "memory.events").read_text(),
            "effective_cores": cores, "affinity": sorted(os.sched_getaffinity(0)),
            "processes": subprocess.check_output(["ps", "-axo", "pid,ppid,pcpu,rss,etime,comm"], text=True)}


def run_linux_tool(binary, request_path, out, *, memory_bytes, threads, seconds,
                   initial_swap=None, job_memory_bytes=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    initial = linux_snapshot(); baseline = initial["swap_used_bytes"] if initial_swap is None else initial_swap
    budget = memory_bytes if job_memory_bytes is None else job_memory_bytes
    if not 1 <= threads <= initial["effective_cores"] or not 0 < memory_bytes <= budget:
        raise ValueError("Invalid measured Linux worker admission")
    request = json.loads(Path(request_path).read_text())
    if request["memory_budget_bytes"] != memory_bytes:
        raise ValueError("Watchdog/request budgets differ")
    atomic_json(out / "machine-before.json", initial)
    started = monotonic(); response = out / "response.jsonl"; failure = None; peak = 0
    with (out / "stderr.log").open("w") as error:
        process = subprocess.Popen(["nice", "-n", "10", str(binary), str(request_path), str(response)],
                                   env=dict(os.environ, RAYON_NUM_THREADS=str(threads)),
                                   stdout=error, stderr=error, start_new_session=True)
        try:
            while True:
                snapshot = linux_snapshot(); rss = rss_for_tree(os.getpid()); peak = max(peak, rss)
                append(out / "resources.jsonl", {"rss_bytes": rss, "elapsed_seconds": monotonic() - started,
                                                "swap_used_bytes": snapshot["swap_used_bytes"],
                                                "cgroup_used_bytes": snapshot["cgroup_used_bytes"]})
                if rss > budget:
                    failure = "worker RSS budget exceeded"
                elif snapshot["swap_used_bytes"] - baseline > 1024**3:
                    failure = "swap growth exceeds 1 GiB"
                elif snapshot["memory_events"] != initial["memory_events"]:
                    failure = "cgroup memory event changed"
                elif monotonic() - started > seconds:
                    failure = "worker deadline exceeded"
                if failure or process.poll() is not None:
                    break
                sleep(.5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
    if process.returncode and not failure:
        failure = f"solver exit {process.returncode}"
    rows = [] if not response.exists() else [json.loads(line) for line in response.read_text().splitlines()]
    if not failure and (not rows or rows[-1].get("event") != "completion"):
        failure = "missing atomic solver completion"
    if not failure and any(r.get("event") == "gate" and not r["passed"] for r in rows):
        failure = "validation gate failed"
    result = {"status": "failure" if failure else "completed", "failure": failure,
              "elapsed_seconds": monotonic() - started, "peak_job_rss_bytes": peak,
              "binary_sha256": file_hash(binary), "request_sha256": file_hash(request_path),
              "response_sha256": file_hash(response) if response.exists() else None,
              "rayon_threads": threads, "memory_budget_bytes": memory_bytes,
              "job_memory_budget_bytes": budget, "swap_baseline_bytes": baseline}
    atomic_json(out / ("failure.json" if failure else "result.json"), result)
    return result


def run_portable_tool(*args, **kwargs):
    if sys.platform == "linux":
        return run_linux_tool(*args, **kwargs)
    from src.diagnostics.flop_check_runtime import run_tool
    return run_tool(*args, **kwargs)
