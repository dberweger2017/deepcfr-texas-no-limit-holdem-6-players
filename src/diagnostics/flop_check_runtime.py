"""File-boundary solver execution with machine-wide swap/RSS guards."""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from time import monotonic, sleep

from src.diagnostics.flop_check import atomic_json
from src.diagnostics.saved_hu20 import file_hash


def machine_snapshot():
    if sys.platform != "darwin":
        raise ValueError("This resource admission profile is for the owner-approved Macs")
    def command(*args):
        return subprocess.check_output(args, text=True)
    vm = command("vm_stat")
    page_size = int(re.search(r"page size of (\d+)", vm).group(1))
    pages = {name: int(value) for name, value in re.findall(r"^([^:\n]+):\s*(\d+)\.", vm, re.M)}
    available = page_size * sum(pages.get(name, 0) for name in
                              ("Pages free", "Pages inactive", "Pages speculative"))
    swap_text = command("sysctl", "vm.swapusage")
    swap = int(float(re.search(r"used = ([\d.]+)M", swap_text).group(1)) * 1024**2)
    processes = command("ps", "-axo", "pid,ppid,pcpu,rss,etime,comm")
    pressure = command("memory_pressure")
    return {"platform": sys.platform, "hostname": command("hostname").strip(),
            "physical_bytes": int(command("sysctl", "-n", "hw.memsize")),
            "cores": int(command("sysctl", "-n", "hw.ncpu")),
            "reclaimable_bytes": available, "swap_used_bytes": swap,
            "vm_stat": vm, "memory_pressure": pressure, "processes": processes}


def rss_for_tree(pid):
    rows = subprocess.check_output(["ps", "-axo", "pid,ppid,rss"], text=True).splitlines()[1:]
    parsed = [tuple(map(int, line.split())) for line in rows]
    owned = {pid}
    while True:
        more = {p for p, parent, _ in parsed if parent in owned}
        if more <= owned:
            break
        owned |= more
    return sum(rss * 1024 for p, _, rss in parsed if p in owned)


def append(path, row):
    with Path(path).open("a") as target:
        target.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def swap_usage():
    value = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    return int(float(re.search(r"used = ([\d.]+)M", value).group(1)) * 1024**2)


def run_tool(binary, request_path, out, *, memory_bytes, threads, seconds,
             initial_swap=None, job_memory_bytes=None):
    """One external process group; stop only owned work on any guard failure."""
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    initial = machine_snapshot()
    baseline = initial["swap_used_bytes"] if initial_swap is None else initial_swap
    job_budget = memory_bytes if job_memory_bytes is None else job_memory_bytes
    if (not 1 <= threads <= initial["cores"] or not 0 < memory_bytes <= 10 * 1024**3
            or not memory_bytes <= job_budget <= 10 * 1024**3):
        raise ValueError("Invalid M4 worker budget")
    request = json.loads(Path(request_path).read_text())
    if request["memory_budget_bytes"] != memory_bytes:
        raise ValueError("Request and watchdog memory budgets differ")
    if request.get("dump_path") and Path(request["dump_path"]).exists():
        raise FileExistsError("Preserve the existing external profile dump")
    atomic_json(out / "machine-before.json", initial)
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    response = out / "response.jsonl"; started = monotonic(); failure = None; seen = 0
    peak_rss = 0; peak_swap = baseline; resource_time = 0
    with (out / "stderr.log").open("w") as error:
        process = subprocess.Popen(["nice", "-n", "10", str(binary),
                                    str(request_path), str(response)], env=env,
                                   stderr=error, stdout=error, start_new_session=True)
        try:
            while True:
                rss = rss_for_tree(os.getpid())
                swap = swap_usage(); peak_rss = max(peak_rss, rss); peak_swap = max(peak_swap, swap)
                if monotonic() - resource_time >= 5:
                    append(out / "progress.jsonl", {"event": "resources", "spot": request["spot"],
                           "rss_bytes": rss, "swap_used_bytes": swap,
                           "elapsed_seconds": monotonic() - started})
                    resource_time = monotonic()
                if rss > job_budget:
                    failure = "RSS budget exceeded"
                elif swap - baseline > 1024**3:
                    failure = "Swap growth exceeds 1 GiB"
                elif monotonic() - started > seconds:
                    failure = "External solver wall deadline"
                if response.exists():
                    with response.open("rb") as source:
                        source.seek(seen)
                        for line in source:
                            if not line.endswith(b"\n"):
                                break
                            seen += len(line); row = json.loads(line)
                            row.update(spot=request["spot"], rss_bytes=rss)
                            append(out / "progress.jsonl", row)
                            if row.get("event") == "gate" and not row["passed"]:
                                failure = f'{row["gate"]} failed'
                if failure:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    break
                if process.poll() is not None:
                    break
                sleep(0.5)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM); process.wait()
            raise
    if not failure and process.returncode != 0:
        failure = f"External solver exit {process.returncode}"
    result = {"status": "failure" if failure else "completed", "failure": failure,
              "exit_code": process.returncode, "elapsed_seconds": monotonic() - started,
              "binary_sha256": file_hash(binary), "request_sha256": file_hash(request_path),
              "response_sha256": file_hash(response) if response.exists() else None,
              "profile_sha256": file_hash(request["dump_path"])
                  if request.get("dump_path") and Path(request["dump_path"]).exists() else None,
              "memory_budget_bytes": memory_bytes, "rayon_threads": threads,
              "job_memory_budget_bytes": job_budget,
              "peak_job_rss_bytes": peak_rss, "peak_swap_used_bytes": peak_swap,
              "swap_baseline_bytes": baseline}
    atomic_json(out / ("failure.json" if failure else "result.json"), result)
    return result
