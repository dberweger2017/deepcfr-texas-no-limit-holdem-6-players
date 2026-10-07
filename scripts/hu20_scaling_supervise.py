"""One sequential heavy child per host, with retained attempts and resource guards."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from time import sleep, time
from math import isfinite

from scripts.hu20_scaling_common import acquire, identity, inventory
from scripts.run_tp20_campaign import swap_bytes
from scripts.tp20_common import append
from scripts.train_hu20 import system, write_json


TERMINATION_GRACE_SECONDS = 5


def terminate_child(child):
    """Stop only the session created by this supervisor, including surviving descendants."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    # The direct child may have exited while a descendant ignored TERM. The session
    # belongs to this attempt, so escalation covers that group even after parent exit.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=TERMINATION_GRACE_SECONDS)


def run(jobs, out, deadline, swap_before=None, coordinator_pid=None, require_ac=False, rss_gib=10.5, disk_gib=8, swap_gib=.5):
    if (not all(isfinite(n) for n in (deadline, rss_gib, disk_gib, swap_gib))
        or rss_gib <= 0 or disk_gib <= 0 or swap_gib < 0):
        raise ValueError("Finite resource limits must be positive (swap may be zero)")
    acquire(out)
    record = {"status": "running", "started": time(), "deadline": deadline,
              "identity": None, "attempts": [], "failure": None,
              "swap_baseline": None,
              "limits": {"rss_gib": rss_gib, "disk_gib": disk_gib, "swap_gib": swap_gib}}
    received_signal = [None]
    def interrupted(signum, _frame):
        # Do not raise during Popen: ownership must be recorded before cleanup can run.
        received_signal[0] = signal.Signals(signum).name
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous: signal.signal(signum, interrupted)
    child = None; cleaned_child = False; attempt = None; reason = None; peak = aggregate_peak = 0
    try:
        write_json(out / "campaign.json", record)
        record["identity"] = identity()
        record["swap_baseline"] = swap_before or system(["sysctl", "vm.swapusage"])
        for job in jobs:
            if received_signal[0]: break
            child = None; cleaned_child = False; reason = None; peak = aggregate_peak = 0
            attempt = {"name": job["name"], "command": job["command"], "status": "running", "started": time()}
            record["attempts"].append(attempt); write_json(out / "campaign.json", record)
            end = min(deadline, job.get("deadline", deadline))
            if not isfinite(end): raise ValueError("Phase deadline must be finite")
            # Refuse admission before starting a child on a host without adequate resources.
            power = system(["pmset", "-g", "batt"])
            reason = ("Absolute phase/deadline guard" if time() >= end else
                      "Free disk guard" if shutil.disk_usage(out).free < disk_gib*1024**3 else
                      "Main worker AC power guard" if require_ac and (power is None or "AC Power" not in power) else None)
            if reason or received_signal[0]: break
            with (out / f'{job["name"]}.log').open("w") as log:
                try:
                    child = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    attempt["pid"] = child.pid; write_json(out / "campaign.json", record)
                    next_sample = 0
                    while child.poll() is None:
                        if received_signal[0]: break
                        if time() >= next_sample:
                            listing = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True, timeout=10)
                            processes = [tuple(map(int, line.split())) for line in listing.splitlines() if line.strip()]
                            owned = {os.getpid(), child.pid}
                            if coordinator_pid: owned.add(coordinator_pid)
                            for _ in range(4): owned.update(pid for pid, parent, _ in processes if parent in owned)
                            sizes = [size*1024 for pid, _, size in processes if pid in owned]
                            peak = max(peak, max(sizes, default=0)); aggregate_peak = max(aggregate_peak, sum(sizes))
                            swap = system(["sysctl", "vm.swapusage"])
                            growth = swap_bytes(swap)-swap_bytes(record["swap_baseline"])
                            free = shutil.disk_usage(out).free
                            power=system(["pmset","-g","batt"])
                            append(out / "resources.jsonl", {"unix_seconds": time(), "phase": job["name"],
                                   "rss_bytes": max(sizes, default=0), "aggregate_job_rss_bytes": sum(sizes),
                                   "swap": swap, "swap_growth_bytes": growth, "free_disk_bytes": free,"power":power})
                            if sum(sizes) >= rss_gib*1024**3: reason = "Aggregate job RSS guard"
                            if growth > swap_gib*1024**3: reason = "Swap growth guard"
                            if free < disk_gib*1024**3: reason = "Free disk guard"
                            if require_ac and (power is None or "AC Power" not in power): reason="Main worker AC power guard"
                            next_sample = time()+5
                        if time() >= end: reason = "Absolute phase/deadline guard"
                        if reason or received_signal[0]: break
                        sleep(1)
                finally:
                    if child is not None:
                        try:
                            terminate_child(child)
                        except BaseException as cleanup_error:
                            attempt["cleanup_failure"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                            raise
                        cleaned_child = True
            attempt.update(exit_code=child.returncode, guard_failure=reason, finished=time(),
                           peak_process_rss_bytes=peak, peak_aggregate_job_rss_bytes=aggregate_peak)
            attempt["status"] = "complete" if child.returncode == 0 and not reason and not received_signal[0] else "failed"
            write_json(out / "campaign.json", record)
            if attempt["status"] != "complete": break
    except BaseException as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, KeyboardInterrupt): received_signal[0] = "KeyboardInterrupt"
    finally:
        try:
            # Includes failures while publishing the PID, monitoring, writing receipts or opening logs.
            if child is not None and not cleaned_child:
                try:
                    terminate_child(child)
                except BaseException as cleanup_error:
                    record["cleanup_failure"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                    record["failure"] = record["failure"] or record["cleanup_failure"]
                    if attempt is not None: attempt["cleanup_failure"] = record["cleanup_failure"]
            if attempt is not None and attempt["status"] == "running":
                attempt.update(status="failed", exit_code=child.returncode if child else None,
                               guard_failure=reason, finished=time(),
                               peak_process_rss_bytes=peak, peak_aggregate_job_rss_bytes=aggregate_peak)
            if received_signal[0]:
                record["failure"] = f"Supervisor interrupted: {received_signal[0]}"
                if attempt is not None:
                    attempt.update(status="interrupted", guard_failure=record["failure"])
            elif attempt is not None and record["failure"]:
                attempt.update(status="failed", guard_failure=record["failure"])
            record.update(status="interrupted" if received_signal[0] else
                          "complete" if not record["failure"] and len(record["attempts"]) == len(jobs)
                          and all(a["status"] == "complete" for a in record["attempts"]) else "incomplete", finished=time())
            write_json(out / "campaign.json", record)
            write_json(out.with_name(out.name+"-inventory.json"), inventory(out))
        finally:
            for signum, handler in previous.items(): signal.signal(signum, handler)
    return record


def main():
    p = argparse.ArgumentParser(); p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--swap-baseline"); p.add_argument("--coordinator-pid",type=int)
    p.add_argument("--require-ac",action="store_true")
    p.add_argument("--rss-gib", type=float, default=10.5)
    p.add_argument("--disk-gib", type=float, default=8)
    p.add_argument("--swap-gib", type=float, default=.5)
    a = p.parse_args()
    record = run(json.loads(a.jobs.read_text()), a.out, a.deadline, a.swap_baseline,a.coordinator_pid,a.require_ac,a.rss_gib,a.disk_gib,a.swap_gib)
    print(json.dumps(record)); return record["status"] != "complete"


if __name__ == "__main__": raise SystemExit(main())
