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

from scripts.hu20_scaling_common import acquire, identity, inventory
from scripts.run_tp20_campaign import swap_bytes
from scripts.tp20_common import append
from scripts.train_hu20 import system, write_json


def run(jobs, out, deadline, swap_before=None, coordinator_pid=None, require_ac=False):
    acquire(out)
    record = {"status": "running", "started": time(), "deadline": deadline,
              "identity": identity(), "attempts": [],
              "swap_baseline": swap_before or system(["sysctl", "vm.swapusage"])}
    write_json(out / "campaign.json", record)
    for job in jobs:
        attempt = {"name": job["name"], "command": job["command"], "status": "running", "started": time()}
        record["attempts"].append(attempt); write_json(out / "campaign.json", record)
        end = min(deadline, job.get("deadline", deadline)); reason = None; peak = aggregate_peak = 0
        with (out / f'{job["name"]}.log').open("w") as log:
            child = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            attempt["pid"] = child.pid; write_json(out / "campaign.json", record)
            next_sample = 0
            while child.poll() is None:
                if time() >= next_sample:
                    listing = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
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
                    if sum(sizes) >= 10.5*1024**3: reason = "Aggregate job RSS guard"
                    if growth > .5*1024**3: reason = "Swap growth guard"
                    if free < 8*1024**3: reason = "Free disk guard"
                    if require_ac and (power is None or "AC Power" not in power): reason="Main worker AC power guard"
                    next_sample = time()+5
                if time() >= end: reason = "Absolute phase/deadline guard"
                if reason:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
                    break
                sleep(1)
            attempt.update(exit_code=child.wait(), guard_failure=reason, finished=time(),
                           peak_process_rss_bytes=peak, peak_aggregate_job_rss_bytes=aggregate_peak)
        attempt["status"] = "complete" if attempt["exit_code"] == 0 and not reason else "failed"
        write_json(out / "campaign.json", record)
        if attempt["status"] != "complete": break
    record.update(status="complete" if len(record["attempts"]) == len(jobs)
                  and all(a["status"] == "complete" for a in record["attempts"]) else "incomplete", finished=time())
    write_json(out / "campaign.json", record)
    write_json(out.with_name(out.name+"-inventory.json"), inventory(out))
    return record


def main():
    p = argparse.ArgumentParser(); p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--swap-baseline"); p.add_argument("--coordinator-pid",type=int)
    p.add_argument("--require-ac",action="store_true"); a = p.parse_args()
    record = run(json.loads(a.jobs.read_text()), a.out, a.deadline, a.swap_baseline,a.coordinator_pid,a.require_ac)
    print(json.dumps(record)); return record["status"] != "complete"


if __name__ == "__main__": raise SystemExit(main())
