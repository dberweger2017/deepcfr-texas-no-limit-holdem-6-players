"""Owned-process guards and an append-preserving cumulative experiment clock."""

import json
import fcntl
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
from time import monotonic, time


def install_stop_handlers():
    def stop(signum, frame):
        raise RuntimeError(f"Experiment interrupted by signal {signum}; retain partial evidence")
    for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,stop)


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def owned_rss():
    text = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    records = {int(pid): (int(parent), int(rss)*1024)
               for pid, parent, rss in (line.split() for line in text.splitlines())}
    owned = {os.getpid()}
    while True:
        expanded = owned | {pid for pid, (parent, _) in records.items() if parent in owned}
        if expanded == owned: break
        owned = expanded
    return sum(records.get(pid, (0, 0))[1] for pid in owned)


def swap_bytes():
    if sys.platform == "darwin":
        text = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        match = re.search(r"used = ([\d.]+)([KMG])", text)
        if not match: raise ValueError("Cannot measure macOS swap")
        return int(float(match[1]) * 1024**("KMG".index(match[2])+1))
    cgroup = Path("/sys/fs/cgroup/memory.swap.current")
    if cgroup.exists(): return int(cgroup.read_text())
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return (int(values["SwapTotal"].split()[0])-int(values["SwapFree"].split()[0]))*1024


def macos_memory_admission(vm_stat, *, sidecar_bytes=512*1024**2):
    """Owner-approved cache-inclusive law for the calibration readmission."""
    page = int(re.search(r"page size of (\d+)", vm_stat).group(1))
    counts = {k: int(v) for k, v in re.findall(r"^([^:\n]+):\s*(\d+)\.", vm_stat, re.M)}
    keys = ("Pages free", "Pages inactive", "Pages speculative", "File-backed pages")
    if any(k not in counts for k in keys):
        raise ValueError("Missing memory admission component")
    components = {k: page*counts[k] for k in keys}
    reclaimable = sum(components.values())
    limit = min(8*1024**3, int(.8*reclaimable)-sidecar_bytes)
    if limit <= 0:
        raise MemoryError("No family headroom after sidecar reservation")
    return {"memory_components_bytes": components, "page_size_bytes": page,
            "reclaimable_bytes": reclaimable, "rss_limit_bytes": limit,
            "sidecar_reserved_bytes": sidecar_bytes,
            "reclaimable_formula": "free + inactive + speculative + file-backed",
            "family_cap_formula": "min(8 GiB, 0.8 * reclaimable - 0.5 GiB sidecar)"}


def validate_admission(admission, *, now=None):
    now = time() if now is None else now
    fields = (("148_merged", "148_processes_empty", "149_owner_authorized")
              if admission.get("experiment") == "hu20-board-pooling" else
              ("145_main_complete", "145_final_report_pushed", "145_processes_empty"))
    for field in fields:
        if admission.get(field) is not True: raise ValueError("#145 has not released M4: " + field)
    if admission.get("followup_claim") != "none":
        raise ValueError("M4 follow-up ownership needs owner clarification")
    if not 0 <= now-admission.get("checked_at", 0) <= 300:
        raise ValueError("Fresh ownership/resource admission is required")
    rss = admission.get("rss_limit_bytes", 0)
    if not 0 < rss <= min(10*1024**3, .8*admission.get("reclaimable_bytes", 0)):
        raise ValueError("RSS admission exceeds measured headroom or owner ceiling")
    if rss+admission.get("sidecar_reserved_bytes", 0) > .8*admission.get("reclaimable_bytes", 0):
        raise ValueError("Family plus sidecar exceeds measured headroom")
    if not admission.get("ownership_evidence") or not (admission.get("experiment") == "hu20-board-pooling" or admission.get("145_report_sha256")):
        raise ValueError("Admission needs final-report and ownership evidence")


def native_allocation_budget(budget, requested_bytes):
    """Reserve family headroom before the native process allocates its tree."""
    budget.check()
    admission = budget.admission if hasattr(budget, "admission") else budget.approval
    reserve = admission.get("solver_allocation_reserve_bytes", 256*1024**2)
    if type(reserve) is not int or not 0 < reserve < admission["rss_limit_bytes"]:
        raise ValueError("Invalid native allocation reserve")
    if type(requested_bytes) is not int or requested_bytes <= 0:
        raise ValueError("Invalid configured native memory ceiling")
    available = admission["rss_limit_bytes"]-owned_rss()-reserve
    # Round down; the reserve covers native input/tree overhead and transport.
    available = max(0, available // 1024**2 * 1024**2)
    return min(requested_bytes, available)


def start_resource_watchdog(budget):
    """Guard blocking imports/loads as well as cooperative hand/solver checks."""
    stop=threading.Event()
    limit=(budget.admission if hasattr(budget,"admission") else budget.approval)["rss_limit_bytes"]
    def watch():
        while not stop.wait(.25):
            try:
                if monotonic()>=budget.deadline:raise TimeoutError("Frozen phase/cumulative clock expired during blocking work")
                rss=owned_rss();budget.peak_rss=max(budget.peak_rss,rss)
                if rss>=limit:raise MemoryError("Owned family RSS guard during blocking work")
                if swap_bytes()-budget.swap_baseline>1024**3:raise MemoryError("Swap growth guard during blocking work")
                if shutil.disk_usage(budget.out).free<budget.admission.get("minimum_disk_free_bytes",8*1024**3):raise OSError("Free disk guard during blocking work")
            except Exception as exc:
                if stop.is_set():return
                atomic_json(budget.out/("resource-guard-failure-"+str(os.getpid())+".json"),{
                    "cause":type(exc).__name__,"reason":str(exc),"pid":os.getpid(),
                    "elapsed_seconds":monotonic()-budget.started,"peak_owned_rss_bytes":budget.peak_rss})
                if not stop.is_set():os.kill(os.getpid(),signal.SIGTERM)
                return
    thread=threading.Thread(target=watch,name="hu20-resource-watchdog",daemon=True)
    thread.start()
    def close():
        stop.set();thread.join(timeout=2)
    return close


class RunBudget:
    def __init__(self, path, out, phase, phase_seconds, admission):
        validate_admission(admission)
        self.path = path
        self.out = out
        self.phase = phase
        self.phase_seconds = phase_seconds
        self.admission = admission
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = path.with_suffix(".lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.data = json.loads(path.read_text()) if path.exists() else {
            "limit_seconds": 86400, "used_seconds": 0, "attempts": []}
        if self.data["limit_seconds"] != 86400:
            raise ValueError("Cumulative allowance cannot change")
        if self.data.get("active"):
            raise ValueError("Retain/reconcile the prior active attempt before resuming")
        remaining = 86400-self.data["used_seconds"]
        if phase in ("pilot", "part-a", "part-a-verification"):
            used = sum(a["seconds"] for a in self.data["attempts"]
                       if a["phase"] in ("pilot", "part-a", "part-a-verification"))
            remaining = min(remaining, 21600-used)
        if remaining <= 0: raise TimeoutError("Cumulative M4 allowance exhausted")
        self.started = monotonic()
        self.deadline = self.started+min(phase_seconds, remaining)
        self.swap_baseline = swap_bytes()
        self.last_check = 0
        self.peak_rss = 0
        self.record = {"pid": os.getpid(), "phase": phase, "started_at": time(),
                       "swap_baseline_bytes": self.swap_baseline, "out": str(out)}
        self.data["active"] = self.record
        atomic_json(path, self.data)

    def check(self):
        now = monotonic()
        if now >= self.deadline: raise TimeoutError("Frozen phase/cumulative clock expired")
        if now-self.last_check < 1: return
        self.last_check = now
        self.peak_rss = max(self.peak_rss, owned_rss())
        if self.peak_rss >= self.admission["rss_limit_bytes"]: raise MemoryError("Owned family RSS guard")
        if swap_bytes()-self.swap_baseline > 1024**3: raise MemoryError("Swap growth guard")
        if shutil.disk_usage(self.out).free < self.admission.get("minimum_disk_free_bytes", 8*1024**3): raise OSError("Free disk guard")
        self.record["elapsed_seconds"] = now-self.started
        atomic_json(self.path, self.data)

    def close(self, status, reason=None):
        elapsed = monotonic()-self.started
        self.record.update(status=status, reason=reason, seconds=elapsed,
                           peak_owned_rss_bytes=self.peak_rss, stopped_at=time())
        self.data["used_seconds"] += elapsed
        self.data["attempts"].append(self.record)
        self.data.pop("active")
        atomic_json(self.path, self.data)
        self.lock.close()

    def native_allocation_budget(self, requested_bytes):
        return native_allocation_budget(self, requested_bytes)


class PaidWorkerBudget:
    """Approved independent worker clock; does not debit the M4 allowance."""

    def __init__(self, out, approval, plan_sha256, config_sha256):
        if (approval.get("owner_approved") is not True
                or approval.get("arena_plan_sha256") != plan_sha256
                or approval.get("search_config_sha256") != config_sha256
                or approval.get("selected_settings_parity") != "passed"
                or not approval.get("quote_sha256") or not approval.get("parity_sha256")):
            raise ValueError("Approved quote and actual-pod selected-settings parity required")
        if not 0 < approval.get("worker_seconds", 0) or not 0 < approval.get("rss_limit_bytes", 0):
            raise ValueError("Worker must have quoted clock and resource limits")
        self.out = out
        self.approval = approval
        self.started = monotonic()
        self.deadline = self.started + approval["worker_seconds"]
        self.peak_rss = 0
        self.swap_baseline = swap_bytes()
        self.last_check = 0

    def check(self):
        if monotonic() >= self.deadline: raise TimeoutError("Approved worker clock expired")
        if monotonic()-self.last_check < 1: return
        self.last_check = monotonic()
        self.peak_rss = max(self.peak_rss, owned_rss())
        if self.peak_rss >= self.approval["rss_limit_bytes"]: raise MemoryError("Quoted worker RSS guard")
        if swap_bytes()-self.swap_baseline > 1024**3: raise MemoryError("Swap growth guard")
        if shutil.disk_usage(self.out).free < 8*1024**3: raise OSError("Free disk guard")

    def close(self, status, reason=None):
        atomic_json(self.out / ("worker-budget-"+str(os.getpid())+".json"), {
            "status": status, "reason": reason, "seconds": monotonic()-self.started,
            "peak_owned_rss_bytes": self.peak_rss, "approval": self.approval})

    def native_allocation_budget(self, requested_bytes):
        return native_allocation_budget(self, requested_bytes)
