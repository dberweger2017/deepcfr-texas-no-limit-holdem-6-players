"""Parallel native-Linux two-phase diagnostic with one immutable quote clock."""

import argparse
import json
import os
from pathlib import Path
from random import Random
import shutil
import signal
import subprocess
import sys
from time import sleep, time

from src.diagnostics.board_pooling import pool_statistics
from src.diagnostics.board_pooling_results import completion, statistics, check_replay, common_mask
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.flop_check_runtime import append, rss_for_tree
from src.diagnostics.pooling_runtime import linux_snapshot, run_linux_tool
from src.diagnostics.saved_hu20 import file_hash


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def worker(a):
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Owned worker stopped")))
    manifest = json.loads(a.prepared.joinpath("manifest.json").read_text())
    job = next(j for j in manifest["jobs"] if j["job"] == a.job)
    budget = json.loads(a.approval.read_text())
    request = json.loads(Path(job["request"]).read_text())
    if file_hash(job["request"]) != job["request_sha256"]:
        raise ValueError("Prepared request hash differs")
    if file_hash(request["compact_path"]) != job["compact_sha256"]:
        raise ValueError("Prepared compact features/policy hash differs")
    destination = a.out / a.phase / job["job"]; destination.mkdir(parents=True, exist_ok=False)
    if a.phase == "relock":
        first = rows(a.out / "collect" / job["job"] / "solver/response.jsonl")
        final = completion(first)
        request.update(pooling_phase="relock", max_iterations=final["iterations"], target_pct_pot=-1,
                       pooled_policy_path=str((a.out / "pooled-policy.json").resolve()))
    path = destination / "request.json"; atomic_json(path, request)
    runtime = run_linux_tool(a.binary, path, destination / "solver",
                             memory_bytes=budget["arena_bytes"], threads=budget["threads_per_worker"],
                             seconds=request["seconds"] + 300,
                             job_memory_bytes=budget["worker_rss_bytes"], initial_swap=budget["swap_baseline_bytes"])
    if runtime["status"] != "completed":
        raise RuntimeError(runtime["failure"])
    actual = rows(destination / "solver/response.jsonl"); final = completion(actual)
    gate = check_replay(first, actual, request["pot"]) if a.phase == "relock" else None
    metrics = [r for r in actual if r["event"] == "pooling_metric"]
    expected = {"e_bp", "e_root_v1"} if a.phase == "collect" else {"e_board_v1", "e_board_eq50"}
    if {(r["metric"], r["target_solver_seat"]) for r in metrics} != {(m, s) for m in expected for s in (0, 1)} or len(metrics) != 4:
        raise ValueError("Missing/duplicate pooled loss measurements")
    atomic_json(destination / "result.json", {"job": job, "eligible": True, "metrics": metrics,
                "completion": final, "runtime": runtime, "replay_gate": gate})


def stop(processes):
    for process, _ in processes:
        if process.poll() is None:
            process.terminate()
    for process, _ in processes:
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()


def parallel(a, jobs, phase, budget):
    pending = list(jobs); active = []; completed = []
    initial = linux_snapshot(); last_log = 0
    try:
        while pending or active:
            snapshot = linux_snapshot(); rss = rss_for_tree(os.getpid())
            if time() + budget["retrieval_shutdown_reserve_seconds"] >= budget["rental_deadline_epoch"]:
                raise RuntimeError("Quote clock reached retrieval/shutdown reserve")
            if rss > budget["aggregate_rss_bytes"]:
                raise RuntimeError("Aggregate owned RSS budget exceeded")
            if snapshot["swap_used_bytes"] - budget["swap_baseline_bytes"] > 1024**3:
                raise RuntimeError("Aggregate swap growth exceeds 1 GiB")
            if snapshot["memory_events"] != initial["memory_events"]:
                raise RuntimeError("Cgroup memory event changed")
            if shutil.disk_usage(a.out).free < budget["minimum_disk_free_bytes"]:
                raise RuntimeError("Disk retrieval reserve violated")
            if time() - last_log >= 5:
                append(a.out / "progress.jsonl", {"event": "resources", "phase": phase, "timestamp": time(),
                    "rss_bytes": rss, "active_workers": len(active), "completed_phase_jobs": len(completed),
                    "pending_phase_jobs": len(pending), "cgroup_used_bytes": snapshot["cgroup_used_bytes"],
                    "swap_used_bytes": snapshot["swap_used_bytes"]})
                atomic_json(a.out / "status.json", {"phase": phase, "done": len(completed),
                            "total": len(jobs), "active": [j["job"] for _, j in active], "timestamp": time(),
                            "rental_deadline_epoch": budget["rental_deadline_epoch"]})
                last_log = time()
            for process, job in list(active):
                if process.poll() is None:
                    continue
                active.remove((process, job))
                if process.returncode:
                    raise RuntimeError(f'Worker {job["job"]} failed ({process.returncode})')
                result = json.loads((a.out / phase / job["job"] / "result.json").read_text())
                if not result["eligible"]:
                    raise RuntimeError("Worker failed an eligibility gate")
                completed.append(job)
                append(a.out / "progress.jsonl", {"event": "job_complete", "phase": phase, "job": job["job"],
                                                "completed_phase_jobs": len(completed), "timestamp": time()})
            while pending and len(active) < budget["workers"]:
                job = pending.pop(0)
                command = [sys.executable, "-m", "scripts.run_board_pooling", "--worker", "--phase", phase,
                           "--job", job["job"], "--plan", str(a.plan), "--prepared", str(a.prepared),
                           "--out", str(a.out), "--binary", str(a.binary), "--approval", str(a.approval)]
                with (a.out / (phase + "-" + job["job"] + ".log")).open("x") as output:
                    process = subprocess.Popen(command, stdout=output, stderr=output, start_new_session=True)
                active.append((process, job))
            sleep(.5)
    finally:
        stop(active)


def campaign(a):
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Owned campaign stopped")))
    budget = json.loads(a.approval.read_text())
    if not budget.get("owner_approved_quote") or not budget.get("qualification_passed"):
        raise ValueError("Owner quote approval and Linux parity/real-export gates are required")
    if file_hash(a.binary) != budget["binary_sha256"] or file_hash(a.plan) != budget["plan_sha256"]:
        raise ValueError("Approved binary/protocol fingerprint differs")
    if file_hash(budget["qualification_path"]) != budget["qualification_sha256"]:
        raise ValueError("Qualification evidence hash differs")
    qualification = json.loads(Path(budget["qualification_path"]).read_text())
    if not qualification["passed"] or not all(r["passed"] for r in qualification["gates"]):
        raise ValueError("Qualification gates did not pass")
    pilots = [r["runtime"]["elapsed_seconds"] for r in qualification["gates"] if r["gate"] == "real-pilot-V5"]
    if len(pilots) != 3:
        raise ValueError("Three fixed resource/convergence pilots are required")
    snapshot = linux_snapshot()
    if (budget["workers"] * budget["threads_per_worker"] > snapshot["effective_cores"]
            or budget["aggregate_rss_bytes"] > .8 * snapshot["available_bytes"]
            or budget["workers"] * budget["worker_rss_bytes"] > budget["aggregate_rss_bytes"]):
        raise ValueError("Quote worker shape exceeds measured free memory/CPU")
    a.out.mkdir(parents=True, exist_ok=False)
    atomic_json(a.out / "admission.json", {"budget": budget, "machine": snapshot})
    plan = json.loads(a.plan.read_text()); manifest = json.loads((a.prepared / "manifest.json").read_text())
    if manifest["plan_sha256"] != file_hash(a.plan):
        raise ValueError("Prepared plan differs")
    forecast = 1.5 * max(pilots) * 2 * plan["jobs_total"] / budget["workers"]
    remaining = budget["rental_deadline_epoch"] - time() - budget["retrieval_shutdown_reserve_seconds"]
    atomic_json(a.out / "forecast-admission.json", {"conservative_main_seconds": forecast,
                "remaining_main_seconds": remaining, "pilot_seconds": pilots, "passed": forecast <= remaining})
    if forecast > remaining:
        atomic_json(a.out / "failure.json", {"error": "Pilot forecast exceeds approved clock", "timestamp": time(), "automatic_restart": False})
        raise ValueError("Pilot forecast exceeds approved clock; no production start")
    jobs = sorted(manifest["jobs"], key=lambda j: (j["spot"], j["policy_index"]))
    if len(jobs) + len(manifest["support_exclusions"]) != plan["jobs_total"] or len({j["job"] for j in jobs}) != len(jobs):
        raise ValueError("Frozen job schedule has missing or duplicate entries")
    Random(plan["schedule_seed"]).shuffle(jobs)
    # Rotate all three lineages across a fixed board order, keeping partials balanced.
    roots = sorted({j["spot"] for j in jobs}); Random(plan["schedule_seed"]).shuffle(roots)
    jobs = [j for round_no in range(3) for index, spot in enumerate(roots) for j in jobs
            if j["spot"] == spot and j["policy_index"] == (index + round_no) % 3]
    atomic_json(a.out / "schedule.json", jobs)
    try:
        parallel(a, jobs, "collect", budget)
        results = {j["job"]: json.loads((a.out / "collect" / j["job"] / "result.json").read_text()) for j in jobs}
        mask = common_mask(manifest, results, plan["policies"])
        atomic_json(a.out / "common-mask.json", mask)
        eligible = [j for j in jobs if j["spot"] in mask["admitted"]]
        records = [dict(j, groups=statistics(rows(a.out / "collect" / j["job"] / "solver/response.jsonl"))) for j in eligible]
        atomic_json(a.out / "pooled-policy.json", pool_statistics(records))
        parallel(a, eligible, "relock", budget)
        atomic_json(a.out / "completion.json", {"status": "completed", "phase1_jobs": len(jobs),
                    "phase2_jobs": len(eligible), "common_boards": len(mask["admitted"]),
                    "support_exclusions": manifest["support_exclusions"], "timestamp": time()})
    except BaseException as error:
        atomic_json(a.out / "failure.json", {"error": str(error), "timestamp": time(),
                    "restart": "requires owner review; no automatic restart"})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "prepared", "out", "binary", "approval"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--phase", choices=("collect", "relock"))
    p.add_argument("--job")
    a = p.parse_args()
    if a.worker:
        worker(a)
    else:
        campaign(a)


if __name__ == "__main__":
    main()
