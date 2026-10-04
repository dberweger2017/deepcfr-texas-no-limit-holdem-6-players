"""Guarded M4 two-phase diagnostic with one cumulative experiment clock."""

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

from src.diagnostics.board_pooling import pool_statistics, covered_context_policy
from src.diagnostics.board_pooling_results import (completion, statistics, check_replay, common_mask,
                                                  check_lock_only, check_locked_br_parity)
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.flop_check_runtime import append, rss_for_tree
from src.diagnostics.pooling_runtime import resource_snapshot, run_owned_tool, admit_m4
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
        reference_path = a.out / "collect" / job["job"] / "solver/response.jsonl"
        first = rows(reference_path)
        final = completion(first)
        fit_fold = 1 - job["evaluation_fold"]
        shared = str((a.out / "pooled-policy.json").resolve())
        heldout = str((a.out / f'crossfit-{job["evaluation_fold"]}.json').resolve())
        request.update(pooling_phase="lock-only", max_iterations=0,
                       reference_equilibrium_ev_chips=final["current_ev_chips"],
                       reference_response_sha256=file_hash(reference_path),
                       pooling_measurements=[
                           {"metric": "e_cross_v1", "projection_metric": "v1", "policy_path": heldout, "allow_missing": True},
                           {"metric": "e_cross_eq50", "projection_metric": f"eq50-fit{fit_fold}", "policy_path": heldout, "allow_missing": True},
                           {"metric": "e_board_v1", "projection_metric": "v1", "policy_path": shared},
                           {"metric": "e_board_eq50", "projection_metric": "eq50", "policy_path": shared}])
        covered_path = destination / "covered-policy.json"
        local = pool_statistics([dict(job, groups=statistics(first))])
        atomic_json(covered_path, covered_context_policy(json.loads(Path(heldout).read_text()), local))
        request["pooling_measurements"].append({"metric": "e_cross_v1_covered", "projection_metric": "v1",
            "policy_path": str(covered_path.resolve()), "allow_missing": True})
        inventory = json.loads((a.out / "pool-inventory.json").read_text())
        for policy_path in (shared, heldout):
            if file_hash(policy_path) != inventory[Path(policy_path).name]:
                raise ValueError("Pooled policy hash differs")
    path = destination / "request.json"; atomic_json(path, request)
    runtime = run_owned_tool(a.binary, path, destination / "solver",
                             memory_bytes=budget["arena_bytes"], threads=budget["threads_per_worker"],
                             seconds=request["seconds"] + 300,
                             job_memory_bytes=budget["worker_rss_bytes"], initial_swap=budget["swap_baseline_bytes"])
    if runtime["status"] != "completed":
        raise RuntimeError(runtime["failure"])
    actual = rows(destination / "solver/response.jsonl"); gates = []
    if a.phase == "relock":
        final = actual[-1]
        gates.append(check_lock_only(first, actual, request["pot"], request["reference_response_sha256"]))
        if job["replay_sample"]:
            replay_request = dict(request, pooling_phase="relock", max_iterations=completion(first)["iterations"], target_pct_pot=-1)
            replay_path = destination / "replay-request.json"; atomic_json(replay_path, replay_request)
            replay_runtime = run_owned_tool(a.binary, replay_path, destination / "replay-solver",
                memory_bytes=budget["arena_bytes"], threads=budget["threads_per_worker"],
                seconds=request["seconds"] + 300, job_memory_bytes=budget["worker_rss_bytes"], initial_swap=budget["swap_baseline_bytes"])
            if replay_runtime["status"] != "completed":
                raise RuntimeError(replay_runtime["failure"])
            replay_rows = rows(destination / "replay-solver/response.jsonl")
            gates += [check_replay(first, replay_rows, request["pot"]), check_locked_br_parity(actual, replay_rows, request["pot"])]
    else:
        final = completion(actual)
    metrics = [r for r in actual if r["event"] == "pooling_metric"]
    expected = {"e_bp", "e_root_v1"} if a.phase == "collect" else {"e_cross_v1", "e_cross_eq50", "e_board_v1", "e_board_eq50", "e_cross_v1_covered"}
    if {(r["metric"], r["target_solver_seat"]) for r in metrics} != {(m, s) for m in expected for s in (0, 1)} or len(metrics) != 2*len(expected):
        raise ValueError("Missing/duplicate pooled loss measurements")
    atomic_json(destination / "result.json", {"job": job, "eligible": True, "metrics": metrics,
                "completion": final, "runtime": runtime, "gates": gates,
                "replay_gate": {"passed": all(g["passed"] for g in gates), "sampled": job.get("replay_sample", False)}})


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
    initial = resource_snapshot(); last_log = 0
    try:
        while pending or active:
            snapshot = resource_snapshot(); rss = rss_for_tree(os.getpid())
            if time() + budget["retrieval_shutdown_reserve_seconds"] >= budget["experiment_deadline_epoch"]:
                raise RuntimeError("Quote clock reached retrieval/shutdown reserve")
            if rss > budget["aggregate_rss_bytes"]:
                raise RuntimeError("Aggregate owned RSS budget exceeded")
            if snapshot["swap_used_bytes"] - budget["swap_baseline_bytes"] > 1024**3:
                raise RuntimeError("Aggregate swap growth exceeds 1 GiB")
            if shutil.disk_usage(a.out).free < budget["minimum_disk_free_bytes"]:
                raise RuntimeError("Disk retrieval reserve violated")
            if time() - last_log >= 5:
                append(a.out / "progress.jsonl", {"event": "resources", "phase": phase, "timestamp": time(),
                    "rss_bytes": rss, "active_workers": len(active), "completed_phase_jobs": len(completed),
                    "pending_phase_jobs": len(pending), "memory_pressure": snapshot["memory_pressure"],
                    "swap_used_bytes": snapshot["swap_used_bytes"]})
                atomic_json(a.out / "status.json", {"phase": phase, "done": len(completed),
                            "total": len(jobs), "active": [j["job"] for _, j in active], "timestamp": time(),
                            "experiment_deadline_epoch": budget["experiment_deadline_epoch"]})
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


def write_pooled_policies(jobs, response_path, out, folds):
    """Fit one lineage at a time with the original per-group summation order."""
    if set(folds.values()) != {0, 1} or any(j["spot"] not in folds for j in jobs):
        raise ValueError("Frozen split lacks an eligible root or half")

    def write(path, selected, metadata):
        metadata = dict(metadata, format="hu20-board-pooling-policy-v1", groups=None,
                        root_lineage_records=len(selected), zero_mass_rule="uniform within the actual menu")
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as stream:
            stream.write("{")
            for field_index, name in enumerate(sorted(metadata)):
                if field_index:
                    stream.write(", ")
                stream.write(json.dumps(name) + ": ")
                if name != "groups":
                    stream.write(json.dumps(metadata[name], sort_keys=True))
                    continue
                stream.write("["); count = 0
                for lineage in sorted({j["lineage"] for j in selected}):
                    records = (dict(j, groups=statistics(rows(response_path(j))))
                               for j in selected if j["lineage"] == lineage)
                    policy = pool_statistics(records)
                    for group in policy["groups"]:
                        if count:
                            stream.write(", ")
                        stream.write(json.dumps(group, sort_keys=True)); count += 1
                    del policy
                stream.write("]")
            stream.write("}\n")
        temporary.replace(path)

    write(out / "pooled-policy.json", jobs, {})
    for evaluation_fold in (0, 1):
        training = [j for j in jobs if folds[j["spot"]] != evaluation_fold]
        if not training:
            raise ValueError("No eligible training roots in a frozen half")
        write(out / f"crossfit-{evaluation_fold}.json", training,
              {"evaluation_fold": evaluation_fold, "training_fold": 1-evaluation_fold,
               "training_spots": sorted({j["spot"] for j in training})})


def fit_pools(a, budget):
    """A sequential owned child releases all fitting allocations before solvers resume."""
    command = ["nice", "-n", "10", sys.executable, "-m", "scripts.fit_board_pooling",
               "--plan", str(a.plan), "--run", str(a.out), "--approval", str(a.approval)]
    atomic_json(a.out / "status.json", {"phase": "pool-fit", "done": 0, "total": 3,
                "timestamp": time(), "experiment_deadline_epoch": budget["experiment_deadline_epoch"]})
    with (a.out / "pool-fit.log").open("x") as output:
        process = subprocess.Popen(command, stdout=output, stderr=output, start_new_session=True)
        try:
            while process.poll() is None:
                if time() + budget["retrieval_shutdown_reserve_seconds"] >= budget["experiment_deadline_epoch"]:
                    raise RuntimeError("Pool fit reached retrieval reserve")
                if rss_for_tree(process.pid) > budget["worker_rss_bytes"]:
                    raise RuntimeError("Pool fit worker RSS exceeded")
                sleep(.5)
            if process.returncode:
                raise RuntimeError(f"Pool fit failed ({process.returncode})")
        finally:
            stop([(process, {})])


def campaign(a):
    if json.loads(a.plan.read_text())["format"] != "hu20-board-pooling-plan-v3":
        raise ValueError("Only the held-out revision-3 protocol is admitted")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Owned campaign stopped")))
    budget = json.loads(a.approval.read_text())
    if not budget.get("149_owner_authorized") or not budget.get("owner_resumed") or not budget.get("qualification_passed"):
        raise ValueError("Owner resume, M4 authorization and real-export gates are required")
    if file_hash(a.binary) != budget["binary_sha256"] or file_hash(a.plan) != budget["plan_sha256"]:
        raise ValueError("Approved binary/protocol fingerprint differs")
    if file_hash(budget["qualification_path"]) != budget["qualification_sha256"]:
        raise ValueError("Qualification evidence hash differs")
    qualification = json.loads(Path(budget["qualification_path"]).read_text())
    if not qualification["passed"] or not all(r["passed"] for r in qualification["gates"]):
        raise ValueError("Qualification gates did not pass")
    if qualification.get("worker_rss_bytes") != budget["worker_rss_bytes"]:
        raise ValueError("Worker RSS ceiling differs from qualification")
    if qualification.get("threads_per_worker") != budget["threads_per_worker"]:
        raise ValueError("Replay thread count differs from fixed pilots")
    if (qualification["binary_sha256"] != budget["binary_sha256"]
            or qualification["plan_sha256"] != budget["plan_sha256"]):
        raise ValueError("Qualification belongs to another binary or protocol")
    pilots = [r["runtime"]["elapsed_seconds"] for r in qualification["gates"] if r["gate"] == "real-pilot-V5"]
    if len(pilots) != 3:
        raise ValueError("Three fixed resource/convergence pilots are required")
    snapshot = resource_snapshot()
    admit_m4(budget, snapshot)
    a.out.mkdir(parents=True, exist_ok=False)
    atomic_json(a.out / "admission.json", {"budget": budget, "machine": snapshot})
    plan = json.loads(a.plan.read_text()); manifest = json.loads((a.prepared / "manifest.json").read_text())
    if manifest["plan_sha256"] != file_hash(a.plan):
        raise ValueError("Prepared plan differs")
    locked_pilots = qualification["lock_only_pilot_seconds"]
    if len(locked_pilots) != 3:
        raise ValueError("Three locked-evaluation timings are required")
    forecast = 1.5 * (max(pilots)*(plan["jobs_total"]+plan["replay_jobs"])
                       + max(locked_pilots)*(plan["jobs_total"]+plan["replay_jobs"])) / budget["workers"]
    remaining = budget["experiment_deadline_epoch"] - time() - budget["retrieval_shutdown_reserve_seconds"]
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
        fit_pools(a, budget)
        names = ["pooled-policy.json", "crossfit-0.json", "crossfit-1.json"]
        atomic_json(a.out / "pool-inventory.json", {name: file_hash(a.out/name) for name in names})
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
