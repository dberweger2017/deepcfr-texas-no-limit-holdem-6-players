"""Admit and measure the three-spot M4 diagnostic without exceeding its budget."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic, sleep

from src.diagnostics.flop_check import atomic_json
from src.diagnostics.flop_check_runtime import append, machine_snapshot, rss_for_tree, run_tool, swap_usage
from src.diagnostics.saved_hu20 import file_hash

GIB = 1024**3


def heartbeat(out, stage, done, total, error=None):
    path = Path(out) / "status.md"; temporary = path.with_suffix(".tmp")
    temporary.write_text(f"Stage: {stage}\n\nSpots: {done}/{total}\n\n"
                         f"ETA: unavailable until admitted solve timings exist\n\n"
                         f"Last error: {error or 'none'}\n")
    temporary.replace(path)


def prepare_guarded(command, out, budget, baseline):
    """Native replay/export can consume memory before the solver is launched."""
    started = monotonic(); failure = None; peak = 0
    with (out / "preparation.log").open("w") as log:
        worker = subprocess.Popen(["nice", "-n", "10", *command], stdout=log,
                                  stderr=log, start_new_session=True)
        try:
            while True:
                rss = rss_for_tree(os.getpid()); peak = max(peak, rss)
                swap = swap_usage()
                append(out / "progress.jsonl", {"event": "resources", "rss_bytes": rss,
                       "swap_used_bytes": swap, "stage": "preparation"})
                if rss > budget:
                    failure = "Preparation RSS exceeds budget"
                elif swap - baseline > GIB:
                    failure = "Preparation swap growth exceeds 1 GiB"
                elif monotonic() - started > 1200:
                    failure = "Preparation wall deadline"
                if failure:
                    worker.send_signal(signal.SIGTERM)
                    try:
                        worker.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(worker.pid, signal.SIGKILL); worker.wait()
                    break
                if worker.poll() is not None:
                    break
                sleep(2)
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM); worker.wait()
    result = {"passed": not failure and worker.returncode == 0,
              "failure": failure, "exit_code": worker.returncode,
              "peak_job_rss_bytes": peak, "elapsed_seconds": monotonic() - started}
    atomic_json(out / "preparation-result.json", result)
    if not result["passed"]:
        raise RuntimeError(result)


def preflight(binary, plan, inputs, out, prepared_root=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    before = machine_snapshot(); atomic_json(out / "machine-before.json", before)
    # Reclaimable means free + inactive + speculative, not all RAM. Leave at
    # least 20% of that amount and never claim a 10 GiB allowance by default.
    budget = min(6 * GIB, math.floor(before["reclaimable_bytes"] * 0.8 / GIB) * GIB)
    if budget < 2 * GIB:
        raise MemoryError("Insufficient measured headroom for policy export")
    baseline = before["swap_used_bytes"]; threads = 2
    admission = {"budget_bytes": budget, "swap_baseline_bytes": baseline,
                 "measured_reclaimable_bytes": before["reclaimable_bytes"],
                 "rayon_threads": threads, "nice": 10, "solver_processes": 1,
                 "budget_rule": "min(6 GiB, floor(80% reclaimable GiB))"}
    atomic_json(out / "admission.json", admission)
    records = []
    for cap in (None, 3):
        label = "native" if cap is None else "cap3"
        heartbeat(out, f"{label} tree/range preparation", len(records), 6)
        prepared = Path(prepared_root or out) / f"requests-{label}"
        command = [sys.executable, "-m", "scripts.prepare_flop_check", "--plan", str(plan),
                   "--inputs", str(inputs), "--out", str(prepared), "--memory-gib", str(budget / GIB)]
        if cap is not None:
            command += ["--raise-cap", str(cap)]
        if prepared_root is None:
            prepare_guarded(command, out, budget, baseline)
        manifest = json.loads((prepared / "manifest.json").read_text())
        for item in manifest["records"]:
            if item["status"] != "prepared":
                records.append(item); continue
            heartbeat(out, f'{label} {item["kind"]} memory probe', len(records), 6)
            path = prepared / item["request"]
            if file_hash(path) != item["request_sha256"]:
                raise ValueError("Retained request hash differs")
            if manifest["memory_budget_bytes"] != budget:
                request = json.loads(path.read_text())
                request["memory_budget_bytes"] = budget
                path = out / f'{label}-{item["kind"]}-request.json'
                atomic_json(path, request); del request
            run = out / f'{label}-{item["kind"]}'
            result = run_tool(binary, path, run, memory_bytes=budget, threads=threads,
                              seconds=600, initial_swap=baseline)
            for line in (run / "progress.jsonl").read_text().splitlines():
                row = json.loads(line); row.update(stage="preflight", tree=label, kind=item["kind"])
                append(out / "progress.jsonl", row)
            if result["status"] != "completed":
                heartbeat(out, "stopped", len(records), 6, result["failure"])
                atomic_json(out / "failure.json", {"attempt": result, "records": records})
                raise RuntimeError(result["failure"])
            rows = [json.loads(s) for s in (run / "response.jsonl").read_text().splitlines()]
            memory = next((r for r in rows if r["event"] == "memory"), None)
            structural = next((r for r in rows if r["event"] == "structural_memory"), None)
            record = dict(item, tree=label, memory=memory, structural=structural,
                          completion=rows[-1], runtime=result,
                          solver_request_sha256=file_hash(path),
                          solve_seconds_to_target=None, removed_reach_audit="not solved")
            if memory and memory["compressed_bytes"] <= budget:
                # A capped equilibrium without a removed-reach audit is not
                # admitted. Its memory estimate can still be reported.
                if cap is None:
                    request = json.loads(path.read_text())
                    request.update(mode="solve", max_iterations=10000, progress_every=25,
                                   target_pct_pot=0.2, seconds=1200)
                    solve_path = run / "solve-request.json"; atomic_json(solve_path, request)
                    runtime = run_tool(binary, solve_path, run / "solve", memory_bytes=budget,
                                       threads=threads, seconds=1260, initial_swap=baseline)
                    if runtime["status"] != "completed":
                        raise RuntimeError(runtime["failure"])
                    final = json.loads((run / "solve/response.jsonl").read_text().splitlines()[-1])
                    record["solve"] = final
                    if final["exploitability_pct_pot"] <= 0.2:
                        record["solve_seconds_to_target"] = final["elapsed_seconds"]
            records.append(record)
            atomic_json(out / "preflight.json", {"admission": admission, "records": records})
    admitted = [r for r in records if r.get("solve_seconds_to_target") is not None]
    summary = {"admission": admission, "records": records,
               "main_run_admitted": len(admitted) == 3,
               "projected_main_hours": None,
               "status": "preflight-only; protocol pending" if len(admitted) == 3
                         else "resource-blocked; no main run",
               "after": machine_snapshot()}
    atomic_json(out / "result.json", summary)
    heartbeat(out, summary["status"], len(records), 6)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "plan", "inputs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    args = parser.parse_args()
    preflight(args.binary, args.plan, args.inputs, args.out, args.prepared_root)


if __name__ == "__main__":
    main()
