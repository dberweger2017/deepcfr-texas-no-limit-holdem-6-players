"""Measure locked-policy and projected-profile losses after an admitted solve.

This file runner does not select spots or authorize the main campaign. The
reference equilibrium and gates must already exist. Large jobs still require
the separate resource amendment described by the frozen protocol.
"""

import argparse
import json
from pathlib import Path
import signal
import sys

from src.diagnostics.flop_check import atomic_json, line_key
from src.diagnostics.flop_check_analysis import blueprint_locks, overfold_node, projection
from src.diagnostics.flop_check_equity import EquityBuckets
from src.diagnostics.flop_check_runtime import machine_snapshot, run_tool
from src.diagnostics.saved_hu20 import file_hash
from scripts.preflight_flop_check import prepare_guarded


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def diagnose(binary, request_path, reference_path, profile_path, tables_path, equity_path, out,
             *, job_memory_bytes=6 * 1024**3, initial_swap=None):
    request = json.loads(Path(request_path).read_text()); out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((Path(reference_path).parent / "result.json").read_text())
    if (provenance.get("status") != "completed"
            or provenance["request_sha256"] != file_hash(request_path)
            or provenance["response_sha256"] != file_hash(reference_path)
            or provenance.get("profile_sha256") != file_hash(profile_path)):
        raise ValueError("Equilibrium request/response/profile provenance differs")
    reference_rows = read_rows(reference_path); reference = reference_rows[-1]
    if (reference.get("status") != "solved" or reference["exploitability_pct_pot"] > 0.5
            or not any(r.get("gate") == "V1" and r.get("passed") for r in reference_rows)):
        raise ValueError("Reference solve is not admitted")
    records = read_rows(profile_path)
    templates = {line_key(n["line"]): n["template"]
                 for n in request["nodes"] if not n["terminal"]}
    tables = json.loads(Path(tables_path).read_text())
    if request.get("policy") and request["policy"]["sha256"] != tables["source"]["weights_sha256"]:
        raise ValueError("Blueprint tables are from a different evaluated policy")
    policies = {"e_bp": blueprint_locks(records, request, tables),
                "e_v1proj": projection(records, templates),
                "e_v1proj_line": projection(records, templates, per_line=True)}
    for k in (50, 200):
        buckets = EquityBuckets(equity_path, k)
        policies[f"e_eq{k}"] = projection(records, templates, bucket=buckets)
    features = EquityBuckets(equity_path, 50); flop_equity = features.arrays["flop_equities"]
    folds = []
    for row, bp in zip(records, policies["e_bp"], strict=True):
        if len(row["board"]) != 3 or not any(a["kind"] == "Fold" for a in row["actions"]):
            continue
        fold = overfold_node(row, bp, equity=lambda h: float(flop_equity[features.hands[tuple(sorted(h))]]),
                             opponent_holdings=row["opponent_holdings"],
                             opponent_weights=row["opponent_weights"],
                             template=templates[line_key(row["line"])])
        folds.append(dict(fold, line=row["line"], target_solver_seat=row["player"]))
    request.pop("dump_path", None); request.pop("terminal_queries", None)
    request.update(max_iterations=1, seconds=120)
    results = []
    for metric, profile in policies.items():
        for target in (0, 1):
            label = f"{metric}-target-{target}"
            path = out / (label + ".json")
            atomic_json(path, dict(request, locks=[r for r in profile if r["player"] == target]))
            result = run_tool(binary, path, out / label,
                              memory_bytes=request["memory_budget_bytes"],
                              threads=2, seconds=180, job_memory_bytes=job_memory_bytes,
                              initial_swap=initial_swap)
            if result["status"] != "completed":
                atomic_json(out / "failure.json", {"metric": metric, "target": target,
                                                    "runtime": result, "completed": results})
                raise RuntimeError(result["failure"])
            final = read_rows(out / label / "response.jsonl")[-1]
            responder = 1 - target
            gain = final["mes_ev_chips"][responder] - reference["current_ev_chips"][responder]
            results.append({"metric": metric, "target_solver_seat": target,
                            "target_physical_seat": request["seat_map"][target],
                            "gain_bb": gain / 100, "gain_pct_pot": gain * 100 / request["pot"],
                            "responder_br_chips": final["mes_ev_chips"][responder],
                            "reference_responder_value_chips": reference["current_ev_chips"][responder]})
            atomic_json(out / "partial.json", {"results": results})
    path = out / "both-blueprint.json"; atomic_json(path, dict(request, locks=policies["e_bp"]))
    runtime = run_tool(binary, path, out / "both-blueprint",
                       memory_bytes=request["memory_budget_bytes"], threads=2,
                       seconds=180, job_memory_bytes=job_memory_bytes, initial_swap=initial_swap)
    if runtime["status"] != "completed":
        raise RuntimeError(runtime["failure"])
    locked = read_rows(out / "both-blueprint/response.jsonl")[-1]
    summary = {"spot": request["spot"], "results": results,
               "overfold_nodes": folds,
               "both_blueprint_ev_chips": locked["current_ev_chips"],
               "equilibrium_residual_pct_pot": reference["exploitability_pct_pot"],
               "source": tables["source"], "main_run_admitted": False,
               "note": "Profile diagnostic only; native MC and campaign admission remain separate",
               "input_sha256": {str(p): file_hash(p) for p in
                    (request_path, reference_path, profile_path, tables_path, equity_path)}}
    atomic_json(out / "result.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "request", "reference", "profile", "tables", "equity", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    watch = args.out.with_name(args.out.name + ".watchdog")
    if not args.worker:
        watch.mkdir(parents=True, exist_ok=False)
        before = machine_snapshot(); atomic_json(watch / "machine-before.json", before)
        budget = min(6 * 1024**3, int(before["reclaimable_bytes"] * 0.8 / 1024**3) * 1024**3)
        if budget < 2 * 1024**3:
            raise MemoryError("Insufficient headroom for profile projection")
        atomic_json(watch / "admission.json", {"budget_bytes": budget,
                    "swap_baseline_bytes": before["swap_used_bytes"]})
        prepare_guarded([sys.executable, "-m", "scripts.diagnose_flop_check_profile",
                         *sys.argv[1:], "--worker"], watch, budget, before["swap_used_bytes"])
        return
    def stop(signum, frame):
        # Raising inside run_tool invokes its owned-solver cleanup before exit.
        raise SystemExit("Profile worker stopped by outer resource guard")
    signal.signal(signal.SIGTERM, stop)
    admission = json.loads((watch / "admission.json").read_text())
    diagnose(args.binary, args.request, args.reference, args.profile, args.tables, args.equity, args.out,
             job_memory_bytes=admission["budget_bytes"], initial_swap=admission["swap_baseline_bytes"])


if __name__ == "__main__":
    main()
