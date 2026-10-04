"""Singleton compact-key and shared-policy oracle on #145's tiny turn fixture."""

import argparse
import json
from pathlib import Path

from src.diagnostics.board_pooling import pool_statistics
from src.diagnostics.board_pooling_features import add_pool_keys
from src.diagnostics.board_pooling_results import (completion, statistics, check_replay,
                                                 check_lock_only, check_locked_br_parity)
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.pooling_runtime import run_portable_tool
from src.diagnostics.saved_hu20 import file_hash


def validate(binary, reference_binary, fixture, out, *, threads=2):
    out.mkdir(parents=True, exist_ok=False)
    request = json.loads((fixture / "request.json").read_text())
    data = json.loads((fixture / "export/compact.json").read_text())
    data["crossfit_labels"] = {str(f): data["labels"]["50"] for f in (0, 1)}
    data["pool_keys"] = add_pool_keys(request, data, data)
    atomic_json(out / "compact.json", data)
    request.pop("dump_path", None)
    request["compact_path"] = str((out / "compact.json").resolve())
    request["policy"] = {"seed": data["source"]["training_seed"]}
    gate_rows = {}
    for name, tool, extra in (("reference", reference_binary, {}),
                              ("collect", binary, {"pooling_phase": "collect"})):
        path = out / (name + ".json"); atomic_json(path, dict(request, **extra))
        runtime = run_portable_tool(tool, path, out / name, memory_bytes=1024**3,
                                    job_memory_bytes=1024**3, threads=threads, seconds=240)
        if runtime["status"] != "completed":
            raise RuntimeError(runtime["failure"])
        gate_rows[name] = [json.loads(line) for line in (out / name / "response.jsonl").read_text().splitlines()]
    collected = gate_rows["collect"]; final = completion(collected)
    record = {"lineage": request["policy"]["seed"], "spot": "singleton", "board_weight": 1,
              "groups": statistics(collected)}
    atomic_json(out / "pool.json", pool_statistics([record]))
    relock = dict(request, pooling_phase="relock", max_iterations=final["iterations"], target_pct_pot=-1,
                  pooled_policy_path=str((out / "pool.json").resolve()))
    relock["pooling_measurements"] = [{"metric": name, "projection_metric": projection,
        "policy_path": str((out / "pool.json").resolve())}
        for name, projection in (("e_board_v1", "v1"), ("e_board_eq50", "eq50"),
                                  ("e_cross_v1", "v1"), ("e_cross_eq50", "eq50-fit0"))]
    path = out / "relock.json"; atomic_json(path, relock)
    runtime = run_portable_tool(binary, path, out / "relock", memory_bytes=1024**3,
                                job_memory_bytes=1024**3, threads=threads, seconds=240)
    if runtime["status"] != "completed":
        raise RuntimeError(runtime["failure"])
    replay = [json.loads(line) for line in (out / "relock/response.jsonl").read_text().splitlines()]
    gates = [check_replay(collected, replay, request["pot"])]
    expected = {(r["metric"], r["target_solver_seat"]): r["gain_bb"] for r in gate_rows["reference"]
                if r["event"] == "compact_metric"}
    values = {(r["metric"], r["target_solver_seat"]): r["gain_bb"] for r in collected + replay
              if r["event"] == "pooling_metric"}
    pairs = (("e_bp", "e_bp"), ("e_root_v1", "e_v1proj"),
             ("e_board_v1", "e_v1proj"), ("e_board_eq50", "e_eq50"))
    for actual, original in pairs:
        for seat in (0, 1):
            error = abs(values[actual, seat] - expected[original, seat])
            gates.append({"gate": "singleton-" + actual, "target_solver_seat": seat,
                          "error_bb": error, "passed": error <= 1e-5 * request["pot"] / 100})
    fresh = dict(relock, pooling_phase="lock-only", max_iterations=0,
                 reference_equilibrium_ev_chips=final["current_ev_chips"],
                 reference_response_sha256=file_hash(out / "collect/response.jsonl"))
    path = out / "lock-only.json"; atomic_json(path, fresh)
    runtime = run_portable_tool(binary, path, out / "lock-only", memory_bytes=1024**3,
                                job_memory_bytes=1024**3, threads=threads, seconds=240)
    if runtime["status"] != "completed":
        raise RuntimeError(runtime["failure"])
    fresh_rows = [json.loads(line) for line in (out / "lock-only/response.jsonl").read_text().splitlines()]
    gates += [check_lock_only(collected, fresh_rows, request["pot"], fresh["reference_response_sha256"]),
              check_locked_br_parity(fresh_rows, replay, request["pot"])]
    missing = pool_statistics([record]); missing["groups"] = [r for r in missing["groups"] if r["metric"] != "v1"]
    atomic_json(out / "missing-pool.json", missing)
    measures = [{"metric": "e_cross_v1", "projection_metric": "v1", "allow_missing": True,
                 "policy_path": str((out / "missing-pool.json").resolve())}]
    measured = {}
    for name, config in (("missing-fresh", dict(fresh, pooling_measurements=measures)),
                          ("missing-solved", dict(relock, pooling_measurements=measures))):
        path = out / (name + ".json"); atomic_json(path, config)
        runtime = run_portable_tool(binary, path, out / name, memory_bytes=1024**3,
                                    job_memory_bytes=1024**3, threads=threads, seconds=240)
        if runtime["status"] != "completed":
            raise RuntimeError(runtime["failure"])
        measured[name] = [json.loads(line) for line in (out / name / "response.jsonl").read_text().splitlines()]
    gates.append(check_locked_br_parity(measured["missing-fresh"], measured["missing-solved"], request["pot"]))
    metrics = [r for r in measured["missing-fresh"] if r["event"] == "pooling_metric"]
    fallback = all(sum(c[1] for c in r["fallback_coverage"].values()) > 0 for r in metrics)
    gates.append({"gate": "held-out-missing-key-uniform-fallback", "passed": fallback})
    result = {"passed": all(g["passed"] for g in gates), "gates": gates,
              "fixture_request_sha256": file_hash(fixture / "request.json"),
              "fixture_compact_sha256": file_hash(fixture / "export/compact.json"),
              "new_binary_sha256": file_hash(binary), "reference_binary_sha256": file_hash(reference_binary),
              "scope": "four-holding tiny-SPR #145 fixture; not main outcomes"}
    atomic_json(out / "gates.json", result)
    if not result["passed"]:
        raise ValueError("Shared-policy singleton oracle failed")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "reference-binary", "fixture", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = validate(a.binary, a.reference_binary, a.fixture, a.out)
    print(json.dumps({"passed": result["passed"], "checks": len(result["gates"])}))


if __name__ == "__main__":
    main()
