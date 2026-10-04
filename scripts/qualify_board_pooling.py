"""Run retained macOS fixtures and real-export V4 before M4 production."""

import argparse
import json
from pathlib import Path
import signal
import numpy as np

from scripts.validate_board_pooling import validate as singleton
from scripts.validate_flop_check import monte_carlo
from src.diagnostics.board_pooling_policy import DiskAverage
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.pooling_runtime import run_portable_tool
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.turn_check import replay_root
from src.diagnostics.board_pooling import pool_statistics
from src.diagnostics.board_pooling_results import statistics, check_lock_only, check_locked_br_parity


def parity(binary, fixtures, out):
    out.mkdir(parents=True, exist_ok=False); checks = []
    for path in sorted(fixtures.glob("*.json")):
        request = json.loads(path.read_text())
        reference = fixtures / path.stem / "response.jsonl"
        if "solver_commit" not in request or not reference.exists():
            continue
        if request.get("compact_path") or request.get("dump_path"):
            raise ValueError("Recorded river parity fixtures must have no host-specific external paths")
        runtime = run_portable_tool(binary, path, out / path.stem,
            memory_bytes=request["memory_budget_bytes"], threads=6, seconds=300, job_memory_bytes=6 * 1024**3)
        if runtime["status"] != "completed":
            raise RuntimeError(runtime["failure"])
        actual = [json.loads(line) for line in (out / path.stem / "response.jsonl").read_text().splitlines()]
        expected = [json.loads(line) for line in reference.read_text().splitlines()]
        a, b = actual[-1], expected[-1]
        v1 = next((r for r in actual if r.get("gate") == "V1"), None)
        if not v1 or not v1["passed"]:
            raise ValueError("Native tree parity gate failed")
        checks.append({"gate": "V1", "fixture": path.name, "passed": True, "nodes": v1["nodes"]})
        if a["status"] != b["status"] or a["iterations"] != b["iterations"]:
            raise ValueError("Recorded fixture convergence recipe differs")
        for field in ("current_ev_chips", "mes_ev_chips"):
            error = float(np.max(np.abs(np.asarray(a[field]) - b[field])))
            checks.append({"gate": "mac-mac-" + field, "fixture": path.name,
                           "maximum_chip_error": error, "passed": error <= 1e-5 * request["pot"]})
        checks.append({"gate": "mac-mac-residual", "fixture": path.name, "passed":
                       abs(a["exploitability_pct_pot"] - b["exploitability_pct_pot"]) <= .001})
        queries = [r for r in actual if r["event"] == "payoff_query"]
        original = [r for r in expected if r["event"] == "payoff_query"]
        if len(queries) != len(original):
            raise ValueError("Recorded terminal samples differ")
        if queries:
            errors = [abs(x - y) for a, b in zip(queries, original, strict=True)
                      for x, y in zip(a["payoff_chips"], b["payoff_chips"], strict=True)]
            checks.append({"gate": "mac-mac-V2", "fixture": path.name,
                           "samples": len(queries), "maximum_chip_error": max(errors),
                           "passed": max(errors) < .01})
        if not all(r["passed"] for r in checks):
            atomic_json(out / "failure.json", {"gates": checks}); raise ValueError("Platform parity failed")
    if not checks:
        raise ValueError("No retained macOS river fixtures were qualified")
    atomic_json(out / "gates.json", {"passed": True, "gates": checks})
    return checks


def real_v4(binary, plan, prepared, out):
    config = json.loads(plan.read_text()); corpus = json.loads(Path(config["corpus"]["path"]).read_text())
    manifest = json.loads((prepared / "manifest.json").read_text())
    selected = [corpus["roots"][i] for i in (0, len(corpus["roots"]) // 2, len(corpus["roots"]) - 1)]
    spec = config["policies"][0]
    index_inventory = json.loads((prepared / "policy-0-index.json").read_text())
    if file_hash(prepared / "policy-0.sqlite") != index_inventory["index_sha256"]:
        raise ValueError("Immutable pilot index hash differs")
    source = DiskAverage(prepared / "policy-0.sqlite", spec)
    out.mkdir(parents=True, exist_ok=False); gates = []; locked_seconds = []
    for index, root in enumerate(selected):
        job = next(j for j in manifest["jobs"] if j["spot"] == root["spot"] and j["policy_index"] == 0)
        request = json.loads(Path(job["request"]).read_text()); request.pop("pooling_phase")
        if file_hash(job["request"]) != job["request_sha256"] or file_hash(request["compact_path"]) != job["compact_sha256"]:
            raise ValueError("Pilot request/features hash differs")
        request.update(max_iterations=1, progress_every=1, compact_kind="bp-ev")
        path = out / f"pilot-{index}.json"; atomic_json(path, request)
        runtime = run_portable_tool(binary, path, out / f"pilot-{index}",
            memory_bytes=request["memory_budget_bytes"], threads=6, seconds=600, job_memory_bytes=5 * 1024**3)
        if runtime["status"] != "completed":
            raise RuntimeError(runtime["failure"])
        response = [json.loads(line) for line in (out / f"pilot-{index}/response.jsonl").read_text().splitlines()]
        ranges = {s: tuple((tuple(r["hand"]), r["weight"]) for r in request["ranges"][p])
                  for p, s in enumerate(request["seat_map"])}
        mc = monte_carlo(replay_root(root), ranges, source, seed=202610030310 + index)
        ev = next(r for r in response if r["event"] == "both_blueprint_ev")["current_ev_chips"][request["seat_map"].index(0)] / 100
        gate = {"gate": "real-export-V4", "spot": root["spot"], "source_sha256": spec["sha256"],
                "passed": mc["ci95"][0] <= ev <= mc["ci95"][1], "solver_ev_bb": ev, "native_mc": mc}
        gates.append(gate); atomic_json(out / "gates.json", {"passed": all(g["passed"] for g in gates), "gates": gates})
        if not gate["passed"]:
            raise ValueError("Real-export V4 failed")
        # A fixed pilot qualifies solve time and native memory before admitting
        # the full quoted campaign. Its losses remain prospective pilot evidence.
        equilibrium = json.loads(Path(job["request"]).read_text())
        path = out / f"pilot-{index}-equilibrium.json"; atomic_json(path, equilibrium)
        runtime = run_portable_tool(binary, path, out / f"pilot-{index}-equilibrium",
            memory_bytes=equilibrium["memory_budget_bytes"], threads=6,
            seconds=equilibrium["seconds"] + 300, job_memory_bytes=5 * 1024**3)
        if runtime["status"] != "completed":
            raise RuntimeError(runtime["failure"])
        response = [json.loads(line) for line in (out / f"pilot-{index}-equilibrium/response.jsonl").read_text().splitlines()]
        final = response[-1]
        gates.append({"gate": "real-pilot-V5", "spot": root["spot"],
                      "passed": final.get("status") == "solved" and final["exploitability_pct_pot"] <= .2,
                      "completion": final, "runtime": runtime})
        atomic_json(out / "gates.json", {"passed": all(g["passed"] for g in gates), "gates": gates})
        if not gates[-1]["passed"]:
            raise ValueError("Real pilot convergence failed")
        pool_path = out / f"pilot-{index}-pool.json"
        atomic_json(pool_path, pool_statistics([dict(job, groups=statistics(response))]))
        measures = [{"metric": name, "projection_metric": projection, "policy_path": str(pool_path.resolve())}
                    for name, projection in (("e_board_v1", "v1"), ("e_board_eq50", "eq50"),
                       ("e_cross_v1", "v1"), ("e_cross_v1_covered", "v1"), ("e_cross_eq50", f'eq50-fit{1-job["evaluation_fold"]}'))]
        fresh = dict(equilibrium, pooling_phase="lock-only", max_iterations=0,
                     reference_equilibrium_ev_chips=final["current_ev_chips"],
                     reference_response_sha256=file_hash(out / f"pilot-{index}-equilibrium/response.jsonl"),
                     pooling_measurements=measures)
        fresh_path = out / f"pilot-{index}-locked.json"; atomic_json(fresh_path, fresh)
        locked = run_portable_tool(binary, fresh_path, out / f"pilot-{index}-locked",
            memory_bytes=fresh["memory_budget_bytes"], threads=6, seconds=fresh["seconds"]+300,
            job_memory_bytes=5*1024**3)
        if locked["status"] != "completed":
            raise RuntimeError(locked["failure"])
        actual = [json.loads(line) for line in (out / f"pilot-{index}-locked/response.jsonl").read_text().splitlines()]
        gates.append(check_lock_only(response, actual, fresh["pot"], fresh["reference_response_sha256"]))
        locked_seconds.append(locked["elapsed_seconds"])
        replay = dict(fresh, pooling_phase="relock", max_iterations=final["iterations"], target_pct_pot=-1)
        replay_path = out / f"pilot-{index}-locked-replay.json"; atomic_json(replay_path, replay)
        replay_runtime = run_portable_tool(binary, replay_path, out / f"pilot-{index}-locked-replay",
            memory_bytes=replay["memory_budget_bytes"], threads=6, seconds=replay["seconds"]+300,
            job_memory_bytes=5*1024**3)
        if replay_runtime["status"] != "completed":
            raise RuntimeError(replay_runtime["failure"])
        solved = [json.loads(line) for line in (out / f"pilot-{index}-locked-replay/response.jsonl").read_text().splitlines()]
        gates.append(check_locked_br_parity(actual, solved, fresh["pot"]))
        atomic_json(out / "gates.json", {"passed": all(g["passed"] for g in gates), "gates": gates})
    source.db.close(); source.get.cache_clear()
    return gates, locked_seconds


def main():
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Owned qualification stopped")))
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "reference-binary", "fixture", "river-fixtures", "plan", "prepared", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    gate_k = json.loads((a.prepared / "gate-k.json").read_text())
    if not gate_k["passed"]:
        raise ValueError("Prepared key factorization gate failed")
    gates = [gate_k] + parity(a.binary, a.river_fixtures, a.out / "river-parity")
    gates += singleton(a.binary, a.reference_binary, a.fixture, a.out / "singleton", threads=6)["gates"]
    real_gates, locked_seconds = real_v4(a.binary, a.plan, a.prepared, a.out / "real-v4")
    gates += real_gates
    atomic_json(a.out / "qualification.json", {"passed": all(g["passed"] for g in gates), "gates": gates,
                "binary_sha256": file_hash(a.binary), "plan_sha256": file_hash(a.plan),
                "lock_only_pilot_seconds": locked_seconds, "threads_per_worker": 6,
                "linux_parity": {"status": "not-run", "reason": "M4 revision 3; no Linux deployment"}})


if __name__ == "__main__":
    main()
