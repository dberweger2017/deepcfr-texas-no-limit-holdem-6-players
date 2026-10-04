import json

import pytest

from scripts.check_board_pooling_engineering import compare


def responses(tmp_path, old, new):
    paths = [tmp_path / "old.jsonl", tmp_path / "new.jsonl"]
    for path, value in zip(paths, (old, new), strict=True):
        path.write_text(json.dumps(value) + "\n")
    return paths


def test_exact_parity_allows_only_top_level_native_telemetry(tmp_path):
    value = {"event": "pooling_statistics", "groups": [{"mass": .3, "action_mass": [.1, .2]}],
             "elapsed_seconds": 10, "solver_peak_rss_bytes": 50}
    changed = dict(value, elapsed_seconds=1, solver_peak_rss_bytes=40)
    result = compare(*responses(tmp_path, value, changed))
    assert result["passed"]
    assert result["reference_fingerprint"]["response_sha256"] != result["actual_fingerprint"]["response_sha256"]


@pytest.mark.parametrize("old,new", [(.3, .30000000000000004), (-0.0, 0.0), (1, True)])
def test_exact_parity_rejects_small_changes_and_signed_zero(tmp_path, old, new):
    with pytest.raises(ValueError, match="Scientific responses differ"):
        compare(*responses(tmp_path, {"gain": old}, {"gain": new}))


def test_nested_timing_and_omitted_groups_are_not_silently_ignored(tmp_path):
    with pytest.raises(ValueError):
        compare(*responses(tmp_path, {"groups": [{"elapsed_seconds": 1}]},
                                   {"groups": [{"elapsed_seconds": 2}]}))
    with pytest.raises(ValueError):
        compare(*responses(tmp_path, {"groups": [1, 2]}, {"groups": [1]}))


def test_recovery_rejects_unpinned_evidence_before_native_work(tmp_path):
    from scripts.qualify_board_pooling import recover_engineering_pilot
    from src.diagnostics.saved_hu20 import file_hash
    binary = tmp_path / "binary"; binary.write_bytes(b"fixture")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"passed": True, "binary_sha256": file_hash(binary),
        "baseline": str(tmp_path / "baseline"), "actual": str(tmp_path / "actual"), "pinned_files": {}}))
    with pytest.raises(ValueError, match="unpinned required evidence"):
        recover_engineering_pilot(binary, receipt, tmp_path / "out", worker_rss_bytes=7 * 1024**3,
                                 expected_job={}, expected_policy={})


def test_verified_first_pilot_reuses_exact_results_and_runs_fresh_replay(tmp_path, monkeypatch):
    from scripts import qualify_board_pooling as q
    from src.diagnostics.saved_hu20 import file_hash
    baseline, actual, out = (tmp_path / name for name in ("baseline", "actual", "out"))
    out.mkdir()
    def write(path, value, lines=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(v) + "\n" for v in value) if lines else json.dumps(value))
    binary = tmp_path / "binary"; binary.write_bytes(b"fixture")
    compact = tmp_path / "compact.json"; compact.write_text("{}")
    request = {"spot": "first", "pot": 200, "compact_path": str(compact),
               "memory_budget_bytes": 4*1024**3, "seconds": 100}
    write(baseline / "pilot-0-equilibrium.json", request)
    tree = {"event": "gate", "gate": "V1", "passed": True}
    bp = {"event": "both_blueprint_ev", "current_ev_chips": [-1., 1.]}
    metric = {"event": "pooling_metric", "metric": "e_board_v1", "target_solver_seat": 0,
              "gain_bb": .01, "gain_pct_pot": .5, "responder_br_chips": 2., "reference_responder_value_chips": 1.}
    eq = [tree, {"event": "pooling_statistics", "groups": []}, bp, metric,
          {"event": "completion", "status": "solved", "iterations": 25, "compressed": True,
           "exploitability_pct_pot": .1, "current_ev_chips": [-1., 1.], "mes_ev_chips": [0., 2.]}]
    write(baseline / "pilot-0-equilibrium/response.jsonl", eq, True)
    reference_hash = file_hash(baseline / "pilot-0-equilibrium/response.jsonl")
    fresh = dict(request, reference_response_sha256=reference_hash)
    write(baseline / "pilot-0-locked.json", fresh)
    locked = [tree, bp, metric, {"event": "completion", "status": "locked-evaluated", "iterations": 0,
              "reference_response_sha256": reference_hash, "reference_equilibrium_ev_chips": [-1., 1.]}]
    write(baseline / "pilot-0-locked/response.jsonl", locked, True)
    write(baseline / "pilot-0/response.jsonl", [bp], True)
    write(baseline / "gates.json", {"gates": [{"gate": "real-export-V4", "spot": "first", "passed": True,
          "source_sha256": "export", "native_mc": {"deals": 20000}}]})
    runtimes = {}
    for stage, rows, reqfile in (("solve", eq, "pilot-0-equilibrium.json"), ("lock-only", locked, "pilot-0-locked.json")):
        write(actual / stage / "response.jsonl", rows, True)
        runtime = {"status": "completed", "binary_sha256": file_hash(binary),
          "request_sha256": file_hash(baseline / reqfile), "job_memory_budget_bytes": 7*1024**3,
          "memory_budget_bytes": 4*1024**3, "rayon_threads": 6, "elapsed_seconds": 10.}
        write(actual / stage / "result.json", runtime); runtimes[stage] = runtime
    paths = list(baseline.rglob("*.json*")) + list(actual.rglob("*.json*"))
    receipt = tmp_path / "receipt.json"
    write(receipt, {"passed": True, "binary_sha256": file_hash(binary), "baseline": str(baseline),
                   "actual": str(actual), "pinned_files": {str(p): file_hash(p) for p in paths}})
    calls = []
    def replay(binary, path, destination, **kwargs):
        calls.append((json.loads(path.read_text()), kwargs))
        write(destination / "response.jsonl", eq, True)
        return runtimes["solve"]
    monkeypatch.setattr(q, "run_portable_tool", replay)
    gates, seconds = q.recover_engineering_pilot(binary, receipt, out, worker_rss_bytes=7*1024**3,
        expected_job={"request_sha256": file_hash(baseline / "pilot-0-equilibrium.json"),
                      "compact_sha256": file_hash(compact)}, expected_policy={"sha256": "export"})
    assert all(g["passed"] for g in gates)
    assert seconds == 10 and len(calls) == 1
    assert calls[0][0]["pooling_phase"] == "relock"
    assert calls[0][0]["max_iterations"] == 25
    assert calls[0][1]["threads"] == 6
    assert (out / "pilot-0-resources.json").exists()
