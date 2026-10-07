"""Cross-root equality and frozen signed placement, rather than mirrored code."""

import pytest
from src.diagnostics.board_pooling import pool_statistics, public_line, readout
from src.diagnostics.flop_check import fixture_root
from src.game.types import Street


def row(spot, names, values, weight=1, lineage=1):
    return {"lineage": lineage, "spot": spot, "board_weight": weight,
            "groups": [{"metric": "v1", "key": "same-actual-key", "names": names,
                        "mass": sum(values), "action_mass": values}]}


def test_cross_board_policy_is_shared_and_aligns_menu_names():
    result = pool_statistics([
        row("wet", ["fold", "call"], [1., 0.], weight=3),
        row("dry", ["call", "fold"], [2., 0.]),
        row("wet", ["fold", "call"], [0., 1.], lineage=2),
    ])
    group = result["groups"][0]
    assert group["names"] == ["fold", "call"]
    assert group["probabilities"] == pytest.approx([.6, .4])
    assert group["roots"] == 2
    assert result["groups"][1]["probabilities"] == [0., 1.]


def test_pooling_refuses_duplicate_roots_incompatible_menus_and_nan():
    r = row("root", ["fold", "call"], [1., 0.])
    with pytest.raises(ValueError, match="Duplicate root"):
        pool_statistics([r, r])
    with pytest.raises(ValueError, match="incompatible menus"):
        pool_statistics([r, row("other", ["fold", "jam"], [1., 0.])])
    with pytest.raises(ValueError, match="Invalid projection"):
        pool_statistics([row("root", ["fold", "call"], [float("nan"), 1.])])


def test_public_line_retains_sizes_and_omits_physical_boards():
    a = fixture_root("limped", seed=101, street=Street.TURN)
    b = fixture_root("limped", seed=102, street=Street.TURN)
    c = fixture_root("min-raised", seed=101, street=Street.TURN)
    assert public_line(a, 0) == public_line(b, 0)
    assert public_line(a, 0) != public_line(c, 0)


def test_signed_readout_does_not_clip_or_claim_lower_bound():
    assert readout(3., .5, .75)["classification"] == "trainer/coverage consistent"
    assert readout(3., .5, 2.5)["classification"] == "board-pooling consistent"
    assert readout(3., .5, -.1)["placement"] < 0
    assert readout(.55, .5, .6)["placement"] is None


def test_disk_average_preserves_actual_policy_inference_and_source_hash(tmp_path):
    from tests.diagnostics.test_cfr_average import fixture
    from src.blueprint.average import AveragePolicy
    from src.diagnostics.cfr_average import extract
    from src.diagnostics.board_pooling_policy import build_index, DiskAverage
    _, view, checkpoint, _, spec = fixture(tmp_path)
    output = tmp_path / "average.gz"
    receipt = extract(checkpoint, spec, output)
    indexed = dict(spec, path=output.name, sha256=receipt["sha256"])
    db = tmp_path / "average.sqlite"
    inventory = build_index(indexed, tmp_path, db)
    assert inventory["rows"] == 2
    assert DiskAverage(db, indexed).distribution(view) == AveragePolicy(output, receipt["sha256"]).distribution(view)
    with pytest.raises(ValueError, match="hash"):
        build_index(dict(indexed, sha256="0" * 64), tmp_path, tmp_path / "wrong.sqlite")


def test_missing_later_lineage_stops_before_preparation(tmp_path, monkeypatch):
    import json
    from scripts import prepare_board_pooling as exporter
    from src.diagnostics.saved_hu20 import file_hash

    present = tmp_path / "first.gz"
    present.write_bytes(b"retained source")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"policies": [
        {"path": present.name, "sha256": file_hash(present)},
        {"path": "archived-later.gz", "sha256": "0" * 64}]}))
    def forbidden_tree(*args, **kwargs):
        pytest.fail("Tree preparation preceded lineage admission")
    monkeypatch.setattr(exporter, "compile_tree", forbidden_tree)
    out = tmp_path / "prepared"
    with pytest.raises(FileNotFoundError, match="archived-later"):
        exporter.prepare(plan, tmp_path, out)
    assert not out.exists()


def test_companion_compares_named_actions_and_keeps_missing_lineages():
    from scripts.audit_board_pooling_keys import compare, selected_keys
    assert selected_keys({"top_keys_by_excess_mass": {"A": [{"v1_key": "x"}], "B": [{"v1_key": "x"}, {"v1_key": "y"}]}}, 1) == ["x"]
    assert compare({1: {"names": ["fold", "call"], "average": [1., 0.], "current": [.5, .5]},
                    2: {"names": ["call", "fold"], "average": [1., 0.], "current": [.5, .5]}})[0]["average_tv"] == 1


def test_two_physical_boards_pool_by_real_information_key():
    from itertools import combinations
    from src.blueprint.abstraction import choices, information_key, HU20_UNCAPPED_SCHEMA
    from src.blueprint.search import DECK
    from src.game.observation import replay
    from src.diagnostics.flop_check import descriptor
    roots = [fixture_root("limped", seed=s, street=Street.TURN) for s in (102, 104)]
    boards = [replay(r, 0, ()).board for r in roots]
    assert boards[0] != boards[1]
    compatible = [c for c in DECK if c not in (*boards[0], *boards[1])]
    hand = next(h for h in combinations(compatible, 2) if descriptor(h, boards[0]) == descriptor(h, boards[1]))
    views = [replay(r, 1, hand) for r in roots]
    menus = [choices(v, raise_cap=None, free_fold=False) for v in views]
    keys = [information_key(v, m, schema=HU20_UNCAPPED_SCHEMA) for v, m in zip(views, menus)]
    assert keys[0] == keys[1]
    names = [c.name for c in menus[0]]; count = len(names)
    records = []
    for spot, action in zip(("a", "b"), (0, 1)):
        mass = [0.] * count; mass[action] = 1.
        record = row(spot, names, mass); record["groups"][0]["key"] = keys[0]; records.append(record)
    policy = pool_statistics(records)["groups"][0]["probabilities"]
    assert policy == [.5, .5] + [0.] * (count - 2)
    # Independent two-world decision oracle: each board rewards its own pure
    # action by one unit. One shared key necessarily gives their mean 0.5.
    assert (policy[0] + policy[1]) / 2 == .5


def test_shared_equity_codebook_preserves_equal_features_across_boards():
    import numpy as np
    from src.diagnostics.board_pooling_features import shared_codebook
    def feature(hist):
        return {"board": [], "boards": [], "holdings": [["a", "b"], ["c", "d"]],
                "histograms": np.asarray(hist), "codes": np.zeros((2, 2), dtype=int),
                "river_equity": np.asarray([[.1, .9]])}
    output, codebook = shared_codebook([feature([[1., 0.], [0., 1.]]), feature([[0., 1.], [1., 0.]])], k=2)
    assert output[0]["labels"]["50"][0] == list(reversed(output[1]["labels"]["50"][0]))
    assert output[0]["labels"]["50"][1] == output[1]["labels"]["50"][1]
    assert len(codebook["river_edges"]) == 1


def test_replay_gate_rejects_changed_equilibrium_and_menus():
    from src.diagnostics.board_pooling_results import check_replay, common_mask
    first = [{"event": "gate", "gate": "V1", "passed": True},
             {"event": "pooling_statistics", "groups": [{"metric": "v1", "key": "a", "names": ["check"], "mass": 1., "action_mass": [1.]}]},
             {"event": "completion", "status": "solved", "exploitability_pct_pot": .1,
              "iterations": 25, "compressed": True, "current_ev_chips": [1., -1.], "mes_ev_chips": [1.1, -.9]}]
    assert check_replay(first, first, 200)["passed"]
    changed = first[:-1] + [dict(first[-1], current_ev_chips=[2., -2.])]
    with pytest.raises(ValueError, match="values differ"):
        check_replay(first, changed, 200)
    manifest = {"jobs": [{"spot": "a", "lineage": 1, "job": "a-1"}, {"spot": "a", "lineage": 2, "job": "a-2"}], "support_exclusions": []}
    assert common_mask(manifest, {"a-1": {"eligible": True}, "a-2": {"eligible": True}}, [{"seed": 1}, {"seed": 2}, {"seed": 3}])["admitted"] == []


def test_report_bootstraps_paired_boards_and_ratio_of_means():
    from scripts.report_board_pooling import summarize
    rows = []
    for spot, bp in (("a", 1.), ("b", 9.)):
        for lineage in (1, 2, 3):
            for seat in (0, 1):
                values = {"e_bp": bp, "e_root_v1": .2 * bp, "e_cross_v1": .6 * bp, "e_cross_eq50": .3 * bp, "e_board_v1": .4 * bp, "e_board_eq50": .2 * bp}
                rows.append({"spot": spot, "board_weight": 1., "lineage": lineage, "seat": seat,
                             **values, **{k + "_pct_pot": v * 50 for k, v in values.items()}})
    result = summarize(rows, resamples=100)
    assert result["boards"] == 2
    assert result["placement"]["point"] == pytest.approx(.5)
    assert result["ratios"]["per_root_over_bp"]["point"] == pytest.approx(.2)
    assert result["metrics"]["e_bp_pct_pot"]["mean"] == 250.


def test_incomplete_campaign_report_cannot_classify(tmp_path):
    import json
    from src.diagnostics.saved_hu20 import file_hash
    from scripts.report_board_pooling import report
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"roots": [{"spot": "a", "board_weight": 1.}]}))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"folds": {"a": 0}}))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"corpus": {"path": str(corpus), "sha256": file_hash(corpus)},
                              "policies": [{"seed": s} for s in (1, 2, 3)], "bootstrap_seed": 1,
                              "bootstrap_resamples": 20, "minimum_common_boards": 32,
                              "minimum_common_board_weight_fraction": .8, "minimum_common_boards_per_fold": 16,
                              "crossfit": {"path": str(split), "sha256": file_hash(split)}}))
    run = tmp_path / "run"; run.mkdir()
    result = report(plan, run, tmp_path / "report")
    assert result["common_boards"] == 0
    assert result["classification"] == "incomplete or insufficient common coverage; no hypothesis decision"


def test_crossfit_never_uses_evaluation_half_action_masses():
    from copy import deepcopy
    from src.diagnostics.board_pooling import crossfit_policies
    rows = [{"lineage": 1, "spot": spot, "board_weight": 1,
             "groups": [{"metric": "v1", "key": "shared", "names": ["fold", "call"],
                         "mass": 1, "action_mass": action}]}
            for spot, action in (("a", [1, 0]), ("b", [0, 1]))]
    split = {"a": 0, "b": 1}
    policies = crossfit_policies(rows, split)
    assert policies["0"]["groups"][0]["probabilities"] == [0, 1]
    assert policies["1"]["groups"][0]["probabilities"] == [1, 0]
    changed = deepcopy(rows); changed[0]["groups"][0]["action_mass"] = [.2, .8]
    assert crossfit_policies(changed, split)["0"] == policies["0"]
    assert policies["0"]["training_spots"] == ["b"]


def test_frozen_crossfit_split_and_replay_inventory():
    import json
    from pathlib import Path
    from src.diagnostics.saved_hu20 import file_hash
    plan = json.loads(Path("configs/diagnostics/hu20-board-pooling.json").read_text())
    path = Path(plan["crossfit"]["path"])
    assert file_hash(path) == plan["crossfit"]["sha256"]
    frozen = json.loads(path.read_text())
    assert sorted(frozen["folds"].values()).count(0) == 20
    assert sorted(frozen["folds"].values()).count(1) == 20
    assert len(set(frozen["replay_boards"])) == 6
    assert [sum(frozen["folds"][s] == f for s in frozen["replay_boards"]) for f in (0, 1)] == [3, 3]
    assert frozen["replay_jobs"] == plan["replay_jobs"] == 18


def test_heldout_features_cannot_change_training_codebook():
    import numpy as np
    from copy import deepcopy
    from src.diagnostics.board_pooling_features import crossfit_codebooks
    def feature(hist, equity):
        return {"board": [], "boards": [], "holdings": [["a", "b"], ["c", "d"]],
                "histograms": np.asarray(hist), "codes": np.zeros((2, 2), dtype=int),
                "river_equity": np.asarray([equity])}
    features = [feature([[1., 0.], [0., 1.]], [.1, .9]), feature([[.5, .5], [.2, .8]], [.4, .6])]
    _, original = crossfit_codebooks(features, [0, 1], k=2, seed=7)
    changed = deepcopy(features)
    changed[1]["histograms"] = np.asarray([[.9, .1], [.8, .2]])
    changed[1]["river_equity"] = np.asarray([[.8, .9]])
    _, result = crossfit_codebooks(changed, [0, 1], k=2, seed=7)
    assert result["0"] == original["0"]
    assert result["1"] != original["1"]


def test_lock_only_requires_hash_linked_equilibrium_not_uniform_ev():
    from src.diagnostics.board_pooling_results import check_lock_only
    reference = [{"event": "gate", "gate": "V1", "passed": True},
                 {"event": "completion", "status": "solved", "iterations": 20,
                  "current_ev_chips": [20., -20.], "exploitability_pct_pot": .1}]
    fresh = [{"event": "gate", "gate": "V1", "passed": True},
             {"event": "pooling_metric", "target_solver_seat": 0, "responder_br_chips": 30.,
              "reference_responder_value_chips": -20., "gain_bb": .5, "gain_pct_pot": 25.},
             {"event": "completion", "status": "locked-evaluated", "iterations": 0,
              "reference_equilibrium_ev_chips": [20., -20.], "reference_response_sha256": "fixed"}]
    assert check_lock_only(reference, fresh, 200, "fixed")["passed"]
    with pytest.raises(ValueError, match="reference"):
        check_lock_only(reference, fresh, 200, "different")
    fresh[-1]["reference_equilibrium_ev_chips"] = [0., 0.]
    with pytest.raises(ValueError, match="reference"):
        check_lock_only(reference, fresh, 200, "fixed")


def test_provider_lease_respects_pause_clock_and_fee_reserve():
    from scripts.watch_board_pooling_rental import validate_lease
    quote = {"maximum_hours": 12, "rate_ceiling_usd_per_hour": .34, "total_cap_usd": 5}
    lease = {"owner_resumed": True, "owner_approved_quote": True, "name": "one-owned-pod",
             "cpu_id": "qualified-cpu", "started": 100, "deadline": 43300,
             "upper_rate": .34, "fee_reserve_usd": .92}
    validate_lease(lease, quote)
    for change in ({"owner_resumed": False}, {"deadline": 43301},
                   {"upper_rate": .35}, {"fee_reserve_usd": .1}):
        with pytest.raises(ValueError, match="lease"):
            validate_lease(dict(lease, **change), quote)


def test_paused_production_never_reaches_machine_or_solver(tmp_path, monkeypatch):
    import json
    from argparse import Namespace
    import scripts.run_board_pooling as runner
    plan = tmp_path / "plan.json"; plan.write_text(json.dumps({"format": "hu20-board-pooling-plan-v3"}))
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({"owner_approved_quote": True, "owner_resumed": False, "qualification_passed": True}))
    monkeypatch.setattr(runner, "resource_snapshot", lambda: pytest.fail("Paused work inspected production machine"))
    with pytest.raises(ValueError, match="required"):
        runner.campaign(Namespace(plan=plan, approval=approval))


def test_covered_sensitivity_only_fills_absent_or_zero_training_groups():
    from src.diagnostics.board_pooling import covered_context_policy
    train = pool_statistics([row("train", ["fold", "call"], [1, 0])])
    local = pool_statistics([row("local", ["fold", "call"], [0, 1])])
    assert covered_context_policy(train, local)["groups"][0]["probabilities"] == [1, 0]
    train["groups"][0]["mass"] = 0
    assert covered_context_policy(train, local)["groups"][0]["probabilities"] == [0, 1]
    assert covered_context_policy(dict(train, groups=[]), local)["groups"][0]["probabilities"] == [0, 1]


def test_m4_admission_reuses_cache_inclusive_law_and_refuses_four_workers():
    from scripts.hu20_search_runtime import macos_memory_admission
    from src.diagnostics.pooling_runtime import admit_m4
    vm = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n" + "\n".join(f"{key}: 200000." for key in ("Pages free", "Pages inactive", "Pages speculative", "File-backed pages"))
    snapshot = dict(macos_memory_admission(vm), effective_cores=10)
    budget = {"workers": 1, "threads_per_worker": 6, "worker_rss_bytes": 5*1024**3,
              "aggregate_rss_bytes": 8*1024**3, "148_merged": True, "148_processes_empty": True,
              "149_owner_authorized": True, "ownership_evidence": "owner released M4", "followup_claim": "none",
              "minimum_disk_free_bytes": 20*1024**3}
    assert admit_m4(budget, snapshot)["rss_limit_bytes"] == 8*1024**3
    with pytest.raises(ValueError, match="worker shape"):
        admit_m4(dict(budget, workers=4), snapshot)


def test_missing_key_coverage_preserves_fold_lineage_and_street():
    from scripts.report_board_pooling import coverage_by_fold
    records = [{"evaluation_fold": f, "lineage": l, "board_weight": 2,
                "fallback_by_metric": {"e_cross_v1": {"turn": [10, m, 1]}}}
               for f,l,m in ((0,1,.6),(1,1,0),(0,2,0))]
    result = coverage_by_fold(records)
    assert result[0]["fraction"] == pytest.approx(.06)
    assert len(result) == 3
    assert result[0]["missing_key_reach_mass"] == 1.2


def test_recovery_refuses_changed_artifact_or_unpinned_inventory(tmp_path):
    import json
    from scripts.prepare_board_pooling import recovery_files
    from src.diagnostics.saved_hu20 import file_hash
    prior = tmp_path / "prior"; prior.mkdir()
    data = prior / "compact.json"; data.write_text("original")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"files": [{"path": "prior/compact.json",
        "bytes": data.stat().st_size, "sha256": file_hash(data)}]}))
    digest = file_hash(inventory)
    with pytest.raises(ValueError, match="inventory fingerprint"):
        recovery_files(prior, inventory, "0" * 64)
    checked = recovery_files(prior, inventory, digest)
    assert checked("compact.json") == data
    data.write_text("changed!")
    with pytest.raises(ValueError, match="Recovery artifact"):
        checked("compact.json")


def test_preparation_reuses_export_only_after_range_and_feature_checks(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import prepare_board_pooling as exporter
    from src.diagnostics.flop_check import atomic_json
    from src.diagnostics.saved_hu20 import file_hash

    raw = tmp_path / "average.gz"; raw.write_bytes(b"immutable input")
    spec = {"path": raw.name, "sha256": file_hash(raw), "seed": 123}
    corpus = tmp_path / "corpus.json"
    atomic_json(corpus, {"chosen_line": "fixed", "roots": [
        {"spot": "root", "button": 0, "board": [1, 2, 3, 4], "board_weight": 1}]})
    split = tmp_path / "split.json"
    atomic_json(split, {"folds": {"root": 0}, "replay_boards": ["root"]})
    plan = tmp_path / "plan.json"
    atomic_json(plan, {"policies": [spec], "corpus": {"path": str(corpus), "sha256": file_hash(corpus)},
        "crossfit": {"path": str(split), "sha256": file_hash(split)}, "boards": 1, "equity_histogram_bins": 20,
        "equity_k": 50, "equity_cluster_seed": 1, "jobs_total": 1, "maximum_iterations": 10,
        "progress_every": 1, "target_pct_pot": .2, "per_solve_seconds": 60})
    feature = {"board": [1, 2, 3, 4], "codes": [[1], [2]]}
    calls = []
    def index(spec, inputs, path):
        path.write_bytes(b"immutable index")
        return {"source_sha256": spec["sha256"], "index_sha256": file_hash(path), "path": str(path)}
    def tables(*args):
        calls.append("export")
        return {"strategy": [0.25, 0.75]}
    monkeypatch.setattr(exporter, "build_index", index)
    monkeypatch.setattr(exporter, "DiskAverage", lambda *args: SimpleNamespace(
        db=SimpleNamespace(close=lambda: None), get=SimpleNamespace(cache_clear=lambda: None)))
    monkeypatch.setattr(exporter, "replay_root", lambda record: record)
    monkeypatch.setattr(exporter, "public_line", lambda *args: "fixed")
    monkeypatch.setattr(exporter, "compile_tree", lambda root: ({"seat_map": [0, 1], "board": root["board"],
        "template": [(False, False), (False, False)]}, None))
    monkeypatch.setattr(exporter, "gate_k", lambda *args: {"passed": True, "samples": 100000})
    monkeypatch.setattr(exporter, "card_features", lambda *args: None)
    monkeypatch.setattr(exporter, "crossfit_codebooks", lambda *args, **kwargs: ([feature], {"seed": 1}))
    monkeypatch.setattr(exporter, "add_pool_keys", lambda *args: {"key": "fixed"})
    monkeypatch.setattr(exporter, "export_policy_tables", tables)
    ranges = {0: [((5, 6), 1.)], 1: [((7, 8), 1.)]}
    monkeypatch.setattr(exporter, "public_ranges", lambda *args: (ranges, {}))
    prior = tmp_path / "prior"
    exporter.prepare(plan, tmp_path, prior)
    inventory = tmp_path / "inventory.json"
    atomic_json(inventory, {"files": [{"path": str(p.relative_to(tmp_path)),
        "bytes": p.stat().st_size, "sha256": file_hash(p)} for p in prior.rglob("*") if p.is_file()]})
    out = tmp_path / "recovered"
    args = dict(reuse_prepared=prior, reuse_manifest=inventory, reuse_manifest_sha256=file_hash(inventory))
    exporter.prepare(plan, tmp_path, out, **args)
    assert calls == ["export"]
    assert (out / "jobs/root-0/compact.json").read_bytes() == (prior / "jobs/root-0/compact.json").read_bytes()
    exported = json.loads((out / "jobs/root-0/request.json").read_text())
    assert exported["compact_path"] == str(out / "jobs/root-0/compact.json")
    # Immutable old bytes alone do not authorize reuse under changed ranges.
    ranges[0] = [((5, 6), .5)]
    with pytest.raises(ValueError, match="request/ranges/menu"):
        exporter.prepare(plan, tmp_path, tmp_path / "changed-ranges", **args)
    ranges[0] = [((5, 6), 1.)]
    feature["codes"] = [[9], [2]]
    with pytest.raises(ValueError, match="card features"):
        exporter.prepare(plan, tmp_path, tmp_path / "changed-features", **args)


def test_m4_seven_gib_amendment_keeps_family_arena_and_worker_shape():
    from src.diagnostics.pooling_runtime import worker_rss_limit
    budget = {"host": "m4", "workers": 1, "threads_per_worker": 6,
              "worker_rss_bytes": 7*1024**3, "aggregate_rss_bytes": 8*1024**3,
              "arena_bytes": 4*1024**3,
              "worker_rss_amendment": "owner-approved-m4-7gib-before-main"}
    assert worker_rss_limit(budget) == 7*1024**3
    for change in ({"host": "linux"}, {"worker_rss_amendment": ""},
                   {"worker_rss_bytes": 8*1024**3}, {"aggregate_rss_bytes": 9*1024**3},
                   {"arena_bytes": 5*1024**3}, {"workers": 2}, {"threads_per_worker": 8}):
        with pytest.raises(ValueError, match="owner-approved M4"):
            worker_rss_limit(dict(budget, **change))


def test_real_pilot_records_separate_rss_and_stops_at_failed_lock(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import qualify_board_pooling as q
    from src.diagnostics.flop_check import atomic_json
    from src.diagnostics.saved_hu20 import file_hash
    prepared = tmp_path / "prepared"; prepared.mkdir()
    corpus = tmp_path / "corpus.json"
    atomic_json(corpus, {"roots": [{"spot": str(i)} for i in range(3)]})
    plan = tmp_path / "plan.json"
    atomic_json(plan, {"corpus": {"path": str(corpus)}, "policies": [{"sha256": "pinned"}]})
    index = prepared / "policy-0.sqlite"; index.write_bytes(b"index")
    atomic_json(prepared / "policy-0-index.json", {"index_sha256": file_hash(index)})
    compact = prepared / "compact.json"; compact.write_text('{}')
    request = prepared / "request.json"
    atomic_json(request, {"spot": "0", "pooling_phase": "collect", "pot": 200,
                "ranges": [[], []], "seat_map": [0,1], "memory_budget_bytes": 4*1024**3,
                "compact_path": str(compact), "seconds": 1})
    job = {"spot": "0", "policy_index": 0, "request": str(request),
           "request_sha256": file_hash(request), "compact_sha256": file_hash(compact),
           "evaluation_fold": 0}
    atomic_json(prepared / "manifest.json", {"jobs": [job]})
    monkeypatch.setattr(q, "DiskAverage", lambda *a: SimpleNamespace())
    monkeypatch.setattr(q, "replay_root", lambda r: r)
    monkeypatch.setattr(q, "monte_carlo", lambda *a, **k: {"ci95": [-1,1], "deals": 20000})
    monkeypatch.setattr(q, "statistics", lambda r: [])
    monkeypatch.setattr(q, "pool_statistics", lambda r: {"groups": []})
    calls = []
    def tool(binary, path, out, **limits):
        req = json.loads(path.read_text()); calls.append((req, limits))
        out.mkdir()
        (out / 'response.jsonl').write_text(json.dumps({"event": "both_blueprint_ev", "current_ev_chips": [0,0]})+'\n'+
            json.dumps({"event": "completion", "status": "solved", "exploitability_pct_pot": .1,
                        "current_ev_chips": [0,0], "iterations": 25})+'\n')
        failed = req.get("pooling_phase") == "lock-only"
        return {"status": "failure" if failed else "completed", "failure": "RSS budget exceeded" if failed else None,
                "peak_job_rss_bytes": (7*1024**3+1) if failed else len(calls)*1024**3,
                "elapsed_seconds": len(calls)}
    monkeypatch.setattr(q, "run_portable_tool", tool)
    out = tmp_path / "qualify"
    with pytest.raises(RuntimeError, match="RSS budget exceeded"):
        q.real_v4(tmp_path/'binary', plan, prepared, out, worker_rss_bytes=7*1024**3)
    assert len(calls) == 3
    assert all(l['job_memory_bytes']==7*1024**3 and l['memory_bytes']==4*1024**3 for _,l in calls)
    assert json.loads(request.read_text())["pooling_phase"] == "collect"
    resources = json.loads((out/'pilot-0-resources.json').read_text())["stages"]
    assert resources['solve']['peak_job_rss_bytes'] == 2*1024**3
    assert resources['lock_only']['peak_job_rss_bytes'] == 7*1024**3+1
    assert not (out/'pilot-1.json').exists()
