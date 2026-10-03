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
    from src.diagnostics.cfr_average import DiagnosticAverage, extract
    from src.diagnostics.board_pooling_policy import build_index, DiskAverage
    _, view, checkpoint, _, spec = fixture(tmp_path)
    output = tmp_path / "average.gz"
    receipt = extract(checkpoint, spec, output)
    indexed = dict(spec, path=output.name, sha256=receipt["sha256"])
    db = tmp_path / "average.sqlite"
    inventory = build_index(indexed, tmp_path, db)
    assert inventory["rows"] == 2
    assert DiskAverage(db, indexed).distribution(view) == DiagnosticAverage(output, receipt["sha256"]).distribution(view)
    with pytest.raises(ValueError, match="hash"):
        build_index(dict(indexed, sha256="0" * 64), tmp_path, tmp_path / "wrong.sqlite")


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
                values = {"e_bp": bp, "e_root_v1": .2 * bp, "e_board_v1": .6 * bp, "e_board_eq50": .3 * bp}
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
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"corpus": {"path": str(corpus), "sha256": file_hash(corpus)},
                              "policies": [{"seed": s} for s in (1, 2, 3)], "bootstrap_seed": 1,
                              "bootstrap_resamples": 20, "minimum_common_boards": 32,
                              "minimum_common_board_weight_fraction": .8}))
    run = tmp_path / "run"; run.mkdir()
    result = report(plan, run, tmp_path / "report")
    assert result["common_boards"] == 0
    assert result["classification"] == "incomplete or insufficient common coverage; no hypothesis decision"
