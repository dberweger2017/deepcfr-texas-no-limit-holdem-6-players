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
