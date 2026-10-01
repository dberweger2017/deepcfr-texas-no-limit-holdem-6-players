"""Behavioral checks for native export, full-key projection and reporting."""

from dataclasses import replace
from itertools import combinations
import json

import numpy as np
import pytest

from src.blueprint.abstraction import _postflop, choices, information_key, HU20_UNCAPPED_SCHEMA
from src.blueprint.search import DECK
from src.diagnostics.flop_check import (compile_tree, decode_descriptor, descriptor,
    descriptor_code, factored_key, fixture_root, key_template)
from src.diagnostics.flop_check_analysis import (bootstrap_spots, decision_rule,
    emd_clusters, equity_quantiles, projection, uniform_river_equities)
from src.diagnostics.exact_ranker import exact_seven_card
from src.game.observation import replay
from src.game.types import Street
from scripts.monitor_flop_check import Monitor


def test_native_amounts_position_and_root_types():
    for button in (0, 1):
        for kind, pot in (("limped", 200), ("min-raised", 400), ("3-bet", 600)):
            root = fixture_root(kind, street=Street.RIVER, button=button)
            req, histories = compile_tree(root)
            assert req["pot"] == pot
            assert req["seat_map"] == [1 - button, button]
            assert req["nodes"][0]["player"] == 0
            for node, history in zip(req["nodes"], histories, strict=True):
                if node["terminal"]:
                    continue
                view = replay(history, req["seat_map"][node["player"]], ("Ac", "Ad"))
                actual = choices(view, raise_cap=None, free_fold=False)
                assert node["native_actions"] == [
                    {"kind": c.action.kind, "raise_to": c.action.raise_to} for c in actual]
                for source, mapped in zip(actual, node["actions"], strict=True):
                    if source.action.raise_to is not None:
                        assert mapped["amount"] == source.action.raise_to


def test_factorization_and_fast_descriptor_independent_reference():
    rng = np.random.default_rng(73)
    for street in (Street.FLOP, Street.TURN, Street.RIVER):
        root = fixture_root("min-raised", street=street)
        view = replay(root, 1, ())
        for _ in range(80):
            holding = tuple(rng.choice([c for c in DECK if c not in view.board], 2, replace=False))
            own = replace(view, hole_cards=holding)
            menu = choices(own, raise_cap=None, free_fold=False)
            value = descriptor(holding, own.board)
            assert value == _postflop(holding, own.board)
            assert decode_descriptor(descriptor_code(value)) == value
            assert factored_key(key_template(own, menu), value) == information_key(own, menu, schema=HU20_UNCAPPED_SCHEMA)


def test_capped_compilation_does_not_change_blueprint_lookup_menu():
    req, _ = compile_tree(fixture_root("limped", street=Street.RIVER), raise_cap=3)
    removed = [n for n in req["nodes"] if not n["terminal"] and n["names"] != n["native_names"]]
    assert removed
    for node in removed:
        assert node["template"][-1] == node["native_names"]
        assert all(a["kind"] in ("Fold", "Call", "AllIn") for a in node["actions"])
    with pytest.raises(ValueError, match="at least three"):
        compile_tree(fixture_root("limped", street=Street.RIVER), raise_cap=2)
    with pytest.raises(MemoryError):
        compile_tree(fixture_root("limped"), max_nodes=10)


def test_uniform_equities_against_independent_exhaustive_holdings():
    board = ("7h", "Td", "Ts", "2s", "2c")
    holdings, equities = uniform_river_equities(board)
    for holding in (("4d", "5s"), ("7s", "Kd"), ("Ac", "Ad")):
        index = next(i for i, h in enumerate(holdings) if set(h) == set(holding))
        own = exact_seven_card(holding + board)
        values = [exact_seven_card(pair + board) for pair in holdings if not set(pair) & set(holding)]
        expected = sum((own > r) + 0.5 * (own == r) for r in values) / len(values)
        assert equities[index] == pytest.approx(expected, abs=1e-14)
    assert np.all((equities >= 0) & (equities <= 1))


def test_projection_pools_aliased_public_lines_and_runouts():
    view = replay(fixture_root("limped", street=Street.RIVER), 1, ("4d", "5s"))
    menu = choices(view, raise_cap=None, free_fold=False)
    template = key_template(view, menu)
    rows = [dict(line=[], board=list(view.board), holdings=[["Ac", "Ad"]], player=0,
                 strategy=[1.0, 0.0], own_weights=[1.0]),
            dict(line=[{"kind": "Check"}], board=list(view.board), holdings=[["Ac", "Ad"]], player=0,
                 strategy=[0.0, 1.0], own_weights=[3.0])]
    templates = {json.dumps(r["line"], separators=(",", ":")): template for r in rows}
    full = projection(rows, templates)
    assert full[0]["strategy"] == pytest.approx([0.25, 0.75])
    assert full[0]["strategy"] == full[1]["strategy"]
    relaxed = projection(rows, templates, per_line=True)
    assert relaxed[0]["strategy"] != relaxed[1]["strategy"]


def test_equity_clusters_and_quantiles_preserve_ties():
    hist = np.asarray([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    labels, centers = emd_clusters(hist, 50, seed=5)
    assert labels[0] == labels[1]
    assert np.allclose(centers.sum(axis=1), 1)
    assert len(set(labels)) == 3
    values = equity_quantiles([0.1, 0.1, 0.9], 200)
    assert values[0] == values[1] != values[2]


def test_spot_bootstrap_does_not_multiply_lineage_independence():
    rows = [dict(spot=s, e_bp=v, decision_eligible=True, reach_weight=1)
            for s, v in (("a", 1), ("a", 3), ("b", 8), ("b", 8))]
    summary = bootstrap_spots(rows, "e_bp")
    assert summary["independent_spots"] == 2
    assert summary["mean"] == 5
    assert summary["ci95"][0] <= 5 <= summary["ci95"][1]
    assert decision_rule(1, 0.8, 0.3, 2, eligible=True)["classification"] == "H1-consistent"
    assert decision_rule(1, 0.1, 0.3, 4, eligible=False)["classification"] == "pending"
    assert decision_rule(1, 0.1, 0.05, 4, eligible=True)["classification"] == "H3"


def test_monitor_partial_records_and_resume_do_not_repeat(tmp_path):
    class Writer:
        values = []
        def add_scalar(self, *args): self.values.append(args)
        def flush(self): pass
        def add_custom_scalars(self, value): pass
    writer = Writer(); monitor = Monitor(tmp_path, writer)
    row = dict(event="progress", spot="x", iteration=20, exploitability_pct_pot=0.15)
    text = json.dumps(row)
    path = tmp_path / "progress.jsonl"; path.write_text(text[:10])
    monitor.poll(); assert not writer.values
    path.write_text(text + "\n")
    monitor.poll(); before = len(writer.values); monitor.poll()
    assert len(writer.values) == before
    assert ("solver/exploitability_pct_pot/x", 0.15, 20) in writer.values
