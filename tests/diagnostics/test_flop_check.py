"""Behavioral checks for native export, full-key projection and reporting."""

from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

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


def test_bootstrap_preserves_strata_and_does_not_invent_singleton_precision():
    rows = [dict(spot=str(i), stratum="limped" if i < 2 else "3-bet",
                 e_bp=v, decision_eligible=True) for i, v in enumerate((1, 3, 90, 110))]
    summary = bootstrap_spots(rows, "e_bp")
    assert summary["sampling_strata"] == 2
    assert 45 <= summary["ci95"][0] <= summary["ci95"][1] <= 57
    assert bootstrap_spots(rows[:3], "e_bp")["ci95"] is None


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


def test_monitor_keeps_selected_and_unselected_policy_sets_separate(tmp_path):
    class Writer:
        def __init__(self): self.values = []
        def add_scalar(self, *args): self.values.append(args)
        def flush(self): pass
        def add_custom_scalars(self, value): pass
    writer = Writer(); monitor = Monitor(tmp_path, writer)
    base = dict(event="spot_complete", spot="root", lineage=1,
                decision_eligible=True, reach_weight=1)
    monitor.emit(dict(base, set="A", strategy="current", e_bp=100))
    monitor.emit(dict(base, set="B", strategy="current", e_bp=2))
    monitor.emit(dict(base, set="B", strategy="stored-average", e_bp=4))
    assert ("results/monitoring_only/B/current/e_bp/mean", 2, 1) in writer.values
    assert ("results/monitoring_only/B/stored-average/e_bp/mean", 4, 1) in writer.values


def test_stratified_selection_preserves_reach_weights_and_fixed_seed():
    from scripts.select_flop_check_spots import select_strata
    population = [dict(spot=str(i), kind="limped" if i < 20 else "3-bet",
                       button=i % 2, multiplicity=1 + i % 3) for i in range(40)]
    a, strata = select_strata(population, 12, 7)
    b, _ = select_strata(population[::-1], 12, 7)
    assert a == b and len(a) == 12
    assert len({r["spot"] for r in a}) == 12
    assert all(r["reach_weight"] == r["multiplicity"] / r["inclusion_probability"] for r in a)
    assert sum(s["selected"] for s in strata.values()) == 12


def test_exact_histogram_runout_accounting_and_bucket_order_invariance(tmp_path, monkeypatch):
    from src.diagnostics import flop_check_equity as eq
    deck = ("2c", "3c", "4c", "5c", "6c", "7c", "8c", "9c")
    monkeypatch.setattr(eq, "DECK", deck)
    def river(board):
        hands = tuple(combinations([c for c in deck if c not in board], 2))
        values = np.asarray([(deck.index(a) + deck.index(b)) / 16 for a, b in hands])
        return hands, values
    monkeypatch.setattr(eq, "uniform_river_equities", river)
    path = tmp_path / "equity.npz"; eq.build_equity_features(deck[:3], path, bins=5)
    with np.load(path) as data:
        assert np.allclose(data["flop_histograms"].sum(axis=1), 1)
        totals = data["turn_histograms"].sum(axis=2)
        assert set(np.unique(totals)) == {0, 2}
        assert np.all(np.isfinite(data["river_equities"]).sum(axis=0) == 3)
    bucket = eq.EquityBuckets(path, 50)
    row = dict(board=list(deck[:3]) + ["5c", "6c"])
    reverse = dict(board=list(deck[:3]) + ["6c", "5c"])
    assert bucket(row, ("7c", "8c")) == bucket(reverse, ("8c", "7c"))
    with pytest.raises(ValueError, match="Blocked"):
        bucket(row, ("5c", "8c"))


def test_public_selected_roots_replay_without_using_private_cards():
    from scripts.select_flop_check_spots import root_record, replay_root
    for button in (0, 1):
        for kind in ("limped", "min-raised", "pot-raised", "3-bet"):
            record = root_record(fixture_root(kind, button=button))
            assert root_record(replay_root(record)) == record


def test_joint_reach_marginal_and_common_range_overfold():
    from src.diagnostics.flop_check_analysis import compatible_reach, overfold_node
    hands = [["Ac", "Ad"], ["Kc", "Kd"]]
    other = [["Ac", "Qs"], ["2c", "3c"], ["Kc", "Kd"]]
    result = compatible_reach(hands, [0.4, 0.6], other, [0.2, 0.5, 0.3])
    brute = [w * sum(v for h, v in zip(other, [0.2, 0.5, 0.3]) if not set(a) & set(h))
             for a, w in zip(hands, [0.4, 0.6])]
    assert result == pytest.approx(brute)
    row = dict(board=["7s", "8s", "9d"], actions=[{"kind": "Fold"}, {"kind": "Call"}],
               holdings=hands, own_weights=[0.4, 0.6], strategy=[0.2, 0.4, 0.8, 0.6])
    bp = dict(row, strategy=[0.8, 0.6, 0.2, 0.4])
    table = overfold_node(row, bp, equity=lambda h: 0.7, opponent_holdings=other,
                          opponent_weights=[0.2, 0.5, 0.3])
    assert table["fold_bp"] > table["fold_eq"]
    assert sum(g["reach_mass"] for g in table["groups"]) == pytest.approx(sum(brute))


def test_factored_locks_survive_rust_json_order_and_action_reordering():
    from types import SimpleNamespace
    from src.diagnostics.flop_check import export_policy_tables, line_key
    from src.diagnostics.flop_check_analysis import blueprint_locks
    req, _ = compile_tree(fixture_root("limped", street=Street.RIVER))
    node = next(n for n in req["nodes"] if not n["terminal"] and n["line"]
                and n["line"][-1].get("amount") == 100 and len(n["actions"]) == 4)
    code = descriptor_code(descriptor(("Ac", "Ad"), req["board"]))
    key = factored_key(node["template"], decode_descriptor(code)); p = [0.1, 0.2, 0.3, 0.4]
    policy = SimpleNamespace(abstraction=HU20_UNCAPPED_SCHEMA, raise_cap=None,
                             entries={key: (tuple(node["names"]), p)}, description={})
    tables = export_policy_tables(req, policy, {"river": {code}})
    rust_line = json.loads(json.dumps(node["line"], sort_keys=True))
    assert line_key(rust_line) == line_key(node["line"])
    row = dict(line=rust_line, board=req["board"], holdings=[["Ac", "Ad"]],
               actions=node["actions"][::-1], player=node["player"])
    result = blueprint_locks([row], req, tables)
    assert result[0]["strategy"] == pytest.approx(p[::-1])


@pytest.mark.skipif(sys.platform != "darwin", reason="Owner-approved Mac resource profile")
def test_gate_failure_stops_only_the_owned_external_process_group(tmp_path):
    from src.diagnostics.flop_check import atomic_json
    from src.diagnostics.flop_check_runtime import run_tool
    binary = tmp_path / "fixture-tool"
    binary.write_text(f"#!{sys.executable}\n" + '''import json, os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(sys.argv[1] + ".pids").write_text(json.dumps([os.getpid(), child.pid]))
with open(sys.argv[2], "w") as output:
    output.write(json.dumps({"event": "gate", "gate": "V1", "passed": False}) + "\\n")
    output.flush()
time.sleep(60)
''')
    binary.chmod(0o755); request = tmp_path / "request.json"
    atomic_json(request, {"spot": "expected-failure-control", "memory_budget_bytes": 1024**3})
    result = run_tool(binary, request, tmp_path / "run", memory_bytes=1024**3,
                      threads=1, seconds=10)
    assert result["status"] == "failure" and result["failure"] == "V1 failed"
    pids = json.loads(Path(str(request) + ".pids").read_text())
    for pid in pids:
        process = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
        assert process.returncode != 0 or process.stdout.strip().startswith("Z")
