"""Validity boundaries for the approved audit; no saved models are loaded."""

import json
from pathlib import Path
from random import Random
from types import SimpleNamespace

import pytest

from scripts import audit_hu20_posterior_v2 as runner
from scripts.evaluate_robustness import Uniform
from scripts.validate_reverse_lbr_acceleration import _PairedDeck
from src.blueprint.search import _sample_world
from src.diagnostics import posterior_worlds
from src.diagnostics.conditional_values import summarize
from src.diagnostics.posterior_audit_v2 import (
    DurableRows, atomic_json, holding_from_uniform, likelihood_seed,
    posterior_from_counts, stability_gate,
)
from src.diagnostics.reverse_lbr import compatible_holdings
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


def view():
    hand = Hand.start(Table(("attacker", "target"), (2000, 2000)), hand_id="audit-fixture", seed=31)
    return hand.apply(Action(ActionKind.RAISE, 300)).observe(1)


def estimate(counts):
    return posterior_from_counts((("Ac", "Ad"), ("Kc", "Kd")),
        [{"counts": counts, "public_event_index": 2, "limited_samples": 0}], 4)


def test_zero_evidence_is_unusable_not_preserved_prior_or_smoothed():
    result = estimate([0, 0])
    assert result["status"] == "unusable"
    assert result["weights"] == [0, 0]
    assert result["ess"] is None
    assert result["zero_evidence_events"] == [2]
    with pytest.raises(ValueError):
        holding_from_uniform(result["holdings"], result["weights"], .2)


def test_counts_are_consumed_for_every_holding_after_previous_zero():
    result = posterior_from_counts((("Ac", "Ad"), ("Kc", "Kd")), [
        {"counts": [0, 4], "public_event_index": 2, "limited_samples": 0},
        {"counts": [4, 1], "public_event_index": 4, "limited_samples": 0}], 4)
    assert result["weights"] == [0, 1]
    with pytest.raises(ValueError, match="Incomplete"):
        posterior_from_counts((("Ac", "Ad"), ("Kc", "Kd")), [
            {"counts": [4], "public_event_index": 4, "limited_samples": 0}], 4)


def test_production_estimator_does_not_skip_zero_mass_holdings(tmp_path, monkeypatch):
    visible = view()
    pairs = compatible_holdings(visible)[:2]
    monkeypatch.setattr(runner, "compatible_holdings", lambda v: pairs)
    events = runner.observed_lbr_actions(visible)
    monkeypatch.setattr(runner, "observed_lbr_actions", lambda v: (events[0], (events[0][0]+1, *events[0][1:])))
    calls = []
    def fake(source, cache, v, seed, ranked):
        calls.append(seed)
        initial = {likelihood_seed(51, "fixed", events[0][0], pairs[0], i) for i in range(4)}
        predicted = Action(ActionKind.FOLD) if seed in initial else events[0][2]
        return {"action": repr(predicted), "requested": 4, "completed": 4, "limited": False, "seconds": 0}
    monkeypatch.setattr(runner, "execution", fake)
    source = SimpleNamespace(source=SimpleNamespace(description={"weights_sha256": "fixture"}))
    cache = SimpleNamespace(telemetry=lambda: {})
    args = SimpleNamespace(root=tmp_path, source_head="fixture")
    result = runner.estimate(args, {"rank": "fixed"}, visible, source, cache, 51, 4, "fixture", lambda: None)
    assert len(calls) == 2*2*4
    assert len(set(calls)) == len(calls)
    assert result["likelihood_calls"] == 16
    assert result["weights"] == [0, 1]
    again = runner.estimate(args, {"rank": "fixed"}, visible, source, cache, 51, 4, "fixture", lambda: None)
    assert again == result and len(calls) == 16


def test_gate_reports_pairwise_and_higher_zero_support_not_just_ess():
    a, b = estimate([4, 0]), estimate([3, 1])
    higher = posterior_from_counts(a["holdings"], [{"counts": [12, 4], "public_event_index": 2,
                                                  "limited_samples": 0}], 16)
    gate = stability_gate([a, b, b], higher)
    assert not gate["passed"]
    assert gate["pairwise_four_tv"][0]["tv"] == .25
    assert gate["four_vs_higher"][0]["higher_mass_on_four_zero_support"] == .25
    assert any("zero support" in failure for failure in gate["failures"])
    assert stability_gate([b, b, b], higher)["passed"]


def test_gate_zero_evidence_and_limit_are_always_failures():
    a = estimate([4, 4])
    bad = dict(a, limited_samples=1)
    assert not stability_gate([a, bad, a], a)["passed"]
    assert not stability_gate([estimate([0, 0]), a, a], a)["passed"]


def test_value_phase_cannot_bypass_unpassed_stability(tmp_path):
    atomic_json(tmp_path / "timer" / "result.json", {"status": "passed"})
    atomic_json(tmp_path / "stability" / "result.json", {"status": "stopped"})
    with pytest.raises(runner.ScientificStop, match="stability"):
        runner.require_gates(SimpleNamespace(root=tmp_path))


def test_durable_rows_resume_skips_committed_ids_and_retains_torn_tail(tmp_path):
    journal = DurableRows(tmp_path, {"frozen": 1})
    journal.add({"id": "0", "status": "complete", "value": -9})
    journal.close()
    with journal.path.open("ab") as handle:
        handle.write(b'{"id":"uncommitted"')
    resumed = DurableRows(tmp_path, {"frozen": 1})
    assert resumed.rows["0"]["value"] == -9
    assert len(resumed.recovery) == 1
    with pytest.raises(ValueError, match="must not be rerun"):
        resumed.add({"id": "0", "value": 9})
    resumed.add({"id": "1", "status": "failed", "failure": "retained"})
    resumed.close()
    with pytest.raises(ValueError, match="identity changed"):
        DurableRows(tmp_path, {"frozen": 2})


def test_durable_committed_hash_mutation_is_rejected(tmp_path):
    journal = DurableRows(tmp_path, {"frozen": 1})
    journal.add({"id": "0", "value": 1})
    journal.close()
    journal.path.write_text(journal.path.read_text().replace('"value":1', '"value":2'))
    with pytest.raises(ValueError, match="prefix changed"):
        DurableRows(tmp_path, {"frozen": 1})


def test_split_selection_cannot_see_evaluation_and_gap_cannot_see_selection():
    p = [.25, .75]
    first = [(8, 0)]*48 + [(0, 16)]*48
    second = [(80000, -30000)]*48 + [(0, 16)]*48
    a, b = summarize(first, p), summarize(second, p)
    assert a["selected_action_index"] == b["selected_action_index"] == 0
    assert a["evaluation_policy_gap_bb"] == b["evaluation_policy_gap_bb"] == -12
    assert a["evaluation_policy_gap_95_interval_bb"] == b["evaluation_policy_gap_95_interval_bb"]
    assert a["descriptive_action_mean_bb"] != b["descriptive_action_mean_bb"]


class PassiveAttacker:
    def __init__(self, *args):
        self.telemetry = []
    def choose_action(self, v):
        self.telemetry.append({"completed": True, "requested_samples": 4, "samples": 4})
        return Action(ActionKind.CALL if v.legal_actions.call_amount else ActionKind.CHECK)


def test_changed_played_hidden_cards_and_future_deck_cannot_affect_conditional_values(monkeypatch):
    monkeypatch.setattr(posterior_worlds, "RankedCachedLocalBestResponse", PassiveAttacker)
    visible = view()
    pairs = compatible_holdings(visible)
    a = _sample_world(visible, {0: ((pairs[0], 1.0),)}, Random(1))
    b = _sample_world(visible, {0: ((pairs[-1], 1.0),)}, Random(999))
    assert a.observe(1) == b.observe(1) == visible
    weights = [1/len(pairs)]*len(pairs)
    first = posterior_worlds.conditional_row(a.observe(1), Uniform(), None, pairs, weights, "fixed", 3)
    second = posterior_worlds.conditional_row(b.observe(1), Uniform(), None, pairs, weights, "fixed", 3)
    assert first == second
    assert sum(first["probabilities"]) == pytest.approx(1)


def test_global_suit_coupling_preserves_sampled_world_returns(monkeypatch):
    monkeypatch.setattr(posterior_worlds, "RankedCachedLocalBestResponse", PassiveAttacker)
    visible = view()
    pairs = compatible_holdings(visible)
    weights = [1/len(pairs)]*len(pairs)
    base = posterior_worlds.conditional_row(visible, Uniform(), None, pairs, weights, "fixed", 4)
    permuted = tuple(tuple(c[0]+runner.SUITS[c[1]] for c in pair) for pair in pairs)
    with _PairedDeck(True):
        control = posterior_worlds.conditional_row(runner.suit_view(visible), Uniform(), None,
                                                   permuted, weights, "fixed", 4)
    assert base == control


def test_likelihood_roots_and_sample_ids_are_disjoint_reproducible():
    assert likelihood_seed(202610050126, "fixed", 3, ("Ac", "Ad"), 0) == likelihood_seed(202610050126, "fixed", 3, ("Ac", "Ad"), 0)
    assert len({likelihood_seed(root, "fixed", 3, ("Ac", "Ad"), i)
                for root in range(202610050126, 202610050131) for i in range(16)}) == 80
