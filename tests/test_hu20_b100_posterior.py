"""Target-perspective posterior and outcome-blind selection boundaries."""

from random import Random

import pytest

from src.blueprint.search import DECK, _sample_world
from src.diagnostics.reverse_lbr import (
    compatible_holdings, observed_lbr_actions, reverse_lbr_posterior,
)
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


def _facing_view():
    hand = Hand.start(Table(("lbr", "target"), (2000, 2000)),
                      hand_id="posterior-visible", seed=31)
    assert hand.actor == 0
    hand = hand.apply(Action(ActionKind.RAISE, 300))
    assert hand.actor == 1
    return hand.observe(1)


def test_reverse_posterior_uses_observed_attacker_action_and_normalizes():
    view = _facing_view()
    events = observed_lbr_actions(view)
    assert len(events) == 1 and events[0][2] == Action(ActionKind.RAISE, 300)
    calls = []

    def fake(source, prefix, seat, pair, observed, seed):
        calls.append((prefix, seat, pair, observed, seed))
        return int(pair[0] < "8"), 0

    pairs, weights, info = reverse_lbr_posterior(
        view, None, root=43, samples=4, coordinate="fixed", likelihood_fn=fake)
    assert len(calls) == len(pairs) * 4
    assert all(seat == 0 and observed == Action(ActionKind.RAISE, 300)
               for _, seat, _, observed, _ in calls)
    assert sum(weights) == pytest.approx(1)
    assert info["positive_mass_holdings"] < len(pairs)
    assert info["uniform_total_variation"] > 0
    assert info["zero_likelihood_events"] == []
    assert info["likelihood_samples_per_holding_action"] == 4


def test_actual_hidden_lbr_cards_and_future_deck_cannot_change_posterior():
    view = _facing_view()
    compatible = [card for card in DECK if card not in view.hole_cards + view.board]
    a = _sample_world(view, {0: ((tuple(compatible[:2]), 1.0),)}, Random(1))
    b = _sample_world(view, {0: ((tuple(compatible[-2:]), 1.0),)}, Random(2))
    assert a.observe(1) == b.observe(1) == view

    def fake(source, prefix, seat, pair, observed, seed):
        return int(seed % 3 != 0), 0

    first = reverse_lbr_posterior(a.observe(1), None, root=711,
                                  samples=4, coordinate="public", likelihood_fn=fake)
    second = reverse_lbr_posterior(b.observe(1), None, root=711,
                                   samples=4, coordinate="public", likelihood_fn=fake)
    assert first == second


def test_zero_evidence_is_flagged_without_inventing_a_softmax():
    view = _facing_view()
    def impossible(source, prefix, seat, pair, observed, seed):
        return 0, 0

    pairs, weights, info = reverse_lbr_posterior(
        view, None, root=5, samples=2, coordinate="zero", likelihood_fn=impossible)
    assert len(pairs) == len(compatible_holdings(view))
    assert all(weight == pytest.approx(1 / len(pairs)) for weight in weights)
    assert len(info["zero_likelihood_events"]) == 1
    assert info["uniform_total_variation"] == pytest.approx(0)
