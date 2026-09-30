"""Independent small fixtures for selectable shared-query LBR caching."""

import pytest
import numpy as np

from src.blueprint.abstraction import choices
from src.diagnostics.cached_lbr import CachedLocalBestResponse, SharedProbabilityCache
from src.diagnostics import robustness
from src.diagnostics.robustness import LBRConfig, LocalBestResponse, posterior
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


class UniformTarget:
    def __init__(self):
        self.calls = 0

    def distribution(self, view):
        self.calls += 1
        menu = choices(view, free_fold=False)
        return menu, tuple(1 / len(menu) for _ in menu), True


def test_cache_key_is_complete_and_source_bound():
    source = UniformTarget()
    hand = Hand.start(Table(("lbr", "target"), (2000, 2000)),
                      hand_id="cache-key", seed=31)
    history = hand.observe(0).history
    pair = hand.observe(0).hole_cards
    cache = SharedProbabilityCache(source)
    first = cache.probabilities(history, 0, pair)
    assert cache.probabilities(tuple(history), 0, pair) is first
    assert source.calls == 1 and cache.telemetry()["hits"] == 1
    other_pair = ("Ac", "Ad") if pair != ("Ac", "Ad") else ("Kc", "Kd")
    cache.probabilities(history, 0, other_pair)
    assert source.calls == 2
    assert cache.telemetry()["entries"] == 2
    with pytest.raises(ValueError, match="different source"):
        CachedLocalBestResponse(UniformTarget(), 3, cache)


def test_cached_lbr_inherits_native_decision():
    assert CachedLocalBestResponse.choose_action is LocalBestResponse.choose_action
    source = UniformTarget()
    hand = Hand.start(Table(("lbr", "target"), (2000, 2000)),
                      hand_id="tie-contract", seed=21)
    view = hand.observe(hand.actor)
    native = LocalBestResponse(source, 19, LBRConfig(4, 5))
    cached = CachedLocalBestResponse(source, 19, SharedProbabilityCache(source), LBRConfig(4, 5))
    assert native.choose_action(view) == cached.choose_action(view)
    assert native.telemetry[-1]["chosen"] == cached.telemetry[-1]["chosen"]
    assert native.telemetry[-1]["samples"] == cached.telemetry[-1]["samples"]
    assert native.telemetry[-1]["values_chips"] == pytest.approx(cached.telemetry[-1]["values_chips"], abs=1e-10)
    assert native.zero_likelihood == cached.zero_likelihood
    assert tuple(native.weights) == pytest.approx(tuple(cached.weights), abs=1e-12)


def test_exact_first_index_tie_is_preserved(monkeypatch):
    source = UniformTarget()
    hand = Hand.start(Table(("lbr", "target"), (2000, 2000)),
                      hand_id="forced-first-index-tie", seed=22)
    view = hand.observe(hand.actor)
    menu = choices(view, free_fold=False)
    assert len(menu) >= 2
    monkeypatch.setattr(robustness, "checkdown_payoffs",
                        lambda _view, _action, outcomes: np.zeros(len(outcomes)))
    monkeypatch.setattr(LocalBestResponse, "_fold_probabilities",
                        lambda self, _view, opts: [np.zeros(len(self.holdings)) for _ in opts])
    native = LocalBestResponse(source, 17, LBRConfig(1, 5))
    cached = CachedLocalBestResponse(source, 17, SharedProbabilityCache(source), LBRConfig(1, 5))
    assert native.choose_action(view) == cached.choose_action(view) == menu[0].action
    assert native.telemetry[-1]["values_chips"] == [0.0] * len(menu)


def test_zero_evidence_preserves_bayes_prior_without_softmax():
    weights, zero = posterior((.25, .75), (0, 0))
    assert zero and tuple(weights) == pytest.approx((.25, .75))


def test_small_river_facing_wager_uses_identical_menu_and_action():
    source = UniformTarget()
    hand = Hand.start(Table(("target", "lbr"), (2000, 2000)),
                      hand_id="cached-river", seed=42)
    while len(hand.observe(hand.actor).board) < 5:
        view = hand.observe(hand.actor)
        hand = hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                                 else ActionKind.CALL))
    if hand.actor == 0:
        view = hand.observe(0)
        hand = hand.apply(Action(ActionKind.RAISE, view.legal_actions.min_raise_to))
    view = hand.observe(hand.actor)
    assert len(view.board) == 5
    native = LocalBestResponse(source, 703, LBRConfig(4, 5))
    cached = CachedLocalBestResponse(source, 703, SharedProbabilityCache(source), LBRConfig(4, 5))
    assert native.choose_action(view) == cached.choose_action(view)
    assert native.telemetry[-1]["requested_samples"] == cached.telemetry[-1]["requested_samples"] == 1
    assert native.telemetry[-1]["values_chips"] == pytest.approx(cached.telemetry[-1]["values_chips"], abs=1e-10)
