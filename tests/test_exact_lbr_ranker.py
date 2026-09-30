"""Independent rank boundaries and unchanged LBR clock/RNG semantics."""

from types import FunctionType

import numpy as np
import pytest

from src.blueprint.abstraction import choices
from src.diagnostics.cached_lbr import CachedLocalBestResponse, SharedProbabilityCache
from src.diagnostics.exact_ranker import (
    RankedCachedLocalBestResponse, _bind_choose_action, exact_seven_card,
)
from src.diagnostics.ranker_fixtures import FIXTURES
from src.diagnostics.robustness import LBRConfig, LocalBestResponse
from src.game.hand import Hand, Table
from src.game.showdown import hand_value
from src.game.types import Action, ActionKind


class UniformTarget:
    def distribution(self, view):
        menu = choices(view, free_fold=False)
        return menu, tuple(1 / len(menu) for _ in menu), True


@pytest.mark.parametrize("cards,expected", FIXTURES)
def test_explicit_category_and_kicker_boundaries(cards, expected):
    cards = tuple(cards.split())
    assert hand_value(cards) == expected
    assert exact_seven_card(cards) == expected


def test_board_playing_tie_and_unsupported_arities():
    board = ("Ac", "Kc", "Qc", "Jc", "Tc")
    assert exact_seven_card(("2d", "3h") + board) == exact_seven_card(("As", "Ad") + board) == (8, 14)
    for cards in (board, board + ("2d",)):
        assert exact_seven_card(cards) == hand_value(cards)


def test_executor_reuses_native_code_and_private_globals():
    candidate = RankedCachedLocalBestResponse.choose_action
    assert candidate.__code__ is LocalBestResponse.choose_action.__code__
    differing = [key for key in candidate.__globals__
                 if candidate.__globals__[key] is not LocalBestResponse.choose_action.__globals__[key]]
    assert differing == ["hand_value"]
    assert candidate.__globals__["hand_value"] is exact_seven_card
    assert LocalBestResponse.choose_action.__globals__["hand_value"] is hand_value


@pytest.mark.parametrize("times,completed", [((0, 6, 6, 7), 1), ((0, .1, 4.9, 5.1, 5.2), 2)])
def test_timer_commits_whole_batches_and_same_rng(times, completed):
    source = UniformTarget()
    view = Hand.start(Table(("lbr", "target"), (2000, 2000)),
                      hand_id="ranker-timer", seed=31).observe(0)
    results = []
    for ranker in (hand_value, exact_seven_card):
        clock = iter(times)
        ranked_calls = []
        def counted(cards):
            ranked_calls.append(cards)
            return ranker(cards)
        class Timed(CachedLocalBestResponse):
            choose_action = _bind_choose_action(counted, clock=lambda: next(clock))
        attacker = Timed(source, 79, SharedProbabilityCache(source), LBRConfig(4, 5))
        action = attacker.choose_action(view)
        row = attacker.telemetry[-1]
        assert row["samples"] == completed and row["requested_samples"] == 4
        assert not row["completed"]
        assert len(ranked_calls) == completed * 2 * row["positive_range_holdings"]
        results.append((action, row["values_chips"], attacker.random.getstate(), ranked_calls))
    assert results[0] == results[1]


def test_exact_tie_and_zero_evidence_are_preserved():
    source = UniformTarget()
    view = Hand.start(Table(("lbr", "target"), (2000, 2000)),
                      hand_id="ranker-tie", seed=21).observe(0)
    menu = choices(view, free_fold=False)
    for ranker in (hand_value, exact_seven_card):
        fn = _bind_choose_action(ranker, clock=lambda: 0)
        namespace = dict(fn.__globals__, checkdown_payoffs=lambda v, a, x: np.zeros(len(x)))
        class Tied(CachedLocalBestResponse):
            choose_action = FunctionType(fn.__code__, namespace)
            def _fold_probabilities(self, view, opts):
                return [np.zeros(len(self.holdings)) for _ in opts]
        attacker = Tied(source, 17, SharedProbabilityCache(source), LBRConfig(1, 5))
        assert attacker.choose_action(view) == menu[0].action
    class FirstOnly:
        def distribution(self, view):
            menu = choices(view, free_fold=False)
            return menu, (1.0,) + (0.0,) * (len(menu) - 1), True
    source = FirstOnly()
    hand = Hand.start(Table(("lbr", "target"), (2000, 2000)), hand_id="ranker-zero", seed=91)
    hand = hand.apply(Action(ActionKind.CALL))
    hand = hand.apply(Action(ActionKind.RAISE, hand.observe(1).legal_actions.min_raise_to))
    outputs = []
    for cls in (CachedLocalBestResponse, RankedCachedLocalBestResponse):
        attacker = cls(source, 93, SharedProbabilityCache(source), LBRConfig(1, 5))
        action = attacker.choose_action(hand.observe(0))
        assert len(attacker.zero_likelihood) == 1
        outputs.append((action, attacker.zero_likelihood, attacker.weights.tolist(),
                        attacker.telemetry[-1]["values_chips"], attacker.random.getstate()))
    assert outputs[0] == outputs[1]
