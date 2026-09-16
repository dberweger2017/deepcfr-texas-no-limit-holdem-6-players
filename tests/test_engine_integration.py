"""Retained engine regressions through the current integer-chip hand interface."""

import random

import pytest

from src.game import Action, ActionKind
from src.game.hand import Hand, Table
from src.game.types import Street
from tests.test_hand_observations import table


def test_minimum_raise_comes_from_the_engine():
    hand = Hand.start(table(), hand_id="minimum", seed=0)
    hand = hand.apply(Action(ActionKind.RAISE, 10))
    assert hand.observe(hand.actor).legal_actions.min_raise_to == 18
    hand = hand.apply(Action(ActionKind.RAISE, 18))
    assert hand.observe(hand.actor).legal_actions.min_raise_to == 26


def test_short_stack_can_raise_all_in_below_the_full_increment():
    hand = Hand.start(table(3, (3, 200, 200)), hand_id="short", seed=0)
    assert hand.observe(hand.actor).legal_actions.max_raise_to == 3
    hand = hand.apply(Action(ActionKind.RAISE, 3))
    assert hand.observe(0).players[0].stack == 0
    assert hand.observe(hand.actor).legal_actions.min_raise_to == 5


@pytest.mark.parametrize("players", [4, 5, 6])
def test_unequal_stack_hands_through_current_adapter(players):
    for seed in range(100):
        rng = random.Random(1000 * players + seed)
        stacks = tuple(rng.randint(1, 20000) for _ in range(players))
        config = Table(tuple(f"p{i}" for i in range(players)), stacks, seed % players)
        hand = Hand.start(config, hand_id=f"unequal-{players}-{seed}", seed=seed)
        for turn in range(500):
            if hand.finished:
                break
            legal = hand.observe(hand.actor).legal_actions
            kind = rng.choice(legal.kinds)
            target = (
                rng.choice(
                    (
                        legal.min_raise_to,
                        legal.max_raise_to,
                        rng.randint(legal.min_raise_to, legal.max_raise_to),
                    )
                )
                if kind == ActionKind.RAISE
                else None
            )
            hand = hand.apply(Action(kind, target))
            view = hand.observe(0)
            if not hand.finished:
                assert sum(p.stack + p.contributed for p in view.players) == sum(stacks)
        assert hand.finished, (players, seed, turn)
        view = hand.observe(0)
        assert view.pot == 0
        assert sum(p.stack for p in view.players) == sum(stacks)
        assert sum(p.stack - p.starting_stack for p in view.players) == 0


def test_engine_settles_all_ins_without_logging_repair():
    hand = Hand.start(table(6, button=1), hand_id="all-in-regression", seed=61)
    for _ in range(6):
        if hand.finished:
            break
        legal = hand.observe(hand.actor).legal_actions
        action = (
            Action(ActionKind.RAISE, legal.max_raise_to)
            if ActionKind.RAISE in legal.kinds
            else Action(ActionKind.CALL)
        )
        hand = hand.apply(action)
    assert hand.finished
    view = hand.observe(0)
    assert view.street == Street.SHOWDOWN
    assert view.legal_actions.kinds == ()
    assert sum(p.stack for p in view.players) == 1200
