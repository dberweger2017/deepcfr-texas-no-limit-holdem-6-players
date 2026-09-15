import random
from dataclasses import fields, is_dataclass
from enum import Enum

import pytest

from src.game import Action, ActionKind
from src.game.hand import Hand, Table
from src.game.observation import (
    ActionTaken,
    BlindPosted,
    BoardDealt,
    CardsMucked,
    CardsShown,
    HandFinished,
    replay,
)
from src.game.showdown import hand_value
from src.game.types import Street

DECK = tuple(rank + suit for suit in "cdhs" for rank in "23456789TJQKA")


def table(n=6, stacks=None, button=0):
    return Table(
        tuple(f"player-{i}" for i in range(n)), stacks or (200,) * n, button, 1, 2, "1"
    )


def rigged(hands, board, stacks, button=0):
    order = [(button + offset) % len(hands) for offset in range(1, len(hands) + 1)]
    dealt = tuple(hands[i][r] for r in range(2) for i in order) + tuple(board)
    deck = dealt + tuple(c for c in DECK if c not in dealt)
    return Hand.from_deck(
        table(len(hands), tuple(stacks), button), hand_id="fixture", deck=deck
    )


def call_down(hand):
    while not hand.finished:
        view = hand.observe(hand.actor)
        kind = (
            ActionKind.CHECK
            if ActionKind.CHECK in view.legal_actions.kinds
            else ActionKind.CALL
        )
        hand = hand.apply(Action(kind))
    return hand


def public_values_only(value):
    if is_dataclass(value):
        assert type(value).__dataclass_params__.frozen
        assert not hasattr(value, "__dict__")
        for field in fields(value):
            public_values_only(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            public_values_only(item)
    else:
        assert (
            value is None or type(value) in (str, int, bool) or isinstance(value, Enum)
        ), type(value)


@pytest.mark.parametrize("n", [4, 5, 6])
def test_hidden_world_changes_leave_decisions_identical(n):
    config = table(n)
    deck = list(DECK)
    original = Hand.from_deck(config, hand_id="same-public-hand", deck=tuple(deck))
    actor = original.actor
    # Keep only the observer's two hole-card positions fixed. Everything else,
    # including other players' cards and the future board, may change.
    positions = [(actor - config.button - 1) % n + round_ * n for round_ in range(2)]
    hidden = [i for i in range(52) if i not in positions]
    rng = random.Random(81)
    for _ in range(20):
        shuffled = [deck[i] for i in hidden]
        rng.shuffle(shuffled)
        changed = deck.copy()
        for i, card in zip(hidden, shuffled):
            changed[i] = card
        alternate = Hand.from_deck(
            config, hand_id="same-public-hand", deck=tuple(changed)
        )
        assert original.observe(actor) == alternate.observe(actor)
    public_values_only(original.observe(actor))
    assert not hasattr(original.observe(actor), "_state")
    assert not hasattr(original.observe(actor), "deck")
    assert not hasattr(original.observe(actor).players[(actor + 1) % n], "hand")


def test_branching_does_not_change_live_history_or_prior_observations():
    hand = Hand.start(table(), hand_id="branches", seed=13)
    before = hand.observe(hand.actor)
    folded = hand.apply(Action(ActionKind.FOLD))
    raised = hand.apply(Action(ActionKind.RAISE, 10))
    assert hand.observe(hand.actor) == before
    assert len(hand.events) + 2 == len(folded.events) == len(raised.events)
    assert folded.events != raised.events
    assert raised.observe(raised.actor).legal_actions.min_raise_to == 18
    with pytest.raises(ValueError):
        hand.apply(Action(ActionKind.RAISE, 3))
    assert hand.observe(hand.actor) == before


def test_fold_winner_does_not_reveal_cards_or_undealt_board():
    hand = Hand.start(table(2), hand_id="fold", seed=13)
    hand = hand.apply(Action(ActionKind.FOLD))
    view = hand.observe(0)
    assert view.finished and not view.board
    assert not any(
        isinstance(e, (BoardDealt, CardsShown, CardsMucked)) for e in view.history
    )
    assert all(not p.shown_cards for p in view.players)


def test_showdown_mucks_only_hands_beaten_in_every_eligible_pot():
    hand = rigged(
        (("Ac", "Ad"), ("Kc", "Kd"), ("Qc", "Qd")),
        ("2c", "3d", "7h", "8s", "9c"),
        (50, 100, 200),
        button=2,
    )
    hand = hand.apply(Action(ActionKind.RAISE, 200))
    hand = hand.apply(Action(ActionKind.CALL))
    hand = hand.apply(Action(ActionKind.CALL))
    view = hand.observe(2)
    assert [p.stack for p in view.players] == [150, 100, 100]
    assert view.players[0].shown_cards == ("Ac", "Ad")
    assert view.players[1].shown_cards == ("Kc", "Kd")  # Wins the side pot.
    assert view.players[2].mucked and not view.players[2].shown_cards
    assert view.hole_cards == ("Qc", "Qd")  # Mucking doesn't erase the owner's memory.
    assert [e.street for e in view.history if isinstance(e, BoardDealt)] == [
        Street.FLOP,
        Street.TURN,
        Street.RIVER,
    ]


def test_last_river_aggressor_shows_before_the_caller():
    hand = rigged(
        (("Kc", "Kd"), ("Ac", "Ad")), ("2c", "3d", "7h", "8s", "9c"), (200, 200)
    )
    for _ in range(6):
        view = hand.observe(hand.actor)
        hand = hand.apply(
            Action(
                ActionKind.CHECK
                if ActionKind.CHECK in view.legal_actions.kinds
                else ActionKind.CALL
            )
        )
    assert hand.observe(hand.actor).street.value == "river"
    hand = hand.apply(Action(ActionKind.CHECK))  # BB checks; button bets.
    hand = hand.apply(Action(ActionKind.RAISE, 10))
    hand = hand.apply(Action(ActionKind.CALL))
    shown = [e.seat for e in hand.events if isinstance(e, CardsShown)]
    assert shown == [0, 1]


@pytest.mark.parametrize("n", [4, 5, 6])
def test_generated_public_replays_match_engine_accounting(n):
    for seed in range(100):
        rng = random.Random(100 * n + seed)
        stacks = tuple(rng.randint(1, 200) for _ in range(n))
        hand = Hand.start(table(n, stacks, seed % n), hand_id=f"deal-{seed}", seed=seed)
        for _ in range(500):
            for seat in range(n):
                view = hand.observe(seat)
                assert replay(view.history, seat, view.hole_cards) == view
                assert sum(p.stack + p.contributed for p in view.players) == sum(stacks)
                public_values_only(view)
                assert [p.stack for p in view.players] == [
                    int(p.stake) for p in hand._state.players_state
                ]
            if hand.finished:
                break
            legal = hand.observe(hand.actor).legal_actions
            kind = rng.choice(legal.kinds)
            amount = (
                rng.randint(legal.min_raise_to, legal.max_raise_to)
                if kind == ActionKind.RAISE
                else None
            )
            hand = hand.apply(Action(kind, amount))
        assert hand.finished
        assert isinstance(hand.events[-1], HandFinished)
        # A hand that was mucked must receive no contested payout.
        contributions = [0] * n
        for event in hand.events:
            if isinstance(event, BlindPosted):
                contributions[event.seat] += event.amount
            elif isinstance(event, ActionTaken):
                contributions[event.seat] += event.paid
        for player in hand.observe(0).players:
            if player.mucked:
                refund = sum(
                    p.amount for p in hand.events[-1].pots if p.refund_to == player.seat
                )
                assert (
                    player.stack
                    == stacks[player.seat] - contributions[player.seat] + refund
                )


def test_disclosure_evaluator_handles_wheels_and_kickers():
    assert hand_value(("Ac", "2d", "3h", "4s", "5c", "9d", "Th")) < hand_value(
        ("2c", "3d", "4h", "5s", "6c", "9s", "Td")
    )
    assert hand_value(("Ac", "Ad", "Ks", "Jh", "8c", "4d", "2s")) > hand_value(
        ("Ah", "As", "Qd", "Jh", "8s", "4h", "2c")
    )
