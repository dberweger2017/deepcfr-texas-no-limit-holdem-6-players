from dataclasses import FrozenInstanceError

import pytest

from src.game.observation import (
    ActionTaken,
    BlindPosted,
    BoardDealt,
    CardsShown,
    Decision,
    HandFinished,
    HandStarted,
    replay,
)
from src.game.types import Action, ActionKind, LegalActions, Street


def opening():
    return (
        HandStarted("hand-1", ("alice", "bob", "carol"), (200, 200, 200), 0, 1, 2, "1"),
        BlindPosted(1, 1),
        BlindPosted(2, 2),
        Decision(0, LegalActions(tuple(ActionKind), 2, 4, 200)),
    )


def test_replay_keeps_every_action_and_board_reveal():
    events = opening() + (
        ActionTaken(0, Street.PREFLOP, Action(ActionKind.RAISE, 10), 10),
        ActionTaken(1, Street.PREFLOP, Action(ActionKind.CALL), 9),
        ActionTaken(2, Street.PREFLOP, Action(ActionKind.CALL), 8),
        BoardDealt(Street.FLOP, ("2c", "3d", "4h")),
        Decision(
            1,
            LegalActions(
                (ActionKind.FOLD, ActionKind.CHECK, ActionKind.RAISE), 0, 2, 190
            ),
        ),
    )
    view = replay(events, 1, ("Ac", "Ad"))
    assert view.hole_cards == ("Ac", "Ad")
    assert view.board == ("2c", "3d", "4h")
    assert view.pot == 30
    assert [p.stack for p in view.players] == [190, 190, 190]
    assert [p.street_bet for p in view.players] == [0, 0, 0]
    assert [p.contributed for p in view.players] == [10, 10, 10]
    assert view.history == events
    assert view.legal_actions.min_raise_to == 2
    assert replay(events, 0, ("Ks", "Qs")).legal_actions == LegalActions()
    with pytest.raises(FrozenInstanceError):
        view.players[0].stack = 1000
    with pytest.raises(FrozenInstanceError):
        view.hole_cards = ("As", "Ah")


def test_private_history_follows_the_player_identity():
    events = opening() + (HandFinished((200, 199, 201), (), False),)
    record = replay(events, 0, ("Ac", "Ad")).record()
    next_start = HandStarted(
        "hand-2", ("bob", "alice", "carol"), (200,) * 3, 1, 1, 2, "1"
    )
    assert replay((next_start,), 1, ("Kh", "Ks"), (record,)).previous_hands == (record,)
    with pytest.raises(ValueError, match="different player"):
        replay((next_start,), 0, ("Kh", "Ks"), (record,))
    with pytest.raises(ValueError, match="completed"):
        replay(opening(), 0, ("Ac", "Ad")).record()


def test_folded_cards_cannot_enter_the_public_replay():
    events = opening() + (
        ActionTaken(0, Street.PREFLOP, Action(ActionKind.FOLD), 0),
        CardsShown(0, ("Ac", "Ad")),
    )
    with pytest.raises(ValueError, match="Folded cards"):
        replay(events, 1, ("Kh", "Ks"))


@pytest.mark.parametrize("amount", [True, 0, -1, 4.5, float("nan")])
def test_raise_targets_are_positive_integer_chips(amount):
    with pytest.raises(ValueError):
        Action(ActionKind.RAISE, amount)


def test_raise_bounds_and_short_all_in():
    legal = LegalActions(
        (ActionKind.FOLD, ActionKind.CALL, ActionKind.RAISE), 2, 4, 200
    )
    legal.validate(Action(ActionKind.RAISE, 4))
    with pytest.raises(ValueError, match="bounds"):
        legal.validate(Action(ActionKind.RAISE, 3))
    short = LegalActions(legal.kinds, 2, 3, 3)
    short.validate(Action(ActionKind.RAISE, 3))
