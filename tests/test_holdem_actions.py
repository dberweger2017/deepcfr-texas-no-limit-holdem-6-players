import random
from dataclasses import replace

import pytest

from src.game.hand import Hand
from src.game.observation import ActionTaken
from src.game.types import Action, ActionKind, Street
from src.holdem.actions import AMOUNT_FIELDS, bet_candidates, record_execution
from src.holdem.encoding import ACTIONS
from tests.test_hand_observations import DECK, table
from tests.test_holdem_encoding import advance, change_suits, rotate


def raises(candidates):
    return tuple(a.raise_to for a in candidates.actions if a.kind == ActionKind.RAISE)


def execute_every(hand):
    view = hand.observe(hand.actor)
    candidates = bet_candidates(view)
    assert len(candidates.actions) == len(set(candidates.actions))
    assert len(candidates.features) == len(candidates.actions)
    for i, action in enumerate(candidates.actions):
        child = hand.apply(action)
        event = child.events[len(hand.events)]
        assert isinstance(event, ActionTaken)
        recorded = record_execution(candidates, i, event)
        assert recorded.event.action == action
        assert recorded.candidates.decision.source is view
    assert hand.observe(hand.actor) == view
    return candidates


@pytest.mark.parametrize("n", [4, 5, 6])
@pytest.mark.parametrize(
    "street", [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]
)
def test_each_candidate_executes_exactly_on_every_street(n, street):
    hand = advance(Hand.start(table(n), hand_id="all-streets", seed=19), street)
    candidates = execute_every(hand)
    view = candidates.decision.source
    assert view.legal_actions.min_raise_to in raises(candidates)
    assert (
        raises(candidates)[-1]
        == view.players[view.seat].street_bet + view.players[view.seat].stack
    )
    assert (
        Action(ActionKind.CHECK if street != Street.PREFLOP else ActionKind.CALL)
        in candidates.actions
    )


def test_preflop_sizes_use_exact_half_up_rounding_and_post_call_pot():
    hand = Hand.start(table(), hand_id="open", seed=10)
    candidates = execute_every(hand)
    # 1/2 blinds: matched wager 2, pot after calling 5; half-pot rounds 2.5 to 3.
    assert raises(candidates) == (4, 5, 6, 7, 8, 12, 200)
    hand = hand.apply(Action(ActionKind.RAISE, 10))
    candidates = execute_every(hand)
    assert raises(candidates) == (18, 22, 33, 56, 200)
    index = candidates.actions.index(Action(ActionKind.RAISE, 33))
    features = dict(zip(AMOUNT_FIELDS, candidates.features[index][len(ACTIONS) :]))
    assert features["paid_bb"] == 16.5
    assert features["raise_increment_bb"] == 11.5
    assert features["raise_increment_pot_after_call"] == 1
    assert features["remaining_bb"] == 83.5


def test_postflop_fraction_is_a_bet_when_nothing_to_call():
    hand = advance(Hand.start(table(), hand_id="flop", seed=20), Street.FLOP)
    candidates = execute_every(hand)
    assert candidates.decision.source.pot == 12
    assert raises(candidates) == (2, 4, 6, 9, 12, 18, 24, 198)


def test_blind_already_committed_is_included_in_raise_to_but_not_payment():
    hand = Hand.start(table(2), hand_id="small-blind", seed=20)
    candidates = execute_every(hand)
    index = candidates.actions.index(Action(ActionKind.RAISE, 6))
    row = dict(zip(AMOUNT_FIELDS, candidates.features[index][len(ACTIONS) :]))
    assert row["paid_bb"] == 2.5 and row["raise_to_bb"] == 3


def test_short_all_in_is_one_candidate_and_does_not_reopen_action():
    hand = Hand.start(table(4, (200, 200, 13, 200)), hand_id="short", seed=20)
    for action in (
        Action(ActionKind.RAISE, 10),
        Action(ActionKind.CALL),
        Action(ActionKind.CALL),
    ):
        hand = hand.apply(action)
    candidates = execute_every(hand)
    assert raises(candidates) == (13,)
    hand = hand.apply(Action(ActionKind.RAISE, 13))
    candidates = execute_every(hand)
    assert candidates.actions == (Action(ActionKind.FOLD), Action(ActionKind.CALL))


def test_cumulative_short_all_ins_reopen_the_full_raise():
    hand = Hand.start(
        table(6, (200, 200, 200, 200, 13, 18)), hand_id="cumulative", seed=20
    )
    for target in (10, 13, 18):
        hand = hand.apply(Action(ActionKind.RAISE, target))
    for _ in range(3):
        hand = hand.apply(Action(ActionKind.CALL))
    assert hand.actor == 3
    candidates = execute_every(hand)
    assert raises(candidates)[0] == 26


def test_short_call_marks_all_in_without_inventing_a_raise():
    hand = Hand.start(table(4, (5, 200, 200, 200)), hand_id="short-call", seed=20)
    hand = hand.apply(Action(ActionKind.RAISE, 10))
    candidates = execute_every(hand)
    assert not raises(candidates)
    call = candidates.actions.index(Action(ActionKind.CALL))
    row = dict(zip(AMOUNT_FIELDS, candidates.features[call][len(ACTIONS) :]))
    assert row["paid_bb"] == 2.5 and row["all_in"] == 1 and row["remaining_bb"] == 0


def test_execution_rejects_changed_amount_actor_street_and_index():
    hand = Hand.start(table(), hand_id="execution", seed=3)
    candidates = bet_candidates(hand.observe(hand.actor))
    index = candidates.actions.index(Action(ActionKind.RAISE, 6))
    event = hand.apply(candidates.actions[index]).events[len(hand.events)]
    for changed in (
        replace(event, paid=event.paid + 1),
        replace(event, seat=0),
        replace(event, street=Street.FLOP),
        replace(event, action=Action(ActionKind.RAISE, 7)),
    ):
        with pytest.raises(ValueError, match="differs"):
            record_execution(candidates, index, changed)
    for bad in (-1, len(candidates.actions), True):
        with pytest.raises(ValueError, match="index"):
            record_execution(candidates, bad, event)


@pytest.mark.parametrize("n", [4, 5, 6])
def test_candidates_do_not_depend_on_suits_hidden_deal_or_physical_seat(n):
    hand = Hand.from_deck(table(n), hand_id="hidden", deck=DECK)
    view = hand.observe(hand.actor)
    candidates = bet_candidates(view)
    changed = list(DECK)
    visible = set(view.hole_cards)
    positions = [i for i, card in enumerate(DECK) if card not in visible]
    for i, j in zip(positions, reversed(positions)):
        changed[i] = DECK[j]
    alternate = Hand.from_deck(table(n), hand_id="hidden", deck=tuple(changed))
    variants = [
        alternate.observe(alternate.actor),
        rotate(view, 2),
        change_suits(view, dict(zip("cdhs", "shdc"))),
    ]
    for variant in variants:
        result = bet_candidates(variant)
        assert result.actions == candidates.actions
        assert result.features == candidates.features
        assert result.decision == candidates.decision


def test_every_candidate_on_generated_unequal_stack_hands_is_accepted():
    rng = random.Random(381)
    decisions = 0
    for seed in range(30):
        n = 4 + seed % 3
        hand = Hand.start(
            table(n, tuple(rng.randint(3, 400) for _ in range(n))),
            hand_id=f"generated-{seed}",
            seed=seed,
        )
        while not hand.finished:
            candidates = execute_every(hand)
            decisions += 1
            # Keep enough small pots to exercise later streets and later raises.
            if rng.random() < 0.7:
                action = next(
                    a
                    for a in candidates.actions
                    if a.kind in (ActionKind.CALL, ActionKind.CHECK)
                )
            else:
                action = rng.choice(candidates.actions)
            hand = hand.apply(action)
    assert decisions > 250
