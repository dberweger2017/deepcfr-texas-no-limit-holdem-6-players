import random
from dataclasses import fields, is_dataclass, replace
from itertools import permutations

import numpy as np
import pytest
import torch

from src.game.hand import Hand, Table
from src.game.observation import (
    ActionTaken,
    BoardDealt,
    replay,
)
from src.game.types import Action, ActionKind, Street, TableSeat
from src.holdem.encoding import (
    CONTEXT_FIELDS,
    CONTEXT_SIZE,
    EVENT_AMOUNTS,
    EVENT_SIZE,
    EVENTS,
    POT_SIZE,
    SCHEMA,
    SEAT_FIELDS,
    SEATS,
    encode_decision,
)
from src.holdem.model import DecisionEncoder
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import DECK, table
from tests.test_sessions import finish
from tests.test_sessions import table as session_table


@pytest.fixture(autouse=True)
def cpu():
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(271)
        yield


def advance(hand, street):
    while hand.observe(hand.actor).street != street:
        legal = hand.observe(hand.actor).legal_actions
        hand = hand.apply(
            Action(
                ActionKind.CHECK if ActionKind.CHECK in legal.kinds else ActionKind.CALL
            )
        )
    return hand


def rotate(view, offset):
    physical = lambda seat: (seat + offset) % view.capacity
    order = sorted(
        range(len(view.players)), key=lambda i: physical(view.seat_numbers[i])
    )
    seats = {old: new for new, old in enumerate(order)}
    start = view.history[0]
    changed = [
        replace(
            start,
            player_ids=tuple(start.player_ids[i] for i in order),
            stacks=tuple(start.stacks[i] for i in order),
            button=seats[start.button],
            seat_numbers=tuple(physical(view.seat_numbers[i]) for i in order),
            table_seats=tuple(
                sorted(
                    (replace(p, seat=physical(p.seat)) for p in view.table_seats),
                    key=lambda p: p.seat,
                )
            ),
            capacity=view.capacity,
        )
    ]
    for event in view.history[1:]:
        changed.append(
            replace(event, seat=seats[event.seat]) if hasattr(event, "seat") else event
        )
    return replay(
        tuple(changed), seats[view.seat], view.hole_cards, view.previous_hands
    )


def change_suits(value, mapping):
    if is_dataclass(value):
        return replace(
            value,
            **{
                f.name: change_suits(getattr(value, f.name), mapping)
                for f in fields(value)
            },
        )
    if isinstance(value, tuple):
        return tuple(change_suits(v, mapping) for v in value)
    if (
        isinstance(value, str)
        and len(value) == 2
        and value[0] in "23456789TJQKA"
        and value[1] in "cdhs"
    ):
        return value[0] + mapping[value[1]]
    return value


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "street", [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]
)
def test_full_hand_shapes_and_every_public_event_are_retained(n, street):
    hand = advance(Hand.start(table(n), hand_id="shapes", seed=41), street)
    view = hand.observe(hand.actor)
    result = encode_decision(view)
    assert result.source is view and result.schema == SCHEMA
    assert len(result.context) == CONTEXT_SIZE
    assert np.shape(result.cards) == (4, 52)
    assert np.shape(result.seats) == (SEATS, len(SEAT_FIELDS))
    assert np.shape(result.pots) == (SEATS, POT_SIZE)
    assert np.shape(result.events) == (len(view.history), EVENT_SIZE)
    assert [sum(group) for group in result.cards] == [
        2,
        3 if street != Street.PREFLOP else 0,
        int(street in (Street.TURN, Street.RIVER)),
        int(street == Street.RIVER),
    ]
    for event, token in zip(view.history, result.events):
        assert token[: len(EVENTS)] == tuple(
            int(type(event) is kind) for kind in EVENTS
        )
    assert all(
        np.isfinite(array).all()
        for array in (
            result.context,
            result.cards,
            result.seats,
            result.pots,
            result.events,
        )
    )


@pytest.mark.parametrize("n", [4, 5, 6])
def test_rotation_and_all_suit_permutations_leave_inputs_and_model_outputs_unchanged(n):
    hand = advance(Hand.start(table(n), hand_id="symmetry", seed=9), Street.RIVER)
    view = hand.observe(hand.actor)
    original = encode_decision(view)
    with torch.random.fork_rng():
        torch.manual_seed(29)
        model = DecisionEncoder(16).eval()
    with torch.no_grad():
        expected = model([original])
        for offset in range(n):
            rotated = encode_decision(rotate(view, offset))
            assert rotated == original
            torch.testing.assert_close(model([rotated]), expected, rtol=0, atol=0)
        for order in permutations("cdhs"):
            transformed = encode_decision(change_suits(view, dict(zip("cdhs", order))))
            assert transformed == original
            torch.testing.assert_close(model([transformed]), expected, rtol=0, atol=0)


def test_unordered_hole_and_flop_cards_do_not_create_new_encodings():
    hand = advance(Hand.start(table(), hand_id="card-order", seed=8), Street.TURN)
    view = hand.observe(hand.actor)
    events = tuple(
        replace(e, cards=tuple(reversed(e.cards)))
        if isinstance(e, BoardDealt) and e.street == Street.FLOP
        else e
        for e in view.history
    )
    changed = replay(events, view.seat, tuple(reversed(view.hole_cards)))
    assert encode_decision(view) == encode_decision(changed)


@pytest.mark.parametrize("n", [4, 5, 6])
@pytest.mark.parametrize("street", [Street.PREFLOP, Street.FLOP, Street.TURN])
def test_hidden_cards_and_undealt_board_do_not_change_encoded_inputs_or_context(
    n, street
):
    observer = 3 if street == Street.PREFLOP else 1
    visible = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4}[street]
    protected = {(observer - 1) % n, (observer - 1) % n + n} | set(
        range(2 * n, 2 * n + visible)
    )
    hidden = [i for i in range(52) if i not in protected]
    changed = list(DECK)
    shuffled = [changed[i] for i in hidden]
    random.Random(19).shuffle(shuffled)
    for i, card in zip(hidden, shuffled):
        changed[i] = card
    views = [
        advance(Hand.from_deck(table(n), hand_id="same", deck=deck), street).observe(
            observer
        )
        for deck in (DECK, tuple(changed))
    ]
    inputs = [encode_decision(view) for view in views]
    assert inputs[0] == inputs[1]
    model = DecisionEncoder(16).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model([inputs[0]]), model([inputs[1]]), rtol=0, atol=0
        )


def test_event_payments_use_the_pot_before_the_action_and_keep_exact_bounds():
    hand = Hand.start(table(), hand_id="amounts", seed=3)
    hand = hand.apply(Action(ActionKind.RAISE, 10))
    view = hand.observe(hand.actor)
    result = encode_decision(view)
    index = next(
        i for i, event in enumerate(view.history) if isinstance(event, ActionTaken)
    )
    token = result.events[index]
    start = len(EVENTS) + SEATS + 4 + 4
    amounts = dict(zip(EVENT_AMOUNTS, token[start : start + len(EVENT_AMOUNTS)]))
    assert amounts["pot_before_bb"] == 1.5
    assert amounts["paid_bb"] == 5 and amounts["paid_pot"] == pytest.approx(10 / 3)
    assert amounts["raise_to_bb"] == 5
    context = dict(zip(CONTEXT_FIELDS, result.context))
    assert context["min_raise_to_bb"] == 9
    assert result.source.legal_actions.min_raise_to == 18
    assert result.source.history[index].action.raise_to == 10


def test_empty_sitting_out_folded_and_all_in_seats_are_distinct():
    roster = tuple(
        TableSeat(i, f"p{i}", stack, status)
        for i, stack, status in [
            (0, 100, "playing"),
            (1, 100, "playing"),
            (2, 2, "playing"),
            (3, 100, "playing"),
            (4, 100, "sitting_out"),
        ]
    )
    config = Table(
        ("p0", "p1", "p2", "p3"),
        (100, 100, 2, 100),
        0,
        1,
        2,
        "1",
        seat_numbers=(0, 1, 2, 3),
        table_seats=roster,
        capacity=6,
    )
    hand = Hand.start(config, hand_id="masks", seed=4)
    hand = hand.apply(Action(ActionKind.FOLD))
    view = hand.observe(hand.actor)
    result = encode_decision(view)
    rows = [dict(zip(SEAT_FIELDS, row)) for row in result.seats]
    assert rows[0]["participating"] == 1
    assert rows[2]["all_in"] == 1 and rows[2]["participating"] == 1
    assert rows[3]["folded"] == 1 and rows[3]["all_in"] == 0
    assert (
        rows[4]["sitting_out"] == 1
        and rows[4]["occupied"] == 1
        and rows[4]["participating"] == 0
    )
    assert rows[5]["exists"] == 1 and rows[5]["occupied"] == 0
    for offset in range(6):
        assert encode_decision(rotate(view, offset)) == result


def test_side_pots_and_pairwise_effective_stacks_are_preserved():
    hand = Hand.start(table(4, (20, 100, 200, 200)), hand_id="pots", seed=11)
    hand = hand.apply(Action(ActionKind.RAISE, 60))
    hand = hand.apply(Action(ActionKind.CALL))
    view = hand.observe(hand.actor)
    result = encode_decision(view)
    assert sum(row[0] for row in result.pots) == len(view.pots) > 1
    assert sum(row[1] for row in result.pots) == view.pot / view.big_blind
    hero = view.players[view.seat]
    for p in view.players:
        offset = (
            view.seat_numbers[p.seat] - view.seat_numbers[view.seat]
        ) % view.capacity
        row = dict(zip(SEAT_FIELDS, result.seats[offset]))
        assert (
            row["effective_remaining_bb"] == min(hero.stack, p.stack) / view.big_blind
        )
        assert (
            row["effective_total_bb"]
            == min(hero.stack + hero.contributed, p.stack + p.contributed)
            / view.big_blind
        )
    assert any(sum(row[2 + SEATS :]) for row in result.pots)


def test_session_lineups_and_identity_owned_records_remain_available():
    session = session_table()
    model = DecisionEncoder(16)
    for n in (6, 5, 4):
        session.start_hand(seed=n, opening_button=0 if n == 6 else None)
        view = session.observe(session.actor)
        encoded = encode_decision(view)
        assert (
            sum(row[SEAT_FIELDS.index("participating")] for row in encoded.seats) == n
        )
        assert encoded.source.previous_hands == view.previous_hands
        assert model([encoded]).shape == (1, 16)
        finish(session)
        if n > 4:
            session.leave(f"p{n - 1}")


def test_rejects_engine_state_nonacting_terminal_and_unsupported_tables():
    hand = Hand.start(table(), hand_id="reject", seed=1)
    with pytest.raises(TypeError):
        encode_decision(hand)
    with pytest.raises(TypeError):
        encode_decision(hand._state)
    with pytest.raises(ValueError):
        encode_decision(hand.observe((hand.actor + 1) % 6))
    with pytest.raises(ValueError):
        encode_decision(Hand.start(table(7), hand_id="large", seed=1).observe(3))
    view = hand.observe(hand.actor)
    with pytest.raises(ValueError, match="public replay"):
        encode_decision(replace(view, big_blind=0))
    with pytest.raises(ValueError, match="visible card"):
        encode_decision(replace(view, hole_cards=("Ac", "Ac")))
    while not hand.finished:
        hand = hand.apply(Action(ActionKind.FOLD))
    with pytest.raises(ValueError):
        encode_decision(hand.observe(0))


def test_model_ignores_padding_preserves_batch_order_and_backpropagates():
    initial = Hand.start(table(4), hand_id="batch", seed=7)
    long = advance(initial, Street.RIVER)
    middle = advance(initial, Street.FLOP)
    values = [encode_decision(h.observe(h.actor)) for h in (middle, initial, long)]
    model = DecisionEncoder(16)
    batch = model(values)
    for i, value in enumerate(values):
        torch.testing.assert_close(
            batch[i : i + 1], model([value]), rtol=1e-5, atol=1e-6
        )
    batch.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    with pytest.raises(TypeError):
        model([initial.observe(initial.actor)])
    with pytest.raises(ValueError):
        model([replace(values[0], schema="unknown")])


def test_equal_current_stacks_and_pot_keep_distinct_action_histories():
    hand = Hand.start(table(4), hand_id="history", seed=13)
    first = hand.apply(Action(ActionKind.RAISE, 4))
    for _ in range(3):
        first = first.apply(Action(ActionKind.CALL))
    second = hand.apply(Action(ActionKind.CALL)).apply(Action(ActionKind.RAISE, 4))
    for _ in range(3):
        second = second.apply(Action(ActionKind.CALL))
    views = [h.observe(h.actor) for h in (first, second)]
    assert views[0].players == views[1].players and views[0].pots == views[1].pots
    encoded = [encode_decision(view) for view in views]
    assert encoded[0].context == encoded[1].context
    assert encoded[0].cards == encoded[1].cards and encoded[0].seats == encoded[1].seats
    assert encoded[0].events != encoded[1].events
    model = DecisionEncoder(16).eval()
    with torch.no_grad():
        assert not torch.equal(model([encoded[0]]), model([encoded[1]]))


def test_board_reveal_order_survives_when_the_final_board_set_matches():
    hand = advance(Hand.from_deck(table(4), hand_id="reveals", deck=DECK), Street.RIVER)
    view = hand.observe(hand.actor)
    turn = next(
        e.cards
        for e in view.history
        if isinstance(e, BoardDealt) and e.street == Street.TURN
    )
    river = next(
        e.cards
        for e in view.history
        if isinstance(e, BoardDealt) and e.street == Street.RIVER
    )
    events = tuple(
        replace(e, cards=river if e.street == Street.TURN else turn)
        if isinstance(e, BoardDealt) and e.street in (Street.TURN, Street.RIVER)
        else e
        for e in view.history
    )
    changed = replay(events, view.seat, view.hole_cards)
    assert set(changed.board) == set(view.board)
    first, second = encode_decision(view), encode_decision(changed)
    assert first.cards[:2] == second.cards[:2]
    assert first.cards[2:] != second.cards[2:] and first.events != second.events


def test_chip_rescaling_preserves_features_and_keeps_exact_integer_records():
    hand = Hand.start(table(4), hand_id="scale", seed=5).apply(
        Action(ActionKind.RAISE, 10)
    )
    view = hand.observe(hand.actor)
    amounts = {
        "starting_stack",
        "stack",
        "street_bet",
        "contributed",
        "amount",
        "paid",
        "call_amount",
        "min_raise_to",
        "max_raise_to",
        "raise_to",
        "small_blind",
        "big_blind",
    }

    def scale(value, name=""):
        if is_dataclass(value):
            return replace(
                value,
                **{
                    f.name: scale(getattr(value, f.name), f.name) for f in fields(value)
                },
            )
        if isinstance(value, tuple):
            return tuple(
                scale(item, "stack" if name == "stacks" else "") for item in value
            )
        return value * 100 if type(value) is int and name in amounts else value

    first, second = encode_decision(view), encode_decision(scale(view))
    assert first == second
    assert (
        second.source.legal_actions.min_raise_to
        == 100 * first.source.legal_actions.min_raise_to
    )


def test_public_identifiers_are_retained_but_not_used_as_numerical_features():
    hand = Hand.start(table(4), hand_id="identities", seed=3)
    view = hand.observe(hand.actor)
    names = {p.player_id: f"another-{i}" for i, p in enumerate(view.players)}
    events = (
        replace(
            view.history[0],
            hand_id="another-hand",
            player_ids=tuple(names[p] for p in view.history[0].player_ids),
            table_seats=tuple(
                replace(p, player_id=names[p.player_id]) for p in view.table_seats
            ),
        ),
    ) + view.history[1:]
    changed = replay(events, view.seat, view.hole_cards)
    assert encode_decision(view) == encode_decision(changed)
    assert encode_decision(changed).source.player_id == names[view.player_id]


def test_long_betting_history_is_not_truncated():
    hand = Hand.start(table(4, (100_000,) * 4), hand_id="long-history", seed=3)
    for _ in range(140):
        legal = hand.observe(hand.actor).legal_actions
        hand = hand.apply(Action(ActionKind.RAISE, legal.min_raise_to))
    view = hand.observe(hand.actor)
    encoded = encode_decision(view)
    assert len(encoded.events) == len(view.history) > 256
    assert encoded.source.history == view.history
    assert torch.isfinite(DecisionEncoder(16)([encoded])).all()


def test_generated_unequal_stack_hands_encode_each_decision_without_mutation():
    rng = random.Random(817)
    count = 0
    for index in range(30):
        n = 4 + index % 3
        hand = Hand.start(
            table(n, tuple(rng.randint(5, 300) for _ in range(n))),
            hand_id=f"generated-{index}",
            seed=index,
        )
        while not hand.finished:
            view = hand.observe(hand.actor)
            before = view.history
            encoded = encode_decision(view)
            assert encoded.source.history == before
            assert len(encoded.events) == len(before)
            assert sum(p[1] for p in encoded.pots) == pytest.approx(
                view.pot / view.big_blind
            )
            kind = rng.choice(view.legal_actions.kinds)
            target = (
                rng.randint(
                    view.legal_actions.min_raise_to, view.legal_actions.max_raise_to
                )
                if kind == ActionKind.RAISE
                else None
            )
            hand = hand.apply(Action(kind, target))
            assert encoded.source.history == before
            count += 1
    assert count > 150
