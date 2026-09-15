"""Versioned current-hand features; exact public records remain alongside them."""

from dataclasses import dataclass, field
from itertools import permutations

from src.game.observation import (
    ActionTaken,
    BlindPosted,
    BoardDealt,
    Decision,
    HandStarted,
    Observation,
    replay,
)
from src.game.types import ActionKind, Street

SCHEMA = "holdem-decision-v1"
SEATS = 6
STREETS = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)
ACTIONS = tuple(ActionKind)
EVENTS = (HandStarted, BlindPosted, Decision, ActionTaken, BoardDealt)
CONTEXT_FIELDS = (
    "pot_bb",
    "small_blind_bb",
    "call_bb",
    "call_pot",
    "min_raise_to_bb",
    "max_raise_to_bb",
    "min_raise_to_pot",
    "max_raise_to_pot",
    "capacity",
    "participants",
)
SEAT_FIELDS = (
    "exists",
    "occupied",
    "participating",
    "folded",
    "all_in",
    "waiting",
    "sitting_out",
    "busted",
    "button",
    "starting_stack_bb",
    "stack_bb",
    "street_bet_bb",
    "contributed_bb",
    "stack_pot",
    "effective_remaining_bb",
    "effective_total_bb",
)
EVENT_AMOUNTS = (
    "pot_before_bb",
    "paid_bb",
    "paid_pot",
    "raise_to_bb",
    "raise_to_pot",
    "call_bb",
    "min_raise_to_bb",
    "max_raise_to_bb",
    "call_pot",
    "min_raise_to_pot",
    "max_raise_to_pot",
    "small_blind_bb",
)
CONTEXT_SIZE = len(CONTEXT_FIELDS) + len(STREETS) + len(ACTIONS)
EVENT_SIZE = len(EVENTS) + SEATS + len(STREETS) + len(ACTIONS) + len(EVENT_AMOUNTS) + 52
POT_SIZE = 2 + 2 * SEATS


@dataclass(frozen=True, slots=True)
class DecisionInput:
    source: Observation = field(compare=False, repr=False)
    context: tuple[float, ...]
    cards: tuple[tuple[int, ...], ...]
    seats: tuple[tuple[float, ...], ...]
    pots: tuple[tuple[float, ...], ...]
    events: tuple[tuple[float, ...], ...]
    schema: str = SCHEMA


def _one_hot(value, choices) -> tuple[int, ...]:
    return tuple(int(value == choice) for choice in choices)


def _legal(kinds) -> tuple[int, ...]:
    return tuple(int(kind in kinds) for kind in ACTIONS)


def _canonical_cards(
    groups: tuple[tuple[str, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    ranks, suits = "23456789TJQKA", "cdhs"
    cards = tuple(card for group in groups for card in group)
    if any(
        len(card) != 2 or card[0] not in ranks or card[1] not in suits for card in cards
    ):
        raise ValueError("Expected standard two-character card names")
    if len(set(cards)) != len(cards):
        raise ValueError("A visible card cannot occur twice")
    # Minimize over all suit relabelings, including ties in unordered hole/flop cards.
    canonical = min(
        tuple(
            tuple(
                sorted(
                    ranks.index(card[0]) * 4 + mapping[suits.index(card[1])]
                    for card in group
                )
            )
            for group in groups
        )
        for mapping in permutations(range(4))
    )
    return tuple(tuple(int(card in group) for card in range(52)) for group in canonical)


def encode_decision(observation: Observation) -> DecisionInput:
    if not isinstance(observation, Observation):
        raise TypeError(
            "The Hold'em encoder accepts an Observation, never engine state"
        )
    view = observation
    if (
        view.finished
        or view.actor != view.seat
        or view.seat < 0
        or not view.legal_actions.kinds
    ):
        raise ValueError("Encode only the owner's live decision")
    if not 2 <= len(view.players) <= view.capacity <= SEATS:
        raise ValueError("The decision schema supports two to six physical seats")
    if len(view.hole_cards) != 2 or view.street not in STREETS:
        raise ValueError("A decision needs the owner's two cards and a betting street")
    if replay(view.history, view.seat, view.hole_cards, view.previous_hands) != view:
        raise ValueError("Observation fields disagree with their public replay")
    if view.big_blind <= 0:
        raise ValueError("Big blind must be positive")
    numbers = view.seat_numbers
    if (
        len(numbers) != len(view.players)
        or len(set(numbers)) != len(numbers)
        or any(
            type(seat) is not int or not 0 <= seat < view.capacity for seat in numbers
        )
    ):
        raise ValueError("Invalid physical-seat mapping")
    hero_physical = numbers[view.seat]

    def relative(physical):
        return (physical - hero_physical) % view.capacity

    def actor(seat):
        return _one_hot(relative(numbers[seat]), range(SEATS))

    bb, pot_scale = view.big_blind, max(view.pot, view.big_blind)
    legal = view.legal_actions
    context = (
        (
            view.pot / bb,
            view.small_blind / bb,
            legal.call_amount / bb,
            legal.call_amount / pot_scale,
            (legal.min_raise_to or 0) / bb,
            (legal.max_raise_to or 0) / bb,
            (legal.min_raise_to or 0) / pot_scale,
            (legal.max_raise_to or 0) / pot_scale,
            float(view.capacity),
            float(len(view.players)),
        )
        + _one_hot(view.street, STREETS)
        + _legal(legal.kinds)
    )

    reveals = [event for event in view.history if isinstance(event, BoardDealt)]
    expected = ((Street.FLOP, 3), (Street.TURN, 1), (Street.RIVER, 1))[
        : STREETS.index(view.street)
    ]
    if [(e.street, len(e.cards)) for e in reveals] != list(expected):
        raise ValueError(
            "Board events must preserve flop, turn and river reveal stages"
        )
    groups = (
        (view.hole_cards,)
        + tuple(e.cards for e in reveals)
        + ((),) * (3 - len(reveals))
    )
    cards = _canonical_cards(groups)

    roster = {p.seat: p for p in view.table_seats}
    participants = {numbers[p.seat]: p for p in view.players}
    hero = view.players[view.seat]
    seats = []
    for offset in range(SEATS):
        if offset >= view.capacity:
            seats.append((0.0,) * len(SEAT_FIELDS))
            continue
        physical = (hero_physical + offset) % view.capacity
        occupant, player = roster.get(physical), participants.get(physical)
        status = None if occupant is None else occupant.status
        stack = player.stack if player else (occupant.stack if occupant else 0)
        starting = player.starting_stack if player else stack
        eligible = player is not None and not player.folded
        seats.append(
            tuple(
                float(x)
                for x in (
                    1,
                    occupant is not None,
                    player is not None,
                    player.folded if player else False,
                    player.all_in if player else False,
                    status == "waiting",
                    status == "sitting_out",
                    status == "busted",
                    physical == numbers[view.button],
                    starting / bb,
                    stack / bb,
                    player.street_bet / bb if player else 0,
                    player.contributed / bb if player else 0,
                    stack / pot_scale,
                    min(hero.stack, stack) / bb if eligible else 0,
                    min(hero.stack + hero.contributed, stack + player.contributed) / bb
                    if eligible
                    else 0,
                )
            )
        )
    pots = []
    for pot in view.pots:
        eligible = {relative(numbers[seat]) for seat in pot.eligible_seats}
        refund = None if pot.refund_to is None else relative(numbers[pot.refund_to])
        pots.append(
            (1.0, pot.amount / bb)
            + tuple(int(s in eligible) for s in range(SEATS))
            + _one_hot(refund, range(SEATS))
        )
    if len(pots) > SEATS:
        raise ValueError("Too many pots for the supported table")
    pots.extend([(0.0,) * POT_SIZE] * (SEATS - len(pots)))

    history, pot_before, street = [], 0, Street.PREFLOP
    for event in view.history:
        if type(event) not in EVENTS:
            raise ValueError(
                "A live decision cannot include settlement or showdown events"
            )
        who, kinds, revealed = (0,) * SEATS, (), (0,) * 52
        paid = target = call = minimum = maximum = small_blind = 0
        if isinstance(event, HandStarted):
            who, small_blind = actor(event.button), event.small_blind
        elif isinstance(event, BlindPosted):
            who, paid = actor(event.seat), event.amount
        elif isinstance(event, Decision):
            who, kinds = actor(event.seat), event.legal_actions.kinds
            call = event.legal_actions.call_amount
            minimum = event.legal_actions.min_raise_to or 0
            maximum = event.legal_actions.max_raise_to or 0
        elif isinstance(event, ActionTaken):
            if event.street != street:
                raise ValueError("Action street disagrees with the revealed board")
            who, paid, target = (
                actor(event.seat),
                event.paid,
                event.action.raise_to or 0,
            )
            kinds = (event.action.kind,)
        elif isinstance(event, BoardDealt):
            street = event.street
            revealed = cards[STREETS.index(street)]
        scale = max(pot_before, bb)
        amounts = (
            pot_before / bb,
            paid / bb,
            paid / scale,
            target / bb,
            target / scale,
            call / bb,
            minimum / bb,
            maximum / bb,
            call / scale,
            minimum / scale,
            maximum / scale,
            small_blind / bb,
        )
        history.append(
            _one_hot(type(event), EVENTS)
            + who
            + _one_hot(street, STREETS)
            + _legal(kinds)
            + amounts
            + revealed
        )
        pot_before += paid
    return DecisionInput(
        view, context, cards, tuple(seats), tuple(pots), tuple(history)
    )
