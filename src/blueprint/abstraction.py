"""Player-visible information and legal actions for the first blueprint pilot."""

from dataclasses import dataclass
from hashlib import blake2b
from json import dumps

from src.game.observation import (
    ActionTaken,
    BlindPosted,
    BoardDealt,
    HandStarted,
    Observation,
)
from src.game.showdown import hand_value
from src.game.types import Action, ActionKind, Street

SCHEMA = "blueprint-abstraction-v1"
SUMMARY_SCHEMA = "blueprint-abstraction-summary-v1"
SUPPORTED_SCHEMAS = (SCHEMA, SUMMARY_SCHEMA)
RANKS = "23456789TJQKA"


@dataclass(frozen=True, slots=True)
class Choice:
    name: str
    action: Action


def choices(view: Observation, *, raise_cap: int = 2) -> tuple[Choice, ...]:
    """Use a small legal menu, with no speculative 100 BB open shove."""
    if view.finished or view.actor != view.seat:
        raise ValueError("An abstract action menu needs the acting player's view")
    if type(raise_cap) is not int or raise_cap < 0:
        raise ValueError("Raise cap must be a nonnegative integer")
    legal = view.legal_actions
    result = [
        Choice(kind.value, Action(kind))
        for kind in (ActionKind.FOLD, ActionKind.CHECK, ActionKind.CALL)
        if kind in legal.kinds
    ]
    street_raises = sum(
        isinstance(event, ActionTaken)
        and event.street == view.street
        and event.action.kind == ActionKind.RAISE
        for event in view.history
    )
    if ActionKind.RAISE in legal.kinds and street_raises < raise_cap:
        player = view.players[view.seat]
        matched = player.street_bet + legal.call_amount
        after_call = view.pot + legal.call_amount
        low, high = legal.min_raise_to, legal.max_raise_to
        targets = (
            ("min", low),
            ("pot", min(high, max(low, matched + after_call))),
            ("jam", high),
        )
        used = set()
        for name, target in targets:
            # The all-in is useful once its size is at most twice the pot.
            if name == "jam" and high - matched > 2 * after_call and high != low:
                continue
            if target in used:
                continue
            used.add(target)
            result.append(Choice(name, Action(ActionKind.RAISE, target)))
    for item in result:
        legal.validate(item.action)
    if not result:
        raise ValueError("The abstraction has no legal action")
    return tuple(result)


def _preflop(cards: tuple[str, str]) -> str:
    a, b = sorted(cards, key=lambda card: RANKS.index(card[0]), reverse=True)
    return a[0] + b[0] + ("p" if a[0] == b[0] else "s" if a[1] == b[1] else "o")


def _postflop(cards: tuple[str, str], board: tuple[str, ...]) -> tuple[int, ...]:
    value = hand_value(cards + board)
    rank = value[0]
    top = value[1] if len(value) > 1 else 0
    top_band = 0 if top < 8 else 1 if top < 12 else 2
    suits = [sum(card[1] == suit for card in cards + board) for suit in "cdhs"]
    flush_draw = int(max(suits) >= 4 and len(board) < 5)
    ranks = {RANKS.index(card[0]) for card in cards + board}
    windows = ({12, 0, 1, 2, 3},) + tuple(
        set(range(start, start + 5)) for start in range(9)
    )
    straight_draw = int(
        len(board) < 5 and any(len(window - ranks) == 1 for window in windows)
    )
    board_paired = int(len({card[0] for card in board}) < len(board))
    return (rank, top_band, flush_draw, straight_draw, board_paired)


def _history(view: Observation) -> tuple[tuple, ...]:
    """Keep order and actors; bucket public raise sizes without inspecting the deck."""
    pot = 0
    remaining = list(view.history[0].stacks)
    result = []
    for event in view.history:
        if isinstance(event, BlindPosted):
            pot += event.amount
            remaining[event.seat] -= event.amount
        elif isinstance(event, BoardDealt):
            result.append((event.street.value, "board"))
        elif isinstance(event, ActionTaken):
            label = event.action.kind.value
            if event.action.kind == ActionKind.RAISE:
                ratio = event.paid / max(pot, view.big_blind)
                size = 0 if ratio < 0.5 else 1 if ratio < 1.5 else 2 if ratio < 3 else 3
                label = f"raise-{size}"
                if event.paid == remaining[event.seat]:
                    label += "-all-in"
            result.append(
                (
                    event.street.value,
                    (event.seat - view.button) % len(view.players),
                    label,
                )
            )
            pot += event.paid
            remaining[event.seat] -= event.paid
    return tuple(result)


def _band(value: float, thresholds: tuple[float, ...]) -> int:
    return sum(value >= threshold for threshold in thresholds)


def _summary_history(view: Observation) -> tuple:
    """Bound long histories while keeping street, aggression and pot context."""
    pot = 0
    remaining = list(view.history[0].stacks)
    per_street: dict[Street, list[tuple]] = {}
    for event in view.history:
        if isinstance(event, BlindPosted):
            pot += event.amount
            remaining[event.seat] -= event.amount
        elif isinstance(event, ActionTaken):
            label = event.action.kind.value
            if event.action.kind == ActionKind.RAISE:
                size = _band(event.paid / max(pot, view.big_blind), (0.5, 1.5, 3.0))
                label = ("raise", size, event.paid == remaining[event.seat])
            per_street.setdefault(event.street, []).append(
                ((event.seat - view.button) % len(view.players), label)
            )
            pot += event.paid
            remaining[event.seat] -= event.paid

    street_summaries = []
    streets = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)
    for street in streets[: streets.index(view.street) + 1]:
        actions = per_street.get(street, [])
        raises = [item for item in actions if isinstance(item[1], tuple)]
        calls = sum(item[1] == "call" for item in actions)
        checks = sum(item[1] == "check" for item in actions)
        street_summaries.append(
            (
                min(len(raises), 3),
                min(calls, 3),
                min(checks, 3),
                raises[-1] if raises else None,
                tuple(actions[-2:]) if street == view.street else (),
            )
        )
    opponent_stacks = [
        amount
        for seat, amount in enumerate(remaining)
        if seat != view.seat and not view.players[seat].folded
    ]
    effective = min(remaining[view.seat], max(opponent_stacks, default=0))
    return (
        tuple(street_summaries),
        _band(pot / view.big_blind, (4, 8, 16, 32, 64, 128)),
        _band(effective / max(pot, view.big_blind), (0.5, 1, 2, 4, 8)),
    )


def information_key(
    view: Observation, menu: tuple[Choice, ...], *, schema: str = SCHEMA
) -> str:
    """Stable abstract infoset; action labels are part of the key."""
    if view.finished or view.actor != view.seat or not menu:
        raise ValueError("An infoset needs the acting player's live observation")
    if not isinstance(view.history[0], HandStarted):
        raise TypeError("Missing public hand start")
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError("Unknown blueprint abstraction schema")
    cards = (
        _preflop(view.hole_cards)
        if view.street == Street.PREFLOP
        else _postflop(view.hole_cards, view.board)
    )
    payload = (
        schema,
        len(view.players),
        (view.seat - view.button) % len(view.players),
        view.street.value,
        cards,
        tuple((p.folded, p.all_in) for p in view.players),
        _history(view) if schema == SCHEMA else _summary_history(view),
        tuple(item.name for item in menu),
    )
    return blake2b(
        dumps(payload, separators=(",", ":")).encode(), digest_size=16
    ).hexdigest()
