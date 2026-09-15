"""Privileged simulation code. Policies must receive Hand.observe(), never Hand."""

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import pokers

from src.game.observation import (
    ActionTaken,
    BlindPosted,
    BoardDealt,
    Decision,
    HandFinished,
    HandStarted,
    Observation,
    ObservedHand,
    PublicEvent,
    replay,
)
from src.game.showdown import disclosures
from src.game.types import Action, ActionKind, LegalActions, Street, TableSeat

ENGINE_ACTIONS = {
    ActionKind.FOLD: pokers.ActionEnum.Fold,
    ActionKind.CHECK: pokers.ActionEnum.Check,
    ActionKind.CALL: pokers.ActionEnum.Call,
    ActionKind.RAISE: pokers.ActionEnum.Raise,
}


def card_name(card) -> str:
    return "23456789TJQKA"[int(card.rank)] + "cdhs"[int(card.suit)]


@dataclass(frozen=True, slots=True)
class Table:
    player_ids: tuple[str, ...]
    stacks: tuple[int, ...]
    button: int = 0
    small_blind: int = 50
    big_blind: int = 100
    chip_unit: str = "0.01"

    seat_numbers: tuple[int, ...] = ()
    table_seats: tuple[TableSeat, ...] = ()
    capacity: int | None = None
    session_profile: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "player_ids", tuple(self.player_ids))
        object.__setattr__(self, "stacks", tuple(self.stacks))
        n = len(self.player_ids)
        if not 2 <= n <= 10 or len(self.stacks) != n:
            raise ValueError("Expected 2–10 players with one stack per player")
        if (
            any(not isinstance(p, str) or not p for p in self.player_ids)
            or len(set(self.player_ids)) != n
        ):
            raise ValueError("Player identities must be nonempty and distinct")
        if type(self.button) is not int or not 0 <= self.button < n:
            raise ValueError("Button must name an occupied seat")
        numbers = tuple(self.seat_numbers) or tuple(range(n))
        capacity = n if self.capacity is None else self.capacity
        if (
            type(capacity) is not int
            or not n <= capacity <= 10
            or len(numbers) != n
            or len(set(numbers)) != n
            or any(type(s) is not int or not 0 <= s < capacity for s in numbers)
            or tuple(sorted(numbers)) != numbers
        ):
            raise ValueError("Physical seats must preserve clockwise hand order")
        roster = tuple(self.table_seats) or tuple(
            TableSeat(s, identity, stack, "playing")
            for s, identity, stack in zip(numbers, self.player_ids, self.stacks)
        )
        expected = tuple(
            (s, p, v) for s, p, v in zip(numbers, self.player_ids, self.stacks)
        )
        actual = tuple(
            (p.seat, p.player_id, p.stack) for p in roster if p.status == "playing"
        )
        if (
            actual != expected
            or len({p.seat for p in roster}) != len(roster)
            or len({p.player_id for p in roster}) != len(roster)
            or any(
                type(p.seat) is not int
                or not 0 <= p.seat < capacity
                or type(p.stack) is not int
                or p.stack < 0
                or not isinstance(p.player_id, str)
                or not p.player_id
                or p.status not in {"playing", "waiting", "sitting_out", "busted"}
                for p in roster
            )
        ):
            raise ValueError("Table roster does not match the hand participants")
        object.__setattr__(self, "seat_numbers", numbers)
        object.__setattr__(self, "table_seats", roster)
        object.__setattr__(self, "capacity", capacity)
        amounts = (*self.stacks, self.small_blind, self.big_blind)
        if any(type(v) is not int or not 0 < v <= 10**12 for v in amounts):
            raise ValueError("Stacks and blinds must be positive integer chips")
        if self.small_blind > self.big_blind:
            raise ValueError("Small blind cannot exceed the big blind")
        try:
            unit = Decimal(self.chip_unit)
        except (InvalidOperation, TypeError):
            raise ValueError("Invalid chip denomination") from None
        if not unit.is_finite() or unit <= 0:
            raise ValueError("Chip denomination must be finite and positive")
        object.__setattr__(self, "chip_unit", str(unit))


@dataclass(frozen=True, slots=True, eq=False)
class Hand:
    table: Table
    events: tuple[PublicEvent, ...]
    _state: pokers.State = field(repr=False)

    @classmethod
    def start(cls, table: Table, *, hand_id: str, seed: int) -> "Hand":
        state = pokers.State.from_seed(
            len(table.stacks),
            table.button,
            table.small_blind,
            table.big_blind,
            table.stacks[0],
            seed,
            chip_unit=1,
            stakes=list(table.stacks),
        )
        return cls._start(table, hand_id, state)

    @classmethod
    def from_deck(cls, table: Table, *, hand_id: str, deck: tuple[str, ...]) -> "Hand":
        cards = [pokers.Card.from_string(c[::-1].upper()) for c in deck]
        if any(c is None for c in cards):
            raise ValueError("Cards use rank/suit notation, for example Ac or Td")
        state = pokers.State.from_deck(
            len(table.stacks),
            table.button,
            table.small_blind,
            table.big_blind,
            table.stacks[0],
            cards,
            chip_unit=1,
            stakes=list(table.stacks),
        )
        return cls._start(table, hand_id, state)

    @classmethod
    def _start(cls, table: Table, hand_id: str, state) -> "Hand":
        if not isinstance(hand_id, str) or not hand_id:
            raise ValueError(
                "Provide a public hand identifier, independent of the deal seed"
            )
        n = len(table.stacks)
        sb = table.button if n == 2 else (table.button + 1) % n
        bb = (sb + 1) % n
        events = (
            HandStarted(
                hand_id,
                table.player_ids,
                table.stacks,
                table.button,
                table.small_blind,
                table.big_blind,
                table.chip_unit,
                seat_numbers=table.seat_numbers,
                table_seats=table.table_seats,
                capacity=table.capacity,
                session_profile=table.session_profile,
            ),
            BlindPosted(sb, min(table.small_blind, table.stacks[sb])),
            BlindPosted(bb, min(table.big_blind, table.stacks[bb])),
        )
        return cls(table, events, state)._publish_transition(0)

    @property
    def finished(self) -> bool:
        return self._state.final_state

    @property
    def actor(self) -> int | None:
        return None if self.finished else self._state.current_player

    def observe(
        self, seat: int, previous_hands: tuple[ObservedHand, ...] = ()
    ) -> Observation:
        if type(seat) is not int or not 0 <= seat < len(self.table.stacks):
            raise ValueError("Unknown observer seat")
        cards = tuple(card_name(c) for c in self._state.players_state[seat].hand)
        return replay(self.events, seat, cards, previous_hands)

    def observe_player(
        self, player_id: str, previous_hands: tuple[ObservedHand, ...] = ()
    ) -> Observation:
        if player_id in self.table.player_ids:
            return self.observe(self.table.player_ids.index(player_id), previous_hands)
        return replay(self.events, -1, (), previous_hands, observer_id=player_id)

    def apply(self, action: Action) -> "Hand":
        if self.finished:
            raise ValueError("Hand is already finished")
        view = self.observe(self.actor)
        view.legal_actions.validate(action)
        player = view.players[self.actor]
        paid = 0
        increment = 0
        if action.kind == ActionKind.CALL:
            paid = view.legal_actions.call_amount
        elif action.kind == ActionKind.RAISE:
            paid = action.raise_to - player.street_bet
            increment = action.raise_to - self._chips(self._state.min_bet)
        new_state = self._state.apply_action(
            pokers.Action(
                ENGINE_ACTIONS[action.kind], increment * self._state.chip_unit
            )
        )
        if new_state.status != pokers.StateStatus.Ok:
            raise RuntimeError(
                f"Engine rejected validated action {action}: {new_state.status}"
            )
        return self._after_action(action, paid, view, new_state)

    def _chips(self, amount: float) -> int:
        return round(amount / self._state.chip_unit)

    def _after_action(self, action, paid, view, new_state):
        event = ActionTaken(self.actor, view.street, action, paid)
        return Hand(self.table, self.events + (event,), new_state)._publish_transition(
            len(view.board)
        )

    def _publish_transition(self, dealt: int) -> "Hand":
        state = self._state
        events = self.events
        board = tuple(card_name(c) for c in state.public_cards)
        for count, street in ((3, Street.FLOP), (4, Street.TURN), (5, Street.RIVER)):
            if dealt < count <= len(board):
                events += (BoardDealt(street, board[dealt:count]),)
                dealt = count
        if state.final_state:
            live = [p for p in state.players_state if p.active]
            showdown = len(live) > 1
            public = replay(events, 0, ())
            if showdown:
                hands = tuple(
                    tuple(card_name(c) for c in p.hand) if p.active else None
                    for p in state.players_state
                )
                events += tuple(
                    disclosures(events, hands, board, public.pots, self.table.button)
                )
            events += (
                HandFinished(
                    tuple(self._chips(p.stake) for p in state.players_state),
                    public.pots,
                    showdown,
                ),
            )
        else:
            player = state.players_state[state.current_player]
            kinds = tuple(
                kind
                for kind, engine_kind in ENGINE_ACTIONS.items()
                if engine_kind in state.legal_actions
            )
            call = min(
                self._chips(player.stake),
                max(0, self._chips(state.min_bet - player.bet_chips)),
            )
            lower = upper = None
            if ActionKind.RAISE in kinds:
                upper = self._chips(player.bet_chips + player.stake)
                lower = min(self._chips(state.min_bet + state.min_raise), upper)
            events += (
                Decision(state.current_player, LegalActions(kinds, call, lower, upper)),
            )
        return Hand(self.table, events, state)
