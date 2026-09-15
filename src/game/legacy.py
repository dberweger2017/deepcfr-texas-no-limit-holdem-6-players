"""Keep existing model features while moving policy calls behind observations.

Only the host uses TrackedState. LegacyView contains public values and the
observer's own cards, without a reference to the simulator or a transition API.
"""

from dataclasses import dataclass
from decimal import Decimal
from uuid import uuid4

import pokers

from src.game.hand import ENGINE_ACTIONS, Hand, Table
from src.game.observation import ActionTaken, BoardDealt, Observation
from src.game.play import PlayerHistory
from src.game.types import Action, ActionKind, Street


@dataclass(frozen=True, slots=True)
class Card:
    rank: int
    suit: int


def cards(names):
    return tuple(Card("23456789TJQKA".index(c[0]), "cdhs".index(c[1])) for c in names)


@dataclass(frozen=True, slots=True)
class PlayerView:
    player: int
    hand: tuple[Card, ...]
    stake: float
    bet_chips: float
    pot_chips: float
    reward: float
    active: bool


@dataclass(frozen=True, slots=True)
class ActionView:
    action: pokers.ActionEnum
    amount: float


@dataclass(frozen=True, slots=True)
class ActionRecord:
    player: int
    stage: pokers.Stage
    action: ActionView


@dataclass(frozen=True, slots=True)
class LegacyView:
    observation: Observation
    players_state: tuple[PlayerView, ...]
    public_cards: tuple[Card, ...]
    current_player: int | None
    button: int
    stage: pokers.Stage
    pot: float
    min_bet: float
    min_raise: float
    bb: float
    chip_unit: float
    legal_actions: tuple[pokers.ActionEnum, ...]
    from_action: ActionRecord | None
    final_state: bool


STAGES = dict(
    zip(
        Street,
        (
            pokers.Stage.Preflop,
            pokers.Stage.Flop,
            pokers.Stage.Turn,
            pokers.Stage.River,
            pokers.Stage.Showdown,
        ),
    )
)


def legacy_view(observation: Observation) -> LegacyView:
    unit = float(Decimal(observation.chip_unit))
    players = tuple(
        PlayerView(
            p.seat,
            cards(
                observation.hole_cards if p.seat == observation.seat else p.shown_cards
            ),
            p.stack * unit,
            p.street_bet * unit,
            (p.contributed - p.street_bet) * unit,
            (p.stack - p.starting_stack) * unit if observation.finished else 0.0,
            not p.folded,
        )
        for p in observation.players
    )
    last = None
    wager = observation.big_blind
    full_raise = observation.big_blind
    for event in observation.history:
        if isinstance(event, BoardDealt):
            wager, full_raise = 0, observation.big_blind
        elif isinstance(event, ActionTaken):
            increment = 0
            if event.action.kind == ActionKind.RAISE:
                increment = event.action.raise_to - wager
                full_raise = max(full_raise, increment)
                wager = event.action.raise_to
            last = ActionRecord(
                event.seat,
                STAGES[event.street],
                ActionView(ENGINE_ACTIONS[event.action.kind], increment * unit),
            )
    if observation.actor is not None:
        able = sum(not p.folded and p.stack > 0 for p in observation.players)
        if able <= 1:
            wager = max(p.street_bet for p in observation.players if not p.folded)
    return LegacyView(
        observation,
        players,
        cards(observation.board),
        observation.actor,
        observation.button,
        STAGES[observation.street],
        observation.pot * unit,
        wager * unit,
        full_raise * unit,
        observation.big_blind * unit,
        unit,
        tuple(ENGINE_ACTIONS[k] for k in observation.legal_actions.kinds),
        last,
        observation.finished,
    )


def require_policy_view(view, *, decision=False):
    if not isinstance(view, LegacyView):
        raise TypeError(
            "Policy calls require a player observation, not simulator state"
        )

    if decision and (
        view.final_state or view.observation.actor != view.observation.seat
    ):
        raise ValueError("Policy needs its own current decision")


class TrackedState:
    """Host adapter for existing training, evaluation, and UI loops."""

    def __init__(self, hand: Hand, previous_hands=None):
        self._hand = hand
        self._previous_hands = previous_hands or ((),) * len(hand.table.stacks)

    def __getattr__(self, name):
        return getattr(self._hand._state, name)

    def observe(self, seat=None):
        seat = self.current_player if seat is None else seat
        return legacy_view(self._hand.observe(seat, self._previous_hands[seat]))

    def completed_histories(self):
        if not self.final_state or self.status != pokers.StateStatus.Ok:
            raise ValueError("Only successfully completed hands become player history")
        return {
            identity: PlayerHistory(identity, self._previous_hands[seat]).append(
                self._hand.observe(seat)
            )
            for seat, identity in enumerate(self._hand.table.player_ids)
        }

    @classmethod
    def from_seed(cls, n_players, button, sb, bb, stake, seed, **kwargs):
        return cls._create(
            pokers.State.from_seed, n_players, button, sb, bb, stake, seed, **kwargs
        )

    @classmethod
    def from_deck(cls, n_players, button, sb, bb, stake, deck, **kwargs):
        return cls._create(
            pokers.State.from_deck, n_players, button, sb, bb, stake, deck, **kwargs
        )

    @classmethod
    def _create(
        cls,
        factory,
        n_players,
        button,
        sb,
        bb,
        stake,
        deal,
        *,
        hand_id=None,
        player_ids=None,
        histories=None,
        **kwargs,
    ):
        raw = factory(n_players, button, sb, bb, stake, deal, **kwargs)
        unit = raw.chip_unit
        amounts = kwargs.get("stakes") or [stake] * n_players
        table = Table(
            tuple(player_ids)
            if player_ids is not None
            else tuple(f"seat-{i}" for i in range(n_players)),
            tuple(round(v / unit) for v in amounts),
            button,
            round(sb / unit),
            round(bb / unit),
            str(unit),
        )
        histories = histories or {}
        prior = []
        for identity in table.player_ids:
            history = histories.get(identity, PlayerHistory(identity))
            if history.player_id != identity:
                raise ValueError("History identity does not match its owner")
            prior.append(history.hands)
        return cls(
            Hand._start(table, str(uuid4()) if hand_id is None else hand_id, raw),
            tuple(prior),
        )

    def apply_action(self, action):
        raw = self._hand._state
        new = raw.apply_action(action)
        if new.status != pokers.StateStatus.Ok:
            return new
        if raw.final_state:
            return self
        kind = next(
            kind for kind, value in ENGINE_ACTIONS.items() if value == action.action
        )
        target = (
            round((raw.min_bet + action.amount) / raw.chip_unit)
            if kind == ActionKind.RAISE
            else None
        )
        public_action = Action(kind, target)
        view = self._hand.observe(raw.current_player)
        view.legal_actions.validate(public_action)
        paid = (
            target - view.players[view.seat].street_bet
            if target is not None
            else view.legal_actions.call_amount
            if kind == ActionKind.CALL
            else 0
        )
        return TrackedState(
            self._hand._after_action(public_action, paid, view, new),
            self._previous_hands,
        )
