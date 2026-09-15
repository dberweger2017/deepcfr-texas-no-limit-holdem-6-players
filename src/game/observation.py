"""Public hand events and their player-specific replay."""

from dataclasses import dataclass, replace

from src.game.types import (
    Action,
    ActionKind,
    LegalActions,
    Player,
    Pot,
    Street,
    pots_for,
)


@dataclass(frozen=True, slots=True)
class HandStarted:
    hand_id: str
    player_ids: tuple[str, ...]
    stacks: tuple[int, ...]
    button: int
    small_blind: int
    big_blind: int
    chip_unit: str


@dataclass(frozen=True, slots=True)
class BlindPosted:
    seat: int
    amount: int


@dataclass(frozen=True, slots=True)
class Decision:
    seat: int
    legal_actions: LegalActions


@dataclass(frozen=True, slots=True)
class ActionTaken:
    seat: int
    street: Street
    action: Action
    paid: int


@dataclass(frozen=True, slots=True)
class BoardDealt:
    street: Street
    cards: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CardsShown:
    seat: int
    cards: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CardsMucked:
    seat: int


@dataclass(frozen=True, slots=True)
class HandFinished:
    stacks: tuple[int, ...]
    pots: tuple[Pot, ...]
    showdown: bool


PublicEvent = (
    HandStarted
    | BlindPosted
    | Decision
    | ActionTaken
    | BoardDealt
    | CardsShown
    | CardsMucked
    | HandFinished
)


@dataclass(frozen=True, slots=True)
class ObservedHand:
    """A seat owner's record, suitable for carrying into a later hand."""

    player_id: str
    seat: int
    hole_cards: tuple[str, ...]
    events: tuple[PublicEvent, ...]


@dataclass(frozen=True, slots=True)
class Observation:
    hand_id: str
    player_id: str
    seat: int
    hole_cards: tuple[str, ...]
    board: tuple[str, ...]
    street: Street
    players: tuple[Player, ...]
    button: int
    small_blind: int
    big_blind: int
    chip_unit: str
    actor: int | None
    legal_actions: LegalActions
    pots: tuple[Pot, ...]
    history: tuple[PublicEvent, ...]
    previous_hands: tuple[ObservedHand, ...]
    finished: bool

    @property
    def pot(self) -> int:
        return sum(p.amount for p in self.pots)

    def record(self) -> ObservedHand:
        if not self.finished:
            raise ValueError("Only completed hands can enter persistent history")
        return ObservedHand(self.player_id, self.seat, self.hole_cards, self.history)


def replay(
    events: tuple[PublicEvent, ...],
    seat: int,
    hole_cards: tuple[str, ...],
    previous_hands: tuple[ObservedHand, ...] = (),
) -> Observation:
    if not events or not isinstance(events[0], HandStarted):
        raise ValueError("A replay must start with HandStarted")
    start = events[0]
    if type(seat) is not int or not 0 <= seat < len(start.stacks):
        raise ValueError("Unknown observer seat")
    if any(hand.player_id != start.player_ids[seat] for hand in previous_hands):
        raise ValueError("Prior private history belongs to a different player")
    players = tuple(
        Player(i, identity, stack, stack)
        for i, (identity, stack) in enumerate(zip(start.player_ids, start.stacks))
    )
    board = ()
    street = Street.PREFLOP
    actor = None
    legal = LegalActions()
    finished = False
    for event in events[1:]:
        if finished:
            raise ValueError("Events cannot follow settlement")
        if isinstance(event, (BlindPosted, ActionTaken)):
            player = players[event.seat]
            paid = event.amount if isinstance(event, BlindPosted) else event.paid
            if not 0 <= paid <= player.stack:
                raise ValueError("Event overdraws a stack")
            folded = player.folded or (
                isinstance(event, ActionTaken) and event.action.kind == ActionKind.FOLD
            )
            updated = replace(
                player,
                stack=player.stack - paid,
                street_bet=player.street_bet + paid,
                contributed=player.contributed + paid,
                folded=folded,
            )
            players = players[: event.seat] + (updated,) + players[event.seat + 1 :]
            actor, legal = None, LegalActions()
        elif isinstance(event, Decision):
            actor, legal = event.seat, event.legal_actions
        elif isinstance(event, BoardDealt):
            street = event.street
            board += event.cards
            players = tuple(replace(p, street_bet=0) for p in players)
            actor, legal = None, LegalActions()
        elif isinstance(event, CardsShown):
            player = players[event.seat]
            if player.folded or player.mucked:
                raise ValueError("Folded cards and mucked cards must stay private")
            players = (
                players[: event.seat]
                + (replace(player, shown_cards=event.cards),)
                + players[event.seat + 1 :]
            )
        elif isinstance(event, CardsMucked):
            player = players[event.seat]
            if player.folded or player.shown_cards:
                raise ValueError("Only an unshown live hand can be mucked")
            players = (
                players[: event.seat]
                + (replace(player, mucked=True),)
                + players[event.seat + 1 :]
            )
        elif isinstance(event, HandFinished):
            if (
                len(event.stacks) != len(players)
                or any(type(v) is not int or v < 0 for v in event.stacks)
                or sum(event.stacks) != sum(start.stacks)
            ):
                raise ValueError("Settlement does not conserve chips")
            players = tuple(
                replace(p, stack=stack, street_bet=0, contributed=0)
                for p, stack in zip(players, event.stacks)
            )
            if event.showdown:
                street = Street.SHOWDOWN
            actor, legal, finished = None, LegalActions(), True
        else:
            raise TypeError(f"Unexpected event: {type(event).__name__}")
    return Observation(
        start.hand_id,
        start.player_ids[seat],
        seat,
        tuple(hole_cards),
        board,
        street,
        players,
        start.button,
        start.small_blind,
        start.big_blind,
        start.chip_unit,
        actor,
        legal if actor == seat else LegalActions(),
        pots_for(players),
        tuple(events),
        tuple(previous_hands),
        finished,
    )
