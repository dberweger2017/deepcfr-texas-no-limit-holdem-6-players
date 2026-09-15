"""Values shared with policies. Amounts are integer multiples of chip_unit."""

from dataclasses import dataclass
from enum import Enum


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ActionKind(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    raise_to: int | None = None

    def __post_init__(self):
        if not isinstance(self.kind, ActionKind):
            raise ValueError("Expected an ActionKind")
        if self.kind == ActionKind.RAISE:
            if type(self.raise_to) is not int or self.raise_to <= 0:
                raise ValueError("raise_to must be a positive integer chip amount")
        elif self.raise_to is not None:
            raise ValueError("Only raises have a raise_to amount")


@dataclass(frozen=True, slots=True)
class LegalActions:
    kinds: tuple[ActionKind, ...] = ()
    call_amount: int = 0
    min_raise_to: int | None = None
    max_raise_to: int | None = None

    def validate(self, action: Action) -> None:
        if not isinstance(action, Action) or action.kind not in self.kinds:
            raise ValueError("Action is not available at this decision")
        if action.kind == ActionKind.RAISE:
            if not self.min_raise_to <= action.raise_to <= self.max_raise_to:
                raise ValueError("Raise-to amount is outside the legal bounds")


@dataclass(frozen=True, slots=True)
class Player:
    seat: int
    player_id: str
    starting_stack: int
    stack: int
    street_bet: int = 0
    contributed: int = 0
    folded: bool = False
    shown_cards: tuple[str, ...] = ()

    @property
    def all_in(self) -> bool:
        return not self.folded and self.stack == 0


@dataclass(frozen=True, slots=True)
class Pot:
    amount: int
    eligible_seats: tuple[int, ...]
    refund_to: int | None = None


def pots_for(players: tuple[Player, ...]) -> tuple[Pot, ...]:
    pots = []
    previous = 0
    for cap in sorted({p.contributed for p in players} - {0}):
        contributors = tuple(p for p in players if p.contributed >= cap)
        amount = (cap - previous) * len(contributors)
        eligible = tuple(p.seat for p in contributors if not p.folded)
        refund = contributors[0].seat if len(contributors) == 1 else None
        pot = Pot(amount, eligible, refund)
        if (
            pots
            and refund is None
            and pots[-1].refund_to is None
            and pots[-1].eligible_seats == eligible
        ):
            pots[-1] = Pot(pots[-1].amount + amount, eligible)
        else:
            pots.append(pot)
        previous = cap
    return tuple(pots)
