"""Two-player Kuhn and two-bet-cap Leduc, with payoffs in ante units."""

from dataclasses import dataclass, replace
from enum import Enum


class Action(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class InformationSet:
    game: str
    player: int
    card: int
    board: int | None
    history: tuple[tuple[Action, ...], ...]
    actions: tuple[Action, ...]


@dataclass(frozen=True, slots=True)
class State:
    game: str
    private: tuple[int, ...] = ()
    board: int | None = None
    history: tuple[tuple[Action, ...], ...] = ((),)
    committed: tuple[int, int] = (1, 1)
    street_bets: tuple[int, int] = (0, 0)
    terminal: bool = False
    folded: int | None = None

    @property
    def actor(self) -> int:
        if self.terminal:
            return -2
        if len(self.private) < 2 or (len(self.history) == 2 and self.board is None):
            return -1
        return len(self.history[-1]) % 2

    def rank(self, card: int) -> int:
        return card if self.game == "kuhn" else card // 2

    def chance_outcomes(self) -> tuple[tuple[int, float], ...]:
        if self.actor != -1:
            raise ValueError("Not a chance node")
        available = tuple(
            c for c in range(3 if self.game == "kuhn" else 6) if c not in self.private
        )
        return tuple((card, 1 / len(available)) for card in available)

    def deal(self, card: int) -> "State":
        if card not in dict(self.chance_outcomes()):
            raise ValueError("Card is not available")
        if len(self.private) < 2:
            return replace(self, private=self.private + (card,))
        return replace(self, board=card)

    def actions(self) -> tuple[Action, ...]:
        if self.actor < 0:
            return ()
        facing_bet = self.street_bets[self.actor] < max(self.street_bets)
        actions = (Action.FOLD, Action.CALL) if facing_bet else (Action.CHECK,)
        cap = 1 if self.game == "kuhn" else 2
        if self.history[-1].count(Action.RAISE) < cap:
            actions += (Action.RAISE,)
        return actions

    def information_set(self) -> InformationSet:
        if self.actor < 0:
            raise ValueError("An information set requires a player decision")
        return InformationSet(
            self.game,
            self.actor,
            self.rank(self.private[self.actor]),
            None if self.board is None else self.rank(self.board),
            self.history,
            self.actions(),
        )

    def play(self, action: Action) -> "State":
        if action not in self.actions():
            raise ValueError("Illegal action")
        actor = self.actor
        history = self.history[:-1] + (self.history[-1] + (action,),)
        if action == Action.FOLD:
            return replace(self, history=history, terminal=True, folded=actor)
        bets, committed = list(self.street_bets), list(self.committed)
        target = max(bets)
        if action == Action.RAISE:
            target += 1 if self.game == "kuhn" else 2 * len(self.history)
        committed[actor] += target - bets[actor]
        bets[actor] = target
        closes = action == Action.CALL or history[-1] == (Action.CHECK, Action.CHECK)
        state = replace(
            self, history=history, committed=tuple(committed), street_bets=tuple(bets)
        )
        if not closes:
            return state
        if self.game == "leduc" and len(history) == 1:
            return replace(state, history=history + ((),), street_bets=(0, 0))
        return replace(state, terminal=True)

    def returns(self) -> tuple[float, float]:
        if not self.terminal:
            raise ValueError("Payoffs require a terminal state")
        if self.folded is not None:
            winner = 1 - self.folded
        else:
            ranks = tuple(self.rank(c) for c in self.private)
            scores = tuple(
                (self.board is not None and r == self.rank(self.board), r)
                for r in ranks
            )
            if scores[0] == scores[1]:
                return (0.0, 0.0)
            winner = 0 if scores[0] > scores[1] else 1
        payoff = float(self.committed[1] if winner == 0 else -self.committed[0])
        return payoff, -payoff


def new_game(name: str) -> State:
    if name not in {"kuhn", "leduc"}:
        raise ValueError("Supported reference games are kuhn and leduc")
    return State(name)
