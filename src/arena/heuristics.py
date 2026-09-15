"""Card-aware style controls. Scores are hand-selection heuristics, not equities."""

from dataclasses import dataclass
from random import Random

from src.game.observation import Observation
from src.game.showdown import hand_value
from src.game.types import Action, ActionKind


@dataclass(frozen=True, slots=True)
class Style:
    continue_at: float
    raise_at: float
    raise_frequency: float
    bluff_frequency: float
    pot_numerator: int
    pot_denominator: int = 2


STYLES = {
    "tight_passive": Style(0.53, 0.82, 0.15, 0.00, 1),
    "loose_passive": Style(0.23, 0.86, 0.10, 0.01, 1),
    "tight_aggressive": Style(0.51, 0.67, 0.85, 0.04, 2),
    "loose_aggressive": Style(0.27, 0.50, 0.85, 0.15, 2),
    "pot_pressure": Style(0.40, 0.60, 0.95, 0.20, 3),
    "train_pressure": Style(0.35, 0.58, 0.60, 0.10, 1),
}


def hand_score(view: Observation) -> float:
    ranks = sorted(
        ("23456789TJQKA".index(card[0]) + 2 for card in view.hole_cards), reverse=True
    )
    high, low = ranks
    if not view.board:
        if high == low:
            return 0.55 + high / 35
        suited = view.hole_cards[0][1] == view.hole_cards[1][1]
        return min(
            0.95, (high + low - 4) / 40 + 0.08 * suited + 0.06 * (high - low <= 2)
        )
    value = hand_value(view.hole_cards + view.board)
    score = (0.12, 0.40, 0.62, 0.74, 0.82, 0.88, 0.93, 0.97, 1.0)[value[0]]
    score += min(value[1], 14) / 140
    # A strong board shared by everyone is weak evidence for committing a stack.
    if len(view.board) == 5 and value == hand_value(view.board):
        score = 0.20
    return min(1.0, score)


def pot_raise(view: Observation, numerator: int, denominator: int) -> Action:
    legal = view.legal_actions
    after_call = view.pot + legal.call_amount
    increment = max(1, (after_call * numerator + denominator // 2) // denominator)
    wager = max(player.street_bet for player in view.players)
    target = min(legal.max_raise_to, max(legal.min_raise_to, wager + increment))
    return Action(ActionKind.RAISE, target)


class StylePolicy:
    def __init__(self, style: Style, seed: int):
        self.style = style
        self._random = Random(seed)

    def choose_action(self, view: Observation) -> Action:
        if not isinstance(view, Observation):
            raise TypeError("Policies require a player observation")
        if view.finished or view.actor != view.seat:
            raise ValueError("Policy needs its own current decision")
        legal, style = view.legal_actions, self.style
        score = hand_score(view)
        price = legal.call_amount / max(1, view.pot + legal.call_amount)
        threshold = style.continue_at + 0.35 * price
        aggressive = self._random.random()
        bluff = self._random.random()
        if ActionKind.RAISE in legal.kinds and (
            (score >= style.raise_at and aggressive < style.raise_frequency)
            or (price < 0.25 and bluff < style.bluff_frequency)
        ):
            return pot_raise(view, style.pot_numerator, style.pot_denominator)
        if ActionKind.CHECK in legal.kinds:
            return Action(ActionKind.CHECK)
        if score >= threshold and ActionKind.CALL in legal.kinds:
            return Action(ActionKind.CALL)
        return Action(ActionKind.FOLD)
