"""A fixed, versioned action abstraction built from exact public chip amounts."""

from dataclasses import dataclass

from src.game.observation import ActionTaken, Observation
from src.game.types import Action, ActionKind, Street
from src.holdem.encoding import ACTIONS, DecisionInput, encode_decision

SCHEMA = "holdem-bets-v1"
OPEN_BB = ((2, 1), (5, 2), (3, 1), (4, 1))
POT_FRACTIONS = {
    Street.PREFLOP: ((1, 2), (1, 1), (2, 1)),
    Street.FLOP: ((1, 3), (1, 2), (3, 4), (1, 1), (3, 2), (2, 1)),
    Street.TURN: ((1, 2), (3, 4), (1, 1), (3, 2), (2, 1)),
    Street.RIVER: ((1, 2), (1, 1), (3, 2), (2, 1)),
}
AMOUNT_FIELDS = (
    "paid_bb",
    "raise_to_bb",
    "raise_increment_bb",
    "paid_pot",
    "raise_increment_pot_after_call",
    "remaining_bb",
    "all_in",
)
FEATURES = len(ACTIONS) + len(AMOUNT_FIELDS)


@dataclass(frozen=True, slots=True)
class BetCandidates:
    decision: DecisionInput
    actions: tuple[Action, ...]
    features: tuple[tuple[float, ...], ...]
    schema: str = SCHEMA


def _round_fraction(amount: int, fraction: tuple[int, int]) -> int:
    numerator, denominator = fraction
    return (2 * amount * numerator + denominator) // (2 * denominator)


def payment(view: Observation, action: Action) -> int:
    if action.kind == ActionKind.RAISE:
        return action.raise_to - view.players[view.seat].street_bet
    return view.legal_actions.call_amount if action.kind == ActionKind.CALL else 0


def bet_candidates(view: Observation) -> BetCandidates:
    decision = encode_decision(view)
    legal, player = view.legal_actions, view.players[view.seat]
    actions = [
        Action(kind)
        for kind in ACTIONS
        if kind in legal.kinds and kind != ActionKind.RAISE
    ]
    after_call = view.pot + legal.call_amount
    matched = player.street_bet + legal.call_amount
    if ActionKind.RAISE in legal.kinds:
        low, high = legal.min_raise_to, legal.max_raise_to
        targets = {low, high}
        targets.update(
            matched + _round_fraction(after_call, ratio)
            for ratio in POT_FRACTIONS[view.street]
        )
        if view.street == Street.PREFLOP and not any(
            isinstance(event, ActionTaken) and event.action.kind == ActionKind.RAISE
            for event in view.history
        ):
            targets.update(_round_fraction(view.big_blind, ratio) for ratio in OPEN_BB)
        # Several menu entries can become the same exact short all-in or minimum raise.
        actions.extend(
            Action(ActionKind.RAISE, target)
            for target in sorted({min(high, max(low, t)) for t in targets})
        )
    features = []
    for action in actions:
        legal.validate(action)
        paid = payment(view, action)
        increment = action.raise_to - matched if action.kind == ActionKind.RAISE else 0
        features.append(
            tuple(
                float(x)
                for x in (
                    *(action.kind == kind for kind in ACTIONS),
                    paid / view.big_blind,
                    (action.raise_to or 0) / view.big_blind,
                    increment / view.big_blind,
                    paid / max(view.pot, view.big_blind),
                    increment / max(after_call, view.big_blind),
                    (player.stack - paid) / view.big_blind,
                    paid == player.stack,
                )
            )
        )
    return BetCandidates(decision, tuple(actions), tuple(features))


@dataclass(frozen=True, slots=True)
class ExecutedBet:
    candidates: BetCandidates
    index: int
    event: ActionTaken


def record_execution(
    candidates: BetCandidates, index: int, event: ActionTaken
) -> ExecutedBet:
    if type(index) is not int or not 0 <= index < len(candidates.actions):
        raise ValueError("Unknown candidate index")
    view, action = candidates.decision.source, candidates.actions[index]
    if event != ActionTaken(view.seat, view.street, action, payment(view, action)):
        raise ValueError("Executed event differs from the selected candidate")
    return ExecutedBet(candidates, index, event)
