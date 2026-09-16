"""Legal baseline policies with distinct, deliberately simple tendencies."""

from src.arena.heuristics import STYLES, StylePolicy
from src.game.observation import Observation
from src.game.play import RandomPolicy
from src.game.types import Action, ActionKind


class CheckCall:
    def choose_action(self, view: Observation) -> Action:
        kind = (
            ActionKind.CHECK
            if ActionKind.CHECK in view.legal_actions.kinds
            else ActionKind.CALL
        )
        return Action(kind)


class Fold:
    def choose_action(self, view: Observation) -> Action:
        return Action(
            ActionKind.FOLD
            if ActionKind.FOLD in view.legal_actions.kinds
            else ActionKind.CHECK
        )


class CheckFold:
    """Never pay voluntarily, including when folding a free check is legal."""

    def choose_action(self, view: Observation) -> Action:
        return Action(
            ActionKind.CHECK
            if ActionKind.CHECK in view.legal_actions.kinds
            else ActionKind.FOLD
        )


POLICIES = {
    "random": RandomPolicy,
    "check_call": lambda seed: CheckCall(),
    "fold": lambda seed: Fold(),
    "check_fold": lambda seed: CheckFold(),
}


def make_policy(name: str, seed: int):
    if name in STYLES:
        return StylePolicy(STYLES[name], seed)
    if name not in POLICIES:
        raise ValueError(f"Unknown arena policy: {name}")
    return POLICIES[name](seed)
