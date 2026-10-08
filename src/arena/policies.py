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


class NativeHU100Uniform:
    """Uniform in the exact uncapped HU100 menu, with the existing weighted sampler."""

    def __init__(self, seed: int):
        from random import Random
        self._random = Random(seed)

    def choose_action(self, view: Observation) -> Action:
        from src.blueprint.abstraction import choices
        start = view.history[0]
        if (view.capacity != 2 or start.stacks != (10000, 10000)
                or view.small_blind != 50 or view.big_blind != 100 or view.chip_unit != '0.01'):
            raise ValueError("Uniform HU100 reference requires fixed 100-BB heads-up")
        menu = choices(view, raise_cap=None, free_fold=False)
        return self._random.choices(menu, weights=(1 / len(menu),) * len(menu), k=1)[0].action


POLICIES = {
    "random": RandomPolicy,
    "check_call": lambda seed: CheckCall(),
    "fold": lambda seed: Fold(),
    "check_fold": lambda seed: CheckFold(),
    "native_hu100_uniform": NativeHU100Uniform,
}


def make_policy(name: str, seed: int):
    if name in STYLES:
        return StylePolicy(STYLES[name], seed)
    if name not in POLICIES:
        raise ValueError(f"Unknown arena policy: {name}")
    return POLICIES[name](seed)
