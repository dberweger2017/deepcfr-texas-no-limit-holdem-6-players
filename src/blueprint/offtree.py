"""Classify observed actions against the blueprint's action menu.

This is an offline diagnostic. It replays public events at each past decision;
the actor's private cards are neither needed nor supplied.
"""

from dataclasses import dataclass

from src.blueprint.abstraction import choices
from src.game.observation import ActionTaken, Decision, Observation, replay
from src.game.types import ActionKind, Street


@dataclass(frozen=True, slots=True)
class ObservedAction:
    seat: int
    street: Street
    kind: ActionKind
    in_menu: bool
    reason: str | None


def classify_history(
    view: Observation, *, raise_cap: int = 2
) -> tuple[ObservedAction, ...]:
    """Classify each executed public action at its original decision.

    The result describes menu reachability, not whether a blueprint table key
    was trained. Those are separate causes of a uniform fallback.
    """
    events = view.history
    result = []
    for index, event in enumerate(events):
        if not isinstance(event, ActionTaken):
            continue
        if index == 0 or not isinstance(events[index - 1], Decision):
            raise ValueError("An action needs its preceding public decision")
        decision = events[index - 1]
        if decision.seat != event.seat:
            raise ValueError("Decision and executed action have different actors")
        prior = replay(events[:index], event.seat, ())
        if (
            prior.street != event.street
            or prior.legal_actions != decision.legal_actions
        ):
            raise ValueError("Public decision replay differs from executed action")
        menu = choices(prior, raise_cap=raise_cap)
        in_menu = any(item.action == event.action for item in menu)
        reason = None
        if not in_menu:
            if event.action.kind != ActionKind.RAISE:
                raise ValueError("A non-raise legal action is missing from the menu")
            raises = sum(
                isinstance(previous, ActionTaken)
                and previous.street == event.street
                and previous.action.kind == ActionKind.RAISE
                for previous in events[:index]
            )
            reason = "raise_cap" if raises >= raise_cap else "raise_size"
        result.append(
            ObservedAction(event.seat, event.street, event.action.kind, in_menu, reason)
        )
    return tuple(result)


def lookup_source(view: Observation, *, trained: bool, raise_cap: int = 2) -> str:
    """Separate absent table keys from histories the trainer cannot generate."""
    off_tree = any(
        not row.in_menu for row in classify_history(view, raise_cap=raise_cap)
    )
    if trained:
        return "trained_after_off_tree" if off_tree else "trained_in_tree"
    return "fallback_after_off_tree" if off_tree else "fallback_in_tree"
