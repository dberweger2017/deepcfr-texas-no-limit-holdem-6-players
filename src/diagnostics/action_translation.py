"""Lookup-only deterministic opponent raise translation for saved HU20 policy."""

from math import log

from src.blueprint.abstraction import choices, information_key
from src.game.observation import ActionTaken, replay
from src.game.types import ActionKind


def _label(paid, pot, big_blind, all_in):
    ratio = paid / max(pot, big_blind)
    size = 0 if ratio < .5 else 1 if ratio < 1.5 else 2 if ratio < 3 else 3
    return f"raise-{size}" + ("-all-in" if all_in else "")


def translated_labels(view):
    """Map only opponent off-menu raise labels; all chip events stay untouched."""
    overrides = {}
    events = []
    for index, event in enumerate(view.history):
        if not (isinstance(event, ActionTaken) and event.seat != view.seat
                and event.action.kind == ActionKind.RAISE):
            continue
        # The history prefix ends with the public Decision event. The placeholder
        # pair is never used by choices(), which depends only on public legality.
        before = replay(view.history[:index], event.seat, ("Ac", "Kd"))
        menu = choices(before, raise_cap=None, free_fold=False)
        if any(item.action == event.action for item in menu):
            continue
        raises = [(order, item) for order, item in enumerate(menu)
                  if item.action.kind == ActionKind.RAISE]
        if not raises:
            events.append({"history_index": index, "status": "no_abstract_raise",
                           "source_paid": event.paid})
            continue
        pot = before.pot
        old_paid = event.paid
        def distance(entry):
            order, item = entry
            new_paid = item.action.raise_to - before.players[event.seat].street_bet
            return (abs(log(old_paid / max(pot, before.big_blind))
                        - log(new_paid / max(pot, before.big_blind))), order)
        _, chosen = min(raises, key=distance)
        new_paid = chosen.action.raise_to - before.players[event.seat].street_bet
        label = _label(new_paid, pot, before.big_blind,
                       new_paid == before.players[event.seat].stack)
        overrides[index] = label
        events.append({"history_index": index, "status": "mapped",
                       "source_paid": old_paid, "target_paid": new_paid,
                       "source_raise_to": event.action.raise_to,
                       "target_raise_to": chosen.action.raise_to,
                       "source_ratio": old_paid / max(pot, before.big_blind),
                       "target_ratio": new_paid / max(pot, before.big_blind),
                       "target_label": label})
    return overrides, events


class NearestOpponentRaiseLookup:
    """Use translated abstract-history labels only if they find a valid node."""

    def __init__(self, target):
        self.target = target
        self.source = target.source
        self.raise_cap = target.raise_cap
        self.visits = target.visits
        self.last_translation = None

    def distribution(self, view):
        original = self.target.distribution(view)
        overrides, events = translated_labels(view)
        self.last_translation = {"events": events, "attempted": bool(overrides),
                                 "translated_hit": False, "used_original": True}
        if not overrides:
            return original
        menu = original[0]
        key = information_key(view, menu, schema=self.source.abstraction,
                              history_label_overrides=overrides)
        saved = self.source.entries.get(key)
        if saved is None:
            self.last_translation["reason"] = "missing_translated_key"
            return original
        names, probabilities = saved
        if names != tuple(item.name for item in menu):
            self.last_translation["reason"] = "different_action_names"
            return original
        self.last_translation.update(translated_hit=True, used_original=False,
                                     key=key)
        return menu, probabilities, True
