"""Observation snapshots and auditable tail counts for generated HU20 hands."""

import json
from collections import Counter
from dataclasses import asdict

from src.arena.schedule import digest
from src.blueprint.abstraction import _postflop, _preflop, choices, information_key
from src.diagnostics.selective_stackoff import LARGE_CALL, is_jam, raise_call_amount, strength
from src.game.observation import ActionTaken, BlindPosted, BoardDealt
from src.game.showdown import hand_value
from src.game.types import ActionKind

CATEGORIES = ("high_card", "pair", "two_pair", "trips", "straight", "flush",
              "full_house", "quads", "straight_flush")


def concrete_category(view):
    if not view.board:
        return _preflop(view.hole_cards)
    value = hand_value(view.hole_cards + view.board)
    if len(view.board) == 5 and value == hand_value(view.board):
        return "board_only"
    if value[0] != 1:
        return CATEGORIES[value[0]]
    ranks = ["23456789TJQKA".index(c[0]) + 2 for c in view.hole_cards]
    top = max("23456789TJQKA".index(c[0]) + 2 for c in view.board)
    if value[1] not in ranks:
        return "board_pair"
    return "overpair" if value[1] > top else "top_pair" if value[1] == top else "lower_pair"


def public_context(view):
    """Exclude hole cards, hand IDs, simulator seeds and unrevealed state."""
    events = []
    for event in view.history:
        if isinstance(event, (ActionTaken, BlindPosted, BoardDealt)):
            item = asdict(event)
            if "seat" in item:
                item["seat"] = (item["seat"] - view.button) % 2
            events.append({"type": type(event).__name__, **item})
    context = {"street": view.street.value, "board": list(view.board),
            "position": "button" if view.seat == view.button else "big_blind",
            "pot": view.pot, "players": [
                {k: getattr(view.players[(view.button + offset) % 2], k)
                 for k in ("starting_stack", "stack", "street_bet", "contributed", "folded")}
                for offset in (0, 1)],
            "legal": asdict(view.legal_actions), "events": events}
    return json.loads(json.dumps(context))


def snapshot(view, menu, probabilities=None, trained=None, visits=None):
    own = view.players[view.seat]
    context = public_context(view)
    large = [raise_call_amount(view, c.action) >= LARGE_CALL for c in menu]
    jams = [is_jam(view, c.action) for c in menu]
    return {"seat": view.seat, "position": context["position"],
            "street": view.street.value, "pot": view.pot,
            "call_amount": view.legal_actions.call_amount,
            "stack": own.stack, "street_bet": own.street_bet,
            "hole_cards": list(view.hole_cards), "board": list(view.board),
            "concrete_category": concrete_category(view), "selection_tier": strength(view),
            "card_bucket": _postflop(view.hole_cards, view.board) if view.board else _preflop(view.hole_cards),
            "public_context": context, "public_context_id": digest(context),
            "menu": [{"name": c.name, "kind": c.action.kind.value, "raise_to": c.action.raise_to,
                      "rival_call_amount": raise_call_amount(view, c.action), "jam": jams[i]}
                     for i, c in enumerate(menu)],
            "large_raise_opportunity": any(large), "jam_opportunity": any(jams),
            "trained": trained, "visits": visits,
            "probabilities": list(probabilities) if probabilities is not None else None,
            "large_raise_probability": sum(p for p, flag in zip(probabilities, large) if flag)
                if probabilities is not None else None,
            "jam_probability": sum(p for p, flag in zip(probabilities, jams) if flag)
                if probabilities is not None else None}


class RecordingTarget:
    def __init__(self, source, visits, decisions, guard=lambda: None):
        self.source, self.visits, self.decisions, self.guard = source, visits, decisions, guard
        self.raise_cap = source.raise_cap

    def distribution(self, view):
        self.guard()
        menu, probabilities, trained = self.source.distribution(view)
        key = information_key(view, menu, schema=self.source.abstraction)
        entry = snapshot(view, menu, probabilities, trained, self.visits.get(key, 0))
        entry.update(logical_player=0, key=key)
        self.decisions.append(entry)
        return menu, probabilities, trained


class RecordingOpponent:
    def __init__(self, source, decisions, guard=lambda: None):
        self.source, self.decisions, self.guard = source, decisions, guard

    def choose_action(self, view):
        self.guard()
        menu = choices(view, raise_cap=None, free_fold=False)
        entry = snapshot(view, menu)
        entry["logical_player"] = 1
        self.decisions.append(entry)
        return self.source.choose_action(view)


def attach_snapshots(row, decisions):
    if len(decisions) != len(row["actions"]):
        raise ValueError("Actual-action and observation counts differ")
    for action, observed in zip(row["actions"], decisions, strict=True):
        if action["seat"] != observed["seat"] or action["logical_player"] != observed["logical_player"]:
            raise ValueError("Observation belongs to a different acting player")
        action["observation"] = observed
    return row


def hand_tails(row):
    """Count actions, then partition whole-hand profit exactly once."""
    if row["status"] != "complete":
        raise ValueError("Incomplete hand cannot enter completed-hand metrics")
    counts = Counter(hands=1, target_decisions=0, large_opportunities=0, large_actions=0,
                     jam_opportunities=0, jam_actions=0, rival_folds=0, rival_continuations=0,
                     no_response=0, trained=0, fallback=0, full_stack_wins=0, full_stack_losses=0)
    first_response = None
    for i, action in enumerate(row["actions"]):
        if action["logical_player"] != 0:
            continue
        view = action["observation"]
        counts["target_decisions"] += 1
        lookup = "trained" if view["trained"] else "fallback"
        counts[lookup] += 1
        counts[f'{view["street"]}_{lookup}'] += 1
        counts["large_opportunities"] += view["large_raise_opportunity"]
        counts["jam_opportunities"] += view["jam_opportunity"]
        if action["kind"] != "raise":
            continue
        selected = next((c for c in view["menu"] if c["kind"] == "raise"
                         and c["raise_to"] == action["raise_to"]), None)
        if selected is None:
            raise ValueError("Target raise is not its concrete restricted action")
        counts["jam_actions"] += selected["jam"]
        if selected["rival_call_amount"] < LARGE_CALL:
            continue
        counts["large_actions"] += 1
        counts[f"large_{lookup}"] += 1
        response = "no_response"
        if i + 1 < len(row["actions"]):
            rival = row["actions"][i + 1]
            if rival["logical_player"] == 1 and rival["street"] == action["street"]:
                if rival["observation"]["call_amount"] != selected["rival_call_amount"]:
                    raise ValueError("Hypothetical raise call amount differs from native response")
                response = "folded" if rival["kind"] == "fold" else "continued"
        counts[{"folded": "rival_folds", "continued": "rival_continuations",
                "no_response": "no_response"}[response]] += 1
        if first_response is None:
            first_response = response
    counts["full_stack_wins"] = int(row["target_chips"] == 2000)
    counts["full_stack_losses"] = int(row["target_chips"] == -2000)
    return {"counts": dict(counts), "first_large_raise_response": first_response or "no_large_raise",
            "target_chips": row["target_chips"]}
