"""Frozen post-Luna stress opponent; card rules are heuristics, not equities."""

from random import Random

from src.blueprint.abstraction import RANKS, choices
from src.game.showdown import hand_value
from src.game.types import ActionKind

VERSION = "selective-stackoff-hu20-v1"
LARGE_CALL = 800
SMALL_CALL = 200
TRAP_FREQUENCY = 0.35


def strength(view):
    """Own cards and board only; return an explicit coarse selection tier."""
    ranks = sorted((RANKS.index(c[0]) + 2 for c in view.hole_cards), reverse=True)
    high, low = ranks
    if not view.board:
        if (high == low and high >= 10) or (high, low) == (14, 13):
            return "strong"
        if high == low or high == 14 or low >= 10:
            return "medium"
        return "weak"
    value = hand_value(view.hole_cards + view.board)
    # Playing the five-card board never qualifies for selective stackoff.
    if len(view.board) == 5 and value == hand_value(view.board):
        return "board_only"
    if value[0] >= 2:
        return "strong"
    if value[0] == 1 and value[1] in ranks:
        return "medium"
    return "weak"


def small_wager(view):
    owed = view.legal_actions.call_amount
    return owed <= SMALL_CALL and 3 * owed <= view.pot


class SelectiveStackoff:
    """One uniform draw per decision, separate from deal/target randomness.

    Strong hands call large wagers. Against small wagers, or with no wager,
    they min-raise with probability .35 and otherwise call/check. Medium hands
    call only small wagers; weak hands fold when facing a bet. Never free-fold.
    No adaptation, policy queries, opponent cards or future-board access.
    """

    version = VERSION

    def __init__(self, seed):
        self.random = Random(seed)

    def choose_action(self, view):
        menu = choices(view, raise_cap=None, free_fold=False)
        by_name = {item.name: item.action for item in menu}
        draw = self.random.random()  # Also consumed for forced/check decisions.
        tier = strength(view)
        owed = view.legal_actions.call_amount
        passive = by_name.get("check", by_name.get("call"))
        if owed >= LARGE_CALL:
            action = passive if tier == "strong" else by_name["fold"]
        elif tier == "strong":
            action = (by_name["min"] if "min" in by_name and small_wager(view)
                      and draw < TRAP_FREQUENCY else passive)
        elif owed and (tier != "medium" or not small_wager(view)):
            action = by_name["fold"]
        else:
            action = passive
        view.legal_actions.validate(action)
        return action


def raise_call_amount(view, action):
    """Exact amount the HU rival could call; raise-to is not amount owed."""
    if action.kind != ActionKind.RAISE:
        return 0
    rival = view.players[1 - view.seat]
    return min(rival.stack, max(0, action.raise_to - rival.street_bet))


def is_jam(view, action):
    player = view.players[view.seat]
    return (action.kind == ActionKind.RAISE
            and action.raise_to - player.street_bet == player.stack)
