"""Cheap card-sensitivity and coverage probe on one fixed public decision."""

from dataclasses import replace

from src.blueprint.abstraction import RANKS
from src.game.hand import Hand, Table


def preflop_first_action(blueprint, table: Table) -> dict:
    """Probe all 169 canonical classes without giving the policy a hidden deal."""
    hand = Hand.start(table, hand_id="blueprint-card-probe", seed=0)
    view = hand.observe(hand.actor)
    policies = {}
    hits = 0
    for high_index, high in enumerate(RANKS):
        for low in RANKS[: high_index + 1]:
            variants = (
                (((high + "c", high + "d"), "p"),)
                if high == low
                else (
                    ((high + "c", low + "c"), "s"),
                    ((high + "c", low + "d"), "o"),
                )
            )
            for cards, suffix in variants:
                menu, probabilities, found = blueprint.distribution(
                    replace(view, hole_cards=cards)
                )
                policies[high + low + suffix] = {
                    "probabilities": dict(
                        zip((item.name for item in menu), probabilities)
                    ),
                    "trained_infoset": found,
                }
                hits += found
    if len(policies) != 169:
        raise AssertionError("The probe must cover exactly 169 hand classes")
    unique = {tuple(sorted(row["probabilities"].items())) for row in policies.values()}
    return {
        "decision": "preflop-first-to-act",
        "hand_classes": 169,
        "trained_infosets": hits,
        "unseen_infosets": 169 - hits,
        "distinct_distributions": len(unique),
        "AA": policies["AAp"],
        "72o": policies["72o"],
        "strength_claim": False,
    }
