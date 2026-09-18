"""Reproducible board diversity for the multi-street diagnostic."""

import json
from collections import Counter
from pathlib import Path
from random import Random

from src.holdem.card_diversity import expanded_plan
from src.holdem.multistreet_reference import flop_key
from src.holdem.representation_reference import specifications
from src.holdem.river_reference import DECK

SPLIT_COUNTS = {"train": 24, "tuning": 8, "validation": 8, "test": 8}


def excluded_flops(paths, *, base_dir=Path(".")):
    keys = set()
    for path in paths:
        prior = json.loads((base_dir / path).read_text())
        if "fresh_board_seed" in prior and "board_counts" in prior:
            prior = expanded_plan(prior, base_dir=base_dir)
        boards = [row["cards"] for row in prior.get("boards", ())]
        boards.extend(row["board"] for row in prior.get("contexts", ()))
        boards.extend(row["flop"] for row in prior.get("families", ()))
        keys.update(flop_key(board) for board in boards)
    return keys


def generate_families(plan, *, base_dir=Path(".")):
    """Draw compatible families, then shuffle their split assignment.

    Holdings follow the earlier diagnostic's fixed AK/pair/suited/random
    candidate rule. Compatibility is checked against the full board; earlier
    street worlds still condition their opponents only on currently visible
    cards when the reference builder runs.
    """

    rng = Random(plan["family_seed"])
    seen = excluded_flops(plan["forbidden_flop_plans"], base_dir=base_dir)
    families = []
    for attempt in range(100_000):
        board = rng.sample(DECK, 5)
        key = flop_key(board)
        if key in seen:
            continue
        try:
            rows = specifications(
                {
                    "boards": [{"split": "train", "cards": board}],
                    "range_templates": plan["range_templates"],
                    "context_seed": plan["family_seed"] + attempt,
                    "hands_per_board": 4,
                }
            )
        except ValueError as error:
            if str(error) != "Insufficient compatible hero holdings":
                raise
            continue
        seen.add(key)
        families.append(
            {
                "flop": board[:3],
                "continuation": board[3:],
                "holdings": [list(row["holding"]) for row in rows if not row["facing"]],
            }
        )
        if len(families) == sum(SPLIT_COUNTS.values()):
            break
    else:
        raise ValueError("Could not draw enough compatible fresh board families")
    rng.shuffle(families)
    split_labels = [split for split, count in SPLIT_COUNTS.items() for _ in range(count)]
    return [dict(family, split=split) for family, split in zip(families, split_labels, strict=True)]


def family_summary(families):
    """Describe visible diversity without inspecting any reference outcomes."""

    result = {}
    for split in SPLIT_COUNTS:
        rows = [row for row in families if row["split"] == split]
        suit_counts = Counter(len({card[1] for card in row["flop"]}) for row in rows)
        rank_counts = Counter(len({card[0] for card in row["flop"]}) for row in rows)
        result[split] = {
            "families": len(rows),
            "flop_suits": {str(k): v for k, v in sorted(suit_counts.items())},
            "flop_distinct_ranks": {str(k): v for k, v in sorted(rank_counts.items())},
            "distinct_holdings": len({tuple(sorted(hand)) for row in rows for hand in row["holdings"]}),
        }
    return result
