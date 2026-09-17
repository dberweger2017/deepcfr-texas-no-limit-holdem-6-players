"""A shared suit-symmetric range for the card-representation experiment."""

from itertools import permutations
from random import Random

from src.holdem.encoding import _canonical_cards
from src.holdem.river_reference import DECK, river_context


def range_support(templates):
    support = []
    for template in templates:
        if (
            len(template) != 10
            or len(set(template)) != 10
            or not set(template) <= set(DECK)
        ):
            raise ValueError("Each joint-range template needs ten distinct cards")
        for suits in permutations("cdhs"):
            mapping = dict(zip("cdhs", suits, strict=True))
            support.append(tuple(c[0] + mapping[c[1]] for c in template))
    if not support:
        raise ValueError("The joint range is empty")
    # Multiplicity is prior mass: do not deduplicate overlapping template orbits.
    return tuple(support)


def compatible_deals(support, board, holding):
    visible = tuple(board) + tuple(holding)
    if len(visible) != 7 or len(set(visible)) != 7 or not set(visible) <= set(DECK):
        raise ValueError("Expected seven distinct visible cards")
    result = tuple(d for d in support if not set(d).intersection(visible))
    if not result:
        raise ValueError("Visible cards have zero probability under the declared range")
    return result


def board_key(board):
    # Group by the complete board, even if reveal order or suit names differ.
    return _canonical_cards((tuple(board),))


def specifications(plan):
    support = range_support(plan["range_templates"])
    seen = set()
    result = []
    for index, spec in enumerate(plan["boards"]):
        board = spec["cards"]
        if len(board) != 5 or len(set(board)) != 5 or not set(board) <= set(DECK):
            raise ValueError("Invalid reference board")
        key = board_key(board)
        if key in seen:
            raise ValueError(
                "A board or suit-equivalent board occurs in multiple groups"
            )
        seen.add(key)
        if spec["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown reference split")
        available = [c for c in DECK if c not in board]
        rng = Random(plan["context_seed"] + index)
        candidates = []
        if "Ac" in available and "Kd" in available:
            candidates.append(("Ac", "Kd"))
        for rank in dict.fromkeys(c[0] for c in board):
            cards = [c for c in available if c[0] == rank]
            if len(cards) >= 2:
                candidates.append(tuple(cards[:2]))
        for suit in "cdhs":
            cards = [c for c in available if c[1] == suit]
            candidates.append(tuple(rng.sample(cards, 2)))
        candidates.extend(tuple(rng.sample(available, 2)) for _ in range(100))
        holdings, keys = [], set()
        for holding in candidates:
            key = tuple(sorted(holding))
            if key in keys:
                continue
            try:
                deals = compatible_deals(support, board, holding)
            except ValueError:
                continue
            keys.add(key)
            holdings.append((holding, deals))
            if len(holdings) == plan["hands_per_board"]:
                break
        if len(holdings) != plan["hands_per_board"]:
            raise ValueError("Insufficient compatible hero holdings")
        for hand_index, (holding, deals) in enumerate(holdings):
            for facing in (False, True):
                result.append(
                    {
                        "name": f"board-{index}/hand-{hand_index}/{'facing' if facing else 'open'}",
                        "board_index": index,
                        "board_group": spec.get("board_group", index),
                        "split": spec["split"],
                        "board": board,
                        "holding": holding,
                        "deals": deals,
                        "facing": facing,
                    }
                )
    return tuple(result)


def build_context(spec):
    return river_context(
        spec["name"],
        spec["split"],
        spec["board"],
        spec["holding"],
        spec["deals"],
        spec["facing"],
    )
