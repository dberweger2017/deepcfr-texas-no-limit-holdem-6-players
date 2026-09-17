"""Fresh nested board sets for the card-diversity diagnostic."""

import json
from pathlib import Path
from random import Random

from src.holdem.representation_reference import board_key, specifications
from src.holdem.river_reference import DECK


def expanded_plan(plan, *, base_dir=Path(".")):
    """Materialize the declared fresh boards and attach their group indices."""

    source = base_dir / plan["prior_board_plan"]
    prior = json.loads(source.read_text())
    forbidden = {board_key(tuple(board["cards"])) for board in prior["boards"]}
    counts = plan["board_counts"]
    total = counts["train"] + counts["validation"] + counts["test"]
    random = Random(plan["fresh_board_seed"])
    boards, keys = [], set(forbidden)
    while len(boards) < total:
        candidate = tuple(random.sample(DECK, 5))
        key = board_key(candidate)
        if key in keys:
            continue
        keys.add(key)
        index = len(boards)
        split = (
            "train"
            if index < counts["train"]
            else "validation"
            if index < counts["train"] + counts["validation"]
            else "test"
        )
        boards.append({"split": split, "cards": list(candidate), "board_group": index})
    materialized = {**plan, "boards": boards}
    # Validate every board's compatible holdings before any reference work starts.
    specs = specifications(materialized)
    if len({board_key(s["board"]) for s in specs}) != total:
        raise ValueError("Fresh board groups are not unique")
    for spec in specs:
        if spec["board_index"] != spec["board_group"]:
            raise ValueError("Board group metadata disagrees with board order")
    return materialized


def training_targets(records, board_count):
    """Return the nested training prefix and the shared held-out contexts."""

    train = [r["target"] for r in records if r["board_group"] < board_count]
    validation = [r["target"] for r in records if r["split"] == "validation"]
    test = [r["target"] for r in records if r["split"] == "test"]
    if len(train) != board_count * 2 * 6:
        raise ValueError("Training arm does not contain the declared board prefix")
    return {"train": train, "validation": validation}, test
