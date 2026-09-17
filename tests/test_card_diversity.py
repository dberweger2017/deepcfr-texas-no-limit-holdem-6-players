import json
from pathlib import Path

from src.holdem.card_diversity import expanded_plan
from src.holdem.representation_reference import board_key


def test_fresh_nested_boards_exclude_all_prior_suit_equivalents():
    plan = json.loads(Path("configs/holdem/card-diversity.json").read_text())
    expanded = expanded_plan(plan)
    old = json.loads(Path(plan["prior_board_plan"]).read_text())
    prior = {board_key(tuple(row["cards"])) for row in old["boards"]}
    fresh = {board_key(tuple(row["cards"])) for row in expanded["boards"]}
    assert len(fresh) == 64
    assert not fresh & prior
    assert [row["split"] for row in expanded["boards"]].count("train") == 48
    assert [row["split"] for row in expanded["boards"]].count("validation") == 8
    assert [row["split"] for row in expanded["boards"]].count("test") == 8
    assert all(row["board_group"] == i for i, row in enumerate(expanded["boards"]))
