import json
from pathlib import Path

import pytest
import torch

from src.holdem.card_diversity import expanded_plan, training_targets
from src.holdem.representation_models import make_model
from src.holdem.representation_reference import (
    board_key,
    build_context,
    specifications,
)
from src.holdem.river_reference import ReferenceProfile, enumerate_reference


@pytest.fixture
def target():
    plan = json.loads(Path("configs/holdem/representation.json").read_text())
    spec = specifications(plan)[1]
    return enumerate_reference(
        build_context(spec),
        ReferenceProfile("uniform"),
        max_nodes=20000,
        deadline=float("inf"),
    ).target


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


def test_nested_target_partition_and_feature_model(target):
    record = {"target": target, "split": "train", "board_group": 0}
    fitting, test = training_targets([record], 0)
    assert fitting["train"] == [] and test == []
    model = make_model("features", 17)
    score = model([target.candidates])[0]
    assert torch.isfinite(score.regrets).all()
    assert torch.isfinite(score.values).all()
