import json
import os
import subprocess
import sys
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
from src.solver.neural.network import deterministic_cpu
from scripts import check_card_diversity as runner


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
    with pytest.raises(ValueError, match="declared board prefix"):
        training_targets(
            [{"target": target, "split": "validation", "board_group": 0}], 1
        )
    model = make_model("features", 17)
    score = model([target.candidates])[0]
    assert torch.isfinite(score.regrets).all()
    assert torch.isfinite(score.values).all()


def test_feature_encoder_reuses_scaled_base_and_is_suit_invariant():
    scaled = make_model("scaled", 17)
    features = make_model("features", 17)
    for name, parameter in scaled.encoder.state_dict().items():
        assert torch.equal(parameter, features.encoder.base.state_dict()[name])

    plan = json.loads(Path("configs/holdem/representation.json").read_text())
    spec = specifications(plan)[1]
    mapping = dict(zip("cdhs", "shdc", strict=True))
    relabel = lambda cards: tuple(card[0] + mapping[card[1]] for card in cards)
    changed = {
        **spec,
        "board": relabel(spec["board"]),
        "holding": relabel(spec["holding"]),
        "deals": tuple(relabel(deal) for deal in spec["deals"]),
    }
    first = enumerate_reference(
        build_context(spec), ReferenceProfile("uniform"), max_nodes=20000, deadline=float("inf")
    ).target
    second = enumerate_reference(
        build_context(changed), ReferenceProfile("uniform"), max_nodes=20000, deadline=float("inf")
    ).target
    with deterministic_cpu(), torch.no_grad():
        first_score = features([first.candidates])[0]
        second_score = features([second.candidates])[0]
    assert torch.equal(first_score.regrets, second_score.regrets)
    assert torch.equal(first_score.values, second_score.values)


def test_small_runner_reloads_in_fresh_process_and_rejects_corruption(
    monkeypatch, tmp_path
):
    plan = json.loads(Path("configs/holdem/card-diversity.json").read_text())
    plan.update(
        {
            "board_counts": {"train": 2, "validation": 1, "test": 1},
            "training_board_counts": [1, 2],
            "seeds": [19],
            "fit_steps": 1,
            "batch_size": 1,
            "measure_steps": [0, 1],
            "calibration_steps": 1,
            "test_comparisons": [
                {
                    "name": "board-diversity",
                    "baseline": {"arm": "train1", "variant": "scaled"},
                    "candidate": {"arm": "train2", "variant": "scaled"},
                },
                {
                    "name": "visible-card-control",
                    "baseline": {"arm": "train2", "variant": "scaled"},
                    "candidate": {"arm": "train2", "variant": "features"},
                },
            ],
        }
    )
    materialized = expanded_plan(plan)
    specs = specifications(materialized)
    target = enumerate_reference(
        build_context(specs[0]),
        ReferenceProfile("uniform"),
        max_nodes=20000,
        deadline=float("inf"),
    ).target

    monkeypatch.setattr(
        runner,
        "reference",
        lambda spec, plan, deadline: (
            target,
            {"name": spec["name"], "worlds": 1, "profiles": []},
        ),
    )
    monkeypatch.setattr(
        runner,
        "_calibrate",
        lambda plan, specs, out: {
            "passed": True,
            "reference_seconds_with_allowance": 0.0,
            "fitting_seconds_with_allowance": 0.0,
        },
    )
    out = tmp_path / "run"
    report = runner.run(plan, out)
    assert report["status"] == "completed"
    assert len(report["test"]) == 3
    assert all(
        "test" not in curve
        for fit in report["fits"]
        for curve in fit["curves"]
    )

    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}
    command = [
        sys.executable,
        "-m",
        "scripts.check_card_diversity",
        "--out",
        str(out),
        "--verify",
    ]
    verified = subprocess.run(
        command, cwd=root, env=env, capture_output=True, text=True, check=False
    )
    assert verified.returncode == 0, verified.stderr
    model_path = out / "train1-original-19.pt"
    original = model_path.read_bytes()
    model_path.write_bytes(original + b"corrupt")
    rejected = subprocess.run(
        command, cwd=root, env=env, capture_output=True, text=True, check=False
    )
    model_path.write_bytes(original)
    assert rejected.returncode != 0
    assert "Artifact changed" in rejected.stderr
