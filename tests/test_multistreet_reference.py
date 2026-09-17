import json
from pathlib import Path

import pytest
import torch

from src.holdem.multistreet_models import VARIANTS, make_model
from src.holdem.multistreet_reference import (
    build_context,
    enumerate_reference,
    flop_key,
    range_support,
)
from src.holdem.river_reference import ReferenceProfile, enumerate_reference as old_enumerate, river_context
from scripts import check_multistreet_representation as runner
from src.holdem.visible_features import PARTIAL_FEATURE_SIZE, partial_visible_features


@pytest.fixture
def support():
    plan = json.loads(Path("configs/holdem/representation.json").read_text())
    return range_support(plan["range_templates"])


def test_worlds_share_only_the_visible_observation_and_sample_future_after_hole_cards(support):
    context = build_context(
        name="smoke",
        split="train",
        street="flop",
        board=("Ac", "Kd", "7h"),
        holding=("Qs", "Jc"),
        support=support,
        samples=2,
        deals_per_sample=1,
        seed=17,
    )
    assert context.worlds[0].observe(context.hero_seat) == context.worlds[1].observe(context.hero_seat)
    assert context.worlds[0].events[0].hand_id == context.worlds[1].events[0].hand_id
    visible = set(("Ac", "Kd", "7h", "Qs", "Jc"))
    for assignment, world in zip(context.assignments, context.worlds, strict=True):
        hidden = {
            card
            for seat, hand in enumerate(assignment)
            if seat != context.hero_seat
            for card in hand
        }
        assert not visible.intersection(hidden)
        assert set(world.observe(context.hero_seat).board) == visible - {"Qs", "Jc"}

    changed_runout = build_context(
        name="smoke",
        split="train",
        street="flop",
        board=("Ac", "Kd", "7h"),
        holding=("Qs", "Jc"),
        support=support,
        samples=2,
        deals_per_sample=1,
        seed=99,
    )
    assert context.worlds[0].observe(context.hero_seat) == changed_runout.worlds[0].observe(
        changed_runout.hero_seat
    )


@pytest.mark.parametrize(
    ("street", "board"),
    [
        ("flop", ("Ac", "Kd", "7h")),
        ("turn", ("Ac", "Kd", "7h", "4s")),
        ("river", ("Ac", "Kd", "7h", "4s", "2c")),
    ],
)
def test_paired_reference_values_are_finite_for_all_streets(support, street, board):
    context = build_context(
        name=f"{street}/open",
        split="train",
        street=street,
        board=board,
        holding=("Qs", "Jc"),
        support=support,
        samples=2,
        deals_per_sample=1,
        seed=19,
    )
    reference = enumerate_reference(
        context, ReferenceProfile("uniform"), max_nodes=100_000, deadline=float("inf")
    )
    assert reference.uncertainty_status == "estimated"
    assert len(reference.world_action_values_bb) == 2
    assert all(torch.isfinite(torch.tensor(reference.target.values_bb)))
    assert all(value >= 0 for value in reference.action_standard_error_bb)


def test_one_world_is_marked_insufficient_for_uncertainty(support):
    context = build_context(
        name="one",
        split="train",
        street="river",
        board=("Ac", "Kd", "7h", "4s", "2c"),
        holding=("Qs", "Jc"),
        support=support,
        samples=1,
        deals_per_sample=1,
        seed=23,
    )
    reference = enumerate_reference(
        context, ReferenceProfile("uniform"), max_nodes=100_000, deadline=float("inf")
    )
    assert reference.uncertainty_status == "insufficient_worlds"


def test_partial_features_and_all_arms_accept_flop_decisions(support):
    context = build_context(
        name="model",
        split="train",
        street="flop",
        board=("Ac", "Kd", "7h"),
        holding=("Qs", "Jc"),
        support=support,
        samples=2,
        deals_per_sample=1,
        seed=29,
    )
    target = enumerate_reference(
        context, ReferenceProfile("uniform"), max_nodes=100_000, deadline=float("inf")
    ).target
    assert len(partial_visible_features(("Qs", "Jc"), ("Ac", "Kd", "7h"))) == PARTIAL_FEATURE_SIZE
    for variant in VARIANTS:
        output = make_model(variant, 31)([target.candidates])[0]
        assert torch.isfinite(output.regrets).all()
        assert torch.isfinite(output.values).all()


def test_flop_ancestor_key_ignores_suit_names_but_not_ranks():
    assert flop_key(("Ac", "Kd", "7h")) == flop_key(("As", "Kc", "7d"))
    assert flop_key(("Ac", "Kd", "7h")) != flop_key(("Ac", "Qd", "7h"))


def test_partial_features_are_suit_relabel_invariant():
    board = ("Ac", "Kd", "7h")
    holding = ("Qs", "Jc")
    mapping = dict(zip("cdhs", "shdc", strict=True))
    relabel = lambda cards: tuple(card[0] + mapping[card[1]] for card in cards)
    assert partial_visible_features(holding, board) == pytest.approx(
        partial_visible_features(relabel(holding), relabel(board))
    )


def test_river_builder_matches_legacy_reference_for_one_complete_world():
    board = ("Ac", "Kd", "7h", "4s", "2c")
    holding = ("Qs", "Jc")
    deal = ("Th", "Ts", "9h", "9s", "8h", "8s", "7c", "7d", "6h", "6s")
    legacy = river_context("same", "train", board, holding, (deal,), False)
    modern = build_context(
        name="same",
        split="train",
        street="river",
        board=board,
        holding=holding,
        support=(deal,),
        samples=2,
        deals_per_sample=1,
        seed=1,
    )
    old_target = old_enumerate(
        legacy, ReferenceProfile("uniform"), max_nodes=100_000, deadline=float("inf")
    ).target
    new_target = enumerate_reference(
        modern, ReferenceProfile("uniform"), max_nodes=100_000, deadline=float("inf")
    ).target
    assert new_target.values_bb == pytest.approx(old_target.values_bb)
    assert new_target.regrets_bb == pytest.approx(old_target.regrets_bb)


def test_small_runner_freezes_duration_before_validation_and_reloads(tmp_path, support, monkeypatch):
    context = build_context(
        name="pilot",
        split="train",
        street="river",
        board=("Ac", "Kd", "7h", "4s", "2c"),
        holding=("Qs", "Jc"),
        support=support,
        samples=2,
        deals_per_sample=1,
        seed=41,
        facing=True,
    )
    reference = enumerate_reference(
        context, ReferenceProfile("uniform"), max_nodes=100_000, deadline=float("inf")
    )
    rows = [
        {
            "name": f"{split}-pilot",
            "split": split,
            "street": "river",
            "situation": "facing",
            "group": split,
            "context": context,
            "target": reference.target,
            "worlds": context.worlds,
            "uncertainty": [reference.uncertainty_status],
            "action_se": [reference.action_standard_error_bb],
            "world_action_values_bb": reference.world_action_values_bb,
        }
        for split in ("train", "tuning", "validation", "test")
    ]
    monkeypatch.setattr(runner, "_reference_rows", lambda plan, deadline: rows)
    plan = json.loads(Path("configs/holdem/multistreet-representation.json").read_text())
    plan.update({"seeds": [11], "durations": [1, 2], "max_fit_seconds": 300})
    out = tmp_path / "run"
    report = runner.run(plan, out)
    assert report["status"] == "completed"
    assert all(set(fit["metrics"]) == {"train", "tuning"} for fit in report["fits"])
    assert {row["duration"] for row in report["validation"]} == set(
        report["duration_selection"]["durations"].values()
    )
    assert all(row["variant"] == plan["variants"][0] for row in report["test"])
    assert runner.verify(out)["status"] == "verified"
    model_path = out / f"{plan['variants'][0]}-11-1.pt"
    original = model_path.read_bytes()
    model_path.write_bytes(original + b"corrupt")
    with pytest.raises(ValueError, match="Artifact changed"):
        runner.verify(out)
    model_path.write_bytes(original)
