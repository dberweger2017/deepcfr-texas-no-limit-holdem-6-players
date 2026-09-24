"""Behavioral checks for the first tabular Hold'em blueprint path."""

from dataclasses import asdict, replace

import pytest

from src.arena.catalog import Checkpoint
from src.blueprint.abstraction import choices, information_key
from src.blueprint.artifact import (
    FrozenBlueprint,
    export_policy,
    load_training,
    save_training,
)
from src.blueprint.diagnostics import preflop_first_action
from src.blueprint.solver import BlueprintTrainer, CollectionLimitExceeded, PilotConfig
from src.game.hand import Hand, Table
from src.game.types import ActionKind


def _table(players=2):
    return Table(
        tuple(f"player-{seat}" for seat in range(players)),
        (200,) * players,
        small_blind=1,
        big_blind=2,
        chip_unit="1",
    )


def test_abstract_policy_uses_only_visible_cards_and_legal_actions():
    hand = Hand.start(_table(6), hand_id="visible-test", seed=17)
    view = hand.observe(hand.actor)
    menu = choices(view)
    assert menu
    for item in menu:
        view.legal_actions.validate(item.action)
    assert not any(
        item.action.kind == ActionKind.RAISE
        and item.action.raise_to == view.legal_actions.max_raise_to
        for item in menu
    )
    strong = replace(view, hole_cards=("As", "Ah"))
    weak = replace(view, hole_cards=("7c", "2d"))
    equivalent = replace(view, hole_cards=("Ac", "Ad"))
    assert information_key(strong, choices(strong)) != information_key(
        weak, choices(weak)
    )
    assert information_key(strong, choices(strong)) == information_key(
        equivalent, choices(equivalent)
    )
    assert all(
        item.action.kind != ActionKind.RAISE for item in choices(view, raise_cap=0)
    )


def test_complete_iteration_recovers_and_exports_playable_policy(tmp_path):
    table = _table()
    config = PilotConfig(seed=11, raise_cap=0, max_nodes=5000, max_seconds=10)
    uninterrupted = BlueprintTrainer(table, config)
    uninterrupted.step()
    checkpoint = tmp_path / "checkpoint.json.gz"
    save_training(uninterrupted, checkpoint)
    resumed = load_training(checkpoint)
    assert resumed.table == uninterrupted.table
    assert resumed.config == uninterrupted.config
    uninterrupted.step()
    resumed.step()
    assert {key: asdict(value) for key, value in resumed.nodes.items()} == {
        key: asdict(value) for key, value in uninterrupted.nodes.items()
    }
    assert resumed.iteration == 2
    export = tmp_path / "policy.json.gz"
    digest = export_policy(resumed, export)
    frozen = FrozenBlueprint(
        Checkpoint("blueprint", str(export), digest, "holdem-blueprint-v1"),
        export,
    )
    policy = frozen.policy(3)
    hand = Hand.start(table, hand_id="arena-test", seed=37)
    view = hand.observe(hand.actor)
    view.legal_actions.validate(policy.choose_action(view))
    probe = preflop_first_action(frozen, table)
    assert probe["hand_classes"] == 169
    assert probe["trained_infosets"] + probe["unseen_infosets"] == 169
    with pytest.raises(ValueError, match="cannot resume"):
        load_training(export)


def test_failed_bounded_iteration_publishes_nothing():
    trainer = BlueprintTrainer(_table(), PilotConfig(raise_cap=0, max_nodes=1))
    with pytest.raises(CollectionLimitExceeded):
        trainer.step()
    assert trainer.iteration == 0
    assert trainer.nodes == {}
