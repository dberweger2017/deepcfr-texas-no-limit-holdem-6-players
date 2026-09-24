"""Behavioral checks for the first tabular Hold'em blueprint path."""

from dataclasses import asdict, replace
from json import dumps, loads
from pathlib import Path

import pytest

from scripts.train_blueprint import main as train_blueprint
from src.arena.catalog import Checkpoint
from src.blueprint.abstraction import (
    SCHEMA,
    SUMMARY_SCHEMA,
    choices,
    information_key,
)
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


def test_summary_key_preserves_visible_cards_and_old_key_identity():
    hand = Hand.start(_table(6), hand_id="visible-test", seed=17)
    view = hand.observe(hand.actor)
    menu = choices(view)
    assert information_key(view, menu) == "8706c97cf0899a8944820fb6a8a6d5e6"
    assert information_key(view, menu, schema=SCHEMA) == information_key(view, menu)
    assert information_key(view, menu, schema=SUMMARY_SCHEMA) != information_key(
        view, menu
    )
    strong = replace(view, hole_cards=("As", "Ah"))
    equivalent = replace(view, hole_cards=("Ac", "Ad"))
    weak = replace(view, hole_cards=("7c", "2d"))
    assert information_key(strong, choices(strong), schema=SUMMARY_SCHEMA) == (
        information_key(equivalent, choices(equivalent), schema=SUMMARY_SCHEMA)
    )
    assert information_key(strong, choices(strong), schema=SUMMARY_SCHEMA) != (
        information_key(weak, choices(weak), schema=SUMMARY_SCHEMA)
    )


def test_summary_checkpoint_recovers_and_exports_its_own_schema(tmp_path):
    config = PilotConfig(
        seed=11,
        raise_cap=0,
        max_nodes=5000,
        max_seconds=10,
        abstraction=SUMMARY_SCHEMA,
    )
    trainer = BlueprintTrainer(_table(), config)
    assert trainer.step().schema == SUMMARY_SCHEMA
    checkpoint = tmp_path / "summary-checkpoint.json.gz"
    save_training(trainer, checkpoint)
    resumed = load_training(checkpoint)
    assert resumed.config == config
    trainer.step()
    resumed.step()
    assert save_training(trainer, tmp_path / "direct.json.gz") == save_training(
        resumed, tmp_path / "resumed.json.gz"
    )
    export = tmp_path / "summary-policy.json.gz"
    digest = export_policy(resumed, export)
    frozen = FrozenBlueprint(
        Checkpoint("blueprint", str(export), digest, "holdem-blueprint-v1"),
        export,
    )
    assert frozen.abstraction == SUMMARY_SCHEMA
    hand = Hand.start(_table(), hand_id="summary-play", seed=37)
    view = hand.observe(hand.actor)
    view.legal_actions.validate(frozen.policy(3).choose_action(view))


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


def test_iteration_borrows_current_policy_without_copying_table(monkeypatch):
    trainer = BlueprintTrainer(_table(), PilotConfig(raise_cap=0, max_nodes=5000))
    monkeypatch.setattr(
        BlueprintTrainer,
        "frozen",
        lambda self: pytest.fail("step copied the full policy table"),
    )
    assert trainer.step().iteration == 1


def test_worker_count_preserves_complete_iteration_and_checkpoint(tmp_path):
    config = PilotConfig(
        seed=13, raise_cap=0, roots_per_seat=2, max_nodes=5000, max_seconds=30
    )
    serial = BlueprintTrainer(_table(), config)
    parallel = BlueprintTrainer(_table(), config)
    for _ in range(2):
        one = serial.step(workers=1)
        many = parallel.step(workers=2)
        assert (one.nodes, one.terminals, one.entries) == (
            many.nodes,
            many.terminals,
            many.entries,
        )
        assert one.coverage == many.coverage
        assert sum(one.coverage.values()) == one.nodes - one.terminals
    assert save_training(serial, tmp_path / "serial.json.gz") == save_training(
        parallel, tmp_path / "parallel.json.gz"
    )


def test_parallel_failure_keeps_previous_iteration():
    trainer = BlueprintTrainer(_table(), PilotConfig(raise_cap=0, max_nodes=1))
    with pytest.raises(CollectionLimitExceeded):
        trainer.step(workers=2)
    assert trainer.iteration == 0
    assert trainer.nodes == {}


def test_sparse_checkpoints_keep_exact_iteration_resume(tmp_path):
    plan = loads(
        (Path(__file__).parents[1] / "configs/blueprint/pilot-v1.json").read_text()
    )
    plan["iterations"] = 3
    plan["evaluation"]["blocks"] = 1
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(dumps(plan))
    paused = tmp_path / "paused"
    resumed = tmp_path / "resumed"
    uninterrupted = tmp_path / "uninterrupted"

    assert (
        train_blueprint(
            [
                "--plan",
                str(plan_path),
                "--out",
                str(paused),
                "--max-wall-seconds",
                "0.0001",
                "--checkpoint-seconds",
                "9999",
            ]
        )
        == 2
    )
    assert loads((paused / "result.json").read_text())["iteration"] == 1
    assert loads((paused / "checkpoints.json").read_text())[0]["iteration"] == 1
    plan["trainer"]["max_entries"] = 2 * plan["trainer"]["max_entries"]
    expanded_path = tmp_path / "expanded-plan.json"
    expanded_path.write_text(dumps(plan))
    assert (
        train_blueprint(
            [
                "--plan",
                str(expanded_path),
                "--out",
                str(resumed),
                "--resume",
                str(paused / "checkpoint.json.gz"),
                "--checkpoint-seconds",
                "9999",
            ]
        )
        == 0
    )
    assert (
        train_blueprint(
            [
                "--plan",
                str(expanded_path),
                "--out",
                str(uninterrupted),
                "--checkpoint-seconds",
                "9999",
            ]
        )
        == 0
    )
    assert (
        loads((uninterrupted / "result.json").read_text())["checkpoint_sha256"]
        == loads((resumed / "result.json").read_text())["checkpoint_sha256"]
    )
    assert [
        row["iteration"]
        for row in loads((uninterrupted / "checkpoints.json").read_text())
    ] == [1, 3]
