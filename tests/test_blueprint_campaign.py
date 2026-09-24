"""A campaign retains evaluated milestones and emits inspectable dashboard data."""

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

from scripts.monitor_blueprint import Monitor
from scripts.confirm_blueprint_04 import main as confirm
from scripts.train_blueprint_campaign import main
from src.arena.catalog import Checkpoint
from src.blueprint.abstraction import choices, information_key
from src.blueprint.artifact import FrozenBlueprint, export_policy, save_training
from src.blueprint.evaluation import _TablePolicy
from src.blueprint.solver import BlueprintTrainer, Node, PilotConfig
from src.game.hand import Hand, Table

ROOT = Path(__file__).resolve().parents[1]


class _Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, tag, value, step):
        self.values[(tag, step)] = value

    def flush(self):
        pass


def test_campaign_keeps_checkpoints_and_street_coverage(tmp_path):
    plan = json.loads((ROOT / "configs/blueprint/pilot-v1.json").read_text())
    plan["iterations"] = 1
    campaign = json.loads((ROOT / "configs/blueprint/checkpoint-04-campaign.json").read_text())
    campaign["source_nodes"] = 0
    campaign["source_iteration"] = 0
    campaign["entry_milestones"] = [1]
    for evaluation in campaign["evaluations"].values():
        evaluation["blocks"] = 1
    campaign["confirmation"]["blocks"] = 1
    plan_path = tmp_path / "plan.json"
    campaign_path = tmp_path / "campaign.json"
    plan_path.write_text(json.dumps(plan))
    campaign_path.write_text(json.dumps(campaign))
    table = Table(tuple(plan["table"]["player_ids"]), tuple(plan["table"]["stacks"]))
    source = tmp_path / "source.json.gz"
    save_training(BlueprintTrainer(table, PilotConfig(**plan["trainer"])), source)
    campaign["source_sha256"] = sha256(source.read_bytes()).hexdigest()
    campaign_path.write_text(json.dumps(campaign))
    output = tmp_path / "run"

    assert main([
        "--plan", str(plan_path), "--campaign", str(campaign_path),
        "--resume", str(source), "--out", str(output),
        "--max-wall-seconds", "120", "--max-rss-gib", "4",
        "--min-free-gib", "1", "--checkpoint-seconds", "60",
    ]) == 0
    progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    evaluations = [json.loads(line) for line in (output / "evaluation.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in progress] == [0, 1]
    assert len(list((output / "snapshots").glob("*.json.gz"))) == 2
    assert len(evaluations) == 4
    assert {row["benchmark"] for row in evaluations} == {"random", "styles"}
    assert all(row["status"] == "valid" for row in evaluations)
    assert all("preflop" in row["coverage"] and "flop" in row["coverage"] for row in evaluations)

    writer = _Writer()
    monitor = Monitor(output, writer)
    assert monitor.poll()
    assert ("training/table_entries", 1) in writer.values
    assert ("poker/random/candidate_bb_per_100", 1) in writer.values
    assert ("coverage/random/preflop/trained_fraction", 1) in writer.values
    before = dict(writer.values)
    assert monitor.poll()
    assert writer.values == before
    assert confirm([
        "--campaign", str(campaign_path),
        "--checkpoint", str(output / "checkpoint.json.gz"),
        "--expected-sha256", progress[-1]["checkpoint_sha256"],
        "--out", str(tmp_path / "confirmation"),
    ]) == 0
    assert json.loads((tmp_path / "confirmation" / "result.json").read_text())["report"]["status"] == "valid"


def test_live_evaluator_uses_the_same_current_strategy_as_export(tmp_path):
    table = Table(tuple(f"player-{i}" for i in range(6)), (10000,) * 6)
    trainer = BlueprintTrainer(table, PilotConfig())
    hand = Hand.start(table, hand_id="comparison", seed=17)
    view = hand.observe(hand.actor)
    menu = choices(view, raise_cap=trainer.config.raise_cap)
    key = information_key(view, menu, schema=trainer.config.abstraction)
    names = tuple(item.name for item in menu)
    trainer.nodes[key] = Node(names, [5.0] + [0.0] * (len(names) - 1), [0.0] * len(names), 1)
    path = tmp_path / "policy.json.gz"
    digest = export_policy(trainer, path)
    frozen = FrozenBlueprint(Checkpoint("blueprint", str(path), digest, "holdem-blueprint-v1"), path)
    live_policy = _TablePolicy(trainer, 41, Counter(), uniform=False)
    frozen_policy = frozen.policy(41)
    assert [live_policy.choose_action(view) for _ in range(20)] == [
        frozen_policy.choose_action(view) for _ in range(20)
    ]
