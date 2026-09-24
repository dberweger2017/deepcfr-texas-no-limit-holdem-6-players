"""A campaign retains evaluated milestones and emits inspectable dashboard data."""

import json
from pathlib import Path

from scripts.monitor_blueprint import Monitor
from scripts.train_blueprint_campaign import main
from src.blueprint.artifact import save_training
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Table

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
