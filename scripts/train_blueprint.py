"""Run a bounded tabular blueprint pilot and evaluate its frozen export."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.arena.artifacts import environment, git, write_json
from src.arena.catalog import Checkpoint
from src.arena.registry import PolicyRegistry
from src.arena.run import run
from src.arena.schedule import Plan
from src.blueprint.artifact import export_policy, load_training, save_training
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    target = plan["iterations"] if args.stop_after is None else args.stop_after
    if type(target) is not int or not 0 <= target <= plan["iterations"]:
        parser.error("Stop after must lie between zero and planned iterations")
    table_data = plan["table"]
    table = Table(
        tuple(table_data["player_ids"]),
        tuple(table_data["stacks"]),
        table_data["button"],
        table_data["small_blind"],
        table_data["big_blind"],
        table_data["chip_unit"],
    )
    config = PilotConfig(**plan["trainer"])
    trainer = (
        load_training(args.resume) if args.resume else BlueprintTrainer(table, config)
    )
    if trainer.table != table or trainer.config != config:
        parser.error("Resume checkpoint differs from the frozen pilot plan")
    if trainer.iteration > target:
        parser.error("Resume checkpoint is beyond the requested stopping point")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "manifest.json",
        {
            "plan": plan,
            "revision": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "environment": environment(),
            "resumed_from": str(args.resume) if args.resume else None,
        },
    )
    reports = []
    for _ in range(trainer.iteration, target):
        report = trainer.step()
        reports.append(asdict(report))
        save_training(trainer, args.out / "checkpoint.json.gz")
        write_json(args.out / "iterations.json", reports)
    checkpoint_hash = save_training(trainer, args.out / "checkpoint.json.gz")
    policy_path = args.out / "policy.json.gz"
    policy_hash = export_policy(trainer, policy_path)
    models_dir = args.out / "models"
    models_dir.mkdir()
    (models_dir / f"{policy_hash}.json.gz").write_bytes(policy_path.read_bytes())
    eval_plan = Plan.from_dict(
        {
            **plan["evaluation"],
            "models": [
                asdict(
                    Checkpoint(
                        "blueprint",
                        str(policy_path),
                        policy_hash,
                        "holdem-blueprint-v1",
                    )
                )
            ],
        }
    )
    evaluation = run(
        eval_plan,
        args.out / "evaluation",
        registry=PolicyRegistry(eval_plan, artifact_dir=models_dir),
    )
    result = {
        "iteration": trainer.iteration,
        "entries": len(trainer.nodes),
        "checkpoint_sha256": checkpoint_hash,
        "policy_sha256": policy_hash,
        "evaluation_status": evaluation["status"],
        "completed_hands": evaluation["completed_hands"],
        "strength_claim": False,
    }
    write_json(args.out / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if evaluation["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
