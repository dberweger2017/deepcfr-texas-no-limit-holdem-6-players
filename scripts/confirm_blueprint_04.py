"""Run the sealed random-opponent test once on the chosen blueprint checkpoint."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from src.arena.artifacts import write_json
from src.arena.schedule import Plan
from src.blueprint.artifact import load_training
from src.blueprint.evaluation import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Confirmation output already exists")
    fingerprint = sha256()
    with args.checkpoint.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    digest = fingerprint.hexdigest()
    if digest != args.expected_sha256:
        parser.error("Chosen checkpoint hash differs from the declared checkpoint")
    config = json.loads(args.campaign.read_text())
    plan = Plan.from_dict(config["confirmation"])
    if plan.split != "test" or plan.opponents != ("random",):
        parser.error("Confirmation requires the frozen random test schedule")
    trainer = load_training(args.checkpoint)
    result = evaluate(trainer, plan)
    args.out.mkdir(parents=True)
    write_json(args.out / "result.json", {
        "checkpoint_sha256": digest,
        "iteration": trainer.iteration,
        "entries": len(trainer.nodes),
        **result,
    })
    if result["report"]["status"] != "valid":
        raise RuntimeError("Invalid final random arena evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
