"""Measure trained lookups and off-menu histories on separate diagnostic deals."""

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

from src.arena.artifacts import write_json
from src.arena.schedule import Plan
from src.blueprint.artifact import load_training
from src.blueprint.evaluation import evaluate
from src.blueprint.offtree import classify_history, lookup_source


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Diagnostic output already exists")
    digest = _sha256(args.checkpoint)
    if digest != args.expected_sha256:
        parser.error("Checkpoint hash differs from the declared checkpoint")
    config = json.loads(args.plan.read_text())
    plans = {name: Plan.from_dict(raw) for name, raw in config["evaluations"].items()}
    if any(plan.split == "test" for plan in plans.values()):
        parser.error("The off-tree diagnostic cannot open a sealed test split")
    trainer = load_training(args.checkpoint)
    results = {}
    for name, plan in plans.items():
        counts = Counter()
        first_divergence = Counter()

        def inspect(view, trained, counts=counts, first_divergence=first_divergence):
            source = lookup_source(
                view, trained=trained, raise_cap=trainer.config.raise_cap
            )
            counts[(view.street.value, source)] += 1
            if source.endswith("off_tree"):
                first = next(
                    row
                    for row in classify_history(
                        view, raise_cap=trainer.config.raise_cap
                    )
                    if not row.in_menu
                )
                first_divergence[(first.street.value, first.reason)] += 1

        result = evaluate(trainer, plan, on_candidate_decision=inspect)
        results[name] = {
            "arena": result,
            "lookup_sources": {
                street: {
                    status: counts[(street, status)]
                    for status in (
                        "trained_in_tree",
                        "trained_after_off_tree",
                        "fallback_in_tree",
                        "fallback_after_off_tree",
                    )
                }
                for street in ("preflop", "flop", "turn", "river")
            },
            "first_divergence": {
                street: {
                    reason: first_divergence[(street, reason)]
                    for reason in ("raise_size", "raise_cap")
                }
                for street in ("preflop", "flop", "turn", "river")
            },
        }
        if result["report"]["status"] != "valid":
            raise RuntimeError(f"Invalid off-tree diagnostic arena: {name}")
    args.out.mkdir(parents=True)
    write_json(
        args.out / "result.json",
        {
            "checkpoint_sha256": digest,
            "iteration": trainer.iteration,
            "entries": len(trainer.nodes),
            "plan": config,
            "evaluations": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
