"""Compare two blueprint keys on exactly the same held-out decisions."""

import argparse
import json
import resource
import sys
from dataclasses import asdict
from hashlib import sha256
from math import isfinite
from pathlib import Path

from src.arena.artifacts import git, write_json
from src.arena.catalog import Checkpoint
from src.arena.registry import PolicyRegistry
from src.arena.run import run
from src.arena.schedule import Plan
from src.blueprint.abstraction import SCHEMA, SUMMARY_SCHEMA
from src.blueprint.artifact import FrozenBlueprint
from src.blueprint.solver import FORMAT


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--uniform", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-rss-gib", default=10.0, type=float)
    args = parser.parse_args(argv)
    plan_data = json.loads(args.plan.read_text(encoding="utf-8"))
    if (
        plan_data.get("candidate") != "blueprint"
        or plan_data.get("baseline") != "blueprint_uniform"
        or not isfinite(args.max_rss_gib)
        or args.max_rss_gib <= 0
    ):
        parser.error("Expected blueprint versus blueprint_uniform and positive RSS cap")
    hashes = {
        label: _hash(path)
        for label, path in (
            ("reference", args.reference),
            ("summary", args.summary),
            ("uniform", args.uniform),
        )
    }
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "manifest.json",
        {
            "revision": git("rev-parse", "HEAD"),
            "source_paths": {
                label: str(path)
                for label, path in (
                    ("reference", args.reference),
                    ("summary", args.summary),
                    ("uniform", args.uniform),
                )
            },
            "sha256": hashes,
            "plan": plan_data,
            "max_rss_gib": args.max_rss_gib,
        },
    )
    specs = (
        Checkpoint("blueprint", str(args.reference), hashes["reference"], FORMAT),
        Checkpoint(
            "blueprint_uniform", str(args.uniform), hashes["uniform"], FORMAT
        ),
    )
    plan = Plan.from_dict({**plan_data, "models": [asdict(spec) for spec in specs]})
    registry = PolicyRegistry(plan)
    reference = registry.models["blueprint"]
    summary = FrozenBlueprint(
        Checkpoint("summary_probe", str(args.summary), hashes["summary"], FORMAT),
        args.summary,
    )
    if reference.abstraction != SCHEMA or summary.abstraction != SUMMARY_SCHEMA:
        raise ValueError("Reference and summary policy schemas differ from the plan")
    if _rss_bytes() >= args.max_rss_gib * 1024**3:
        raise MemoryError("Frozen policies exceed the comparison RSS cap")

    counts: dict[str, dict[str, int]] = {}
    original = reference.distribution

    def measured(view):
        menu, probabilities, reference_hit = original(view)
        summary_menu, _, summary_hit = summary.distribution(view)
        if tuple(item.name for item in menu) != tuple(
            item.name for item in summary_menu
        ):
            raise ValueError("Abstract action menus differ on a held-out observation")
        street = counts.setdefault(
            view.street.value,
            {"decisions": 0, "reference_trained": 0, "summary_trained": 0},
        )
        street["decisions"] += 1
        street["reference_trained"] += int(reference_hit)
        street["summary_trained"] += int(summary_hit)
        return menu, probabilities, reference_hit

    reference.distribution = measured
    report = run(plan, args.out / "arena", registry=registry)
    write_json(
        args.out / "result.json",
        {
            "status": report["status"],
            "completed_hands": report["completed_hands"],
            "invalid_actions": report["invalid_actions"],
            "sha256": hashes,
            "held_out_lookups": counts,
            "peak_rss_bytes": _rss_bytes(),
        },
    )
    return 0 if report["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
