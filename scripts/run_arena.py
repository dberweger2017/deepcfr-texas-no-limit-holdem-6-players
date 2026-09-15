"""Run a declared poker evaluation plan, or reproduce an existing run."""

import argparse
import json
from pathlib import Path

from src.arena.run import reproduce, run
from src.arena.schedule import Plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan", type=Path, help="JSON evaluation plan")
    source.add_argument("--reproduce", type=Path, help="Existing run directory")
    parser.add_argument("--out", required=True, type=Path, help="New output directory")
    args = parser.parse_args(argv)
    try:
        if args.reproduce:
            report = reproduce(args.reproduce, args.out)
        else:
            plan = Plan.from_dict(json.loads(args.plan.read_text(encoding="utf-8")))
            report = run(plan, args.out)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_hands": report["completed_hands"],
                "report": str(args.out / "report.md"),
            }
        )
    )
    return 0 if report["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
