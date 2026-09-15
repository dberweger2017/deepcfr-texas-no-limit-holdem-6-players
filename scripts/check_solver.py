"""Run the declared small-game checks or reproduce a completed reference bundle."""

import argparse
import json
from pathlib import Path

from src.solver.experiment import Plan, reproduce, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan", type=Path)
    source.add_argument("--reproduce", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.reproduce:
            report = reproduce(args.reproduce, args.out)
        else:
            plan = Plan.from_dict(json.loads(args.plan.read_text(encoding="utf-8")))
            report = run(plan, args.out)
    except (ValueError, TypeError, KeyError, OSError, ArithmeticError) as exc:
        parser.exit(2, f"Reference check failed: {exc}\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "runs": len(report["runs"]),
                "report": str(args.out / "report.json"),
            }
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
