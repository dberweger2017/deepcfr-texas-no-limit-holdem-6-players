"""Train, resume or reproduce the declared Hold'em baseline experiment."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.holdem.experiment import Experiment, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--plan", type=Path)
    inputs.add_argument("--resume", type=Path)
    inputs.add_argument("--reproduce", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--seed", type=int, help="Run one seed from a declared plan in this process"
    )
    args = parser.parse_args()
    previous = args.resume or args.reproduce
    data = (
        json.loads((previous / "manifest.json").read_text())["plan"]
        if previous
        else json.loads(args.plan.read_text())
    )
    plan = Experiment.from_dict(data)
    if args.seed is not None:
        if previous or args.seed not in plan.seeds:
            parser.error("--seed must select a declared seed from --plan")
        plan = replace(plan, seeds=(args.seed,))
    result = run(
        plan,
        args.out,
        resume=args.resume,
        reproduce=args.reproduce,
        stop_after=args.stop_after,
    )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "jobs": len(result["jobs"]),
                "promoted": result["promoted"],
            }
        )
    )


if __name__ == "__main__":
    main()
