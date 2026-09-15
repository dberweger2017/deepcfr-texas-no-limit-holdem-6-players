"""Run a declared Deep CFR pilot, its reproduction, or controlled fitting checks."""

import argparse
import json
from pathlib import Path

from src.solver.neural.experiment import (
    Plan,
    check_fitting,
    check_refit,
    reproduce,
    run,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", type=Path)
    mode.add_argument("--reproduce", type=Path)
    mode.add_argument("--fitting-check", action="store_true")
    mode.add_argument("--refit-plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.fitting_check:
            report = check_fitting(args.out)
        elif args.refit_plan:
            report = check_refit(
                Plan.from_dict(json.loads(args.refit_plan.read_text())), args.out
            )
        elif args.reproduce:
            report = reproduce(args.reproduce, args.out)
        else:
            report = run(Plan.from_dict(json.loads(args.plan.read_text())), args.out)
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        ArithmeticError,
    ) as exc:
        parser.exit(2, f"Neural check failed: {exc}\n")
    print(
        json.dumps(
            {"status": report["status"], "report": str(args.out / "report.json")}
        )
    )
    return 0 if report["status"] in {"completed", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
