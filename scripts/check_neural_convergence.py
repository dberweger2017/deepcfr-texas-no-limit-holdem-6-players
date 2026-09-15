"""Run one declared convergence seed, or summarize every seed without dropping failures."""

import argparse
import json
from pathlib import Path

from src.solver.neural.campaign import Campaign, run_seed, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", type=int)
    mode.add_argument("--summarize", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args(argv)
    try:
        if args.summarize and (args.resume or args.stop_after is not None):
            raise ValueError("Resume and stop options require --seed")
        campaign = Campaign.from_dict(json.loads(args.campaign.read_text()))
        result = (
            summarize(campaign, args.summarize, args.out)
            if args.summarize
            else run_seed(
                campaign,
                args.seed,
                args.out,
                resume=args.resume,
                stop_after=args.stop_after,
            )
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        ArithmeticError,
    ) as exc:
        parser.exit(2, f"Neural convergence check failed: {exc}\n")
    print(
        json.dumps(
            {"status": result["status"], "report": str(args.out / "report.json")}
        )
    )
    return 0 if result["status"] in {"passed", "paused"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
