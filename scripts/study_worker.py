"""Execute one study seed in an isolated process."""

import argparse
import json
from pathlib import Path

from src.solver.experiment import write_json
from src.solver.neural.campaign import Campaign, run_seed
from src.solver.neural.study import Study, explore_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "error"}
    try:
        if job["kind"] == "explore":
            result = explore_seed(
                Study.from_dict(job["study"]), job["seed"], args.out / "study"
            )
        elif job["kind"] == "confirm":
            result = run_seed(
                Campaign.from_dict(job["campaign"]), job["seed"], args.out / "campaign"
            )
        else:
            raise ValueError("Unknown study job")
        report = {"status": "completed", "result": result}
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(args.out / "report.json", report)


if __name__ == "__main__":
    main()
