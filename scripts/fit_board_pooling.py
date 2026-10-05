"""Sequential, bounded-memory policy fitting; no native solver or new science."""

import argparse
import json
from pathlib import Path
import signal

from scripts.run_board_pooling import write_pooled_policies
from src.diagnostics.saved_hu20 import file_hash


def fit(plan_path, run, approval_path):
    approval = json.loads(approval_path.read_text())
    if file_hash(plan_path) != approval["plan_sha256"] or not approval["qualification_passed"]:
        raise ValueError("Unqualified or changed fit protocol")
    plan = json.loads(plan_path.read_text())
    split_path = Path(plan["crossfit"]["path"])
    if file_hash(split_path) != plan["crossfit"]["sha256"]:
        raise ValueError("Frozen split fingerprint differs")
    folds = json.loads(split_path.read_text())["folds"]
    jobs = json.loads((run / "schedule.json").read_text())
    mask = json.loads((run / "common-mask.json").read_text())
    eligible = [j for j in jobs if j["spot"] in mask["admitted"]]
    for job in eligible:
        result = json.loads((run / "collect" / job["job"] / "result.json").read_text())
        response = run / "collect" / job["job"] / "solver/response.jsonl"
        if not result["eligible"] or result["job"] != job or file_hash(response) != result["runtime"]["response_sha256"]:
            raise ValueError("Fit response differs from admitted atomic result")
    write_pooled_policies(eligible, lambda j: run / "collect" / j["job"] / "solver/response.jsonl", run, folds)


def main():
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Owned pool fit stopped")))
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "approval"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    fit(a.plan, a.run, a.approval)


if __name__ == "__main__":
    main()
