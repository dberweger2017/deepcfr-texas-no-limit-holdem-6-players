"""Freeze timing-only Part A admission or the selected base's fresh arena plan."""

import argparse
from collections import defaultdict
from copy import deepcopy
import gzip
import json
from pathlib import Path

from scripts.hu20_search_runtime import atomic_json
from src.arena.schedule import digest
from src.blueprint.hu20_turn_solver import file_hash
from src.diagnostics.hu20_search_protocol import freeze_part_a


def freeze_timing(planned, pilot, remaining_seconds):
    summary = json.loads((pilot / "summary.json").read_text())
    if summary["phase"] != "pilot" or summary["status"] != "complete":
        raise ValueError("Timing needs a separate complete pilot")
    if summary["plan"]["models"] != planned["models"] or summary["plan"]["root"] == planned["root"]:
        raise ValueError("Pilot must use the same artifacts and a disjoint schedule")
    timings = defaultdict(list)
    # Deliberately read timing/coordinates only, never returns or selection outcomes.
    for path in sorted(pilot.glob("*.hands.jsonl.gz")):
        with gzip.open(path,"rt") as stream:
            for line in stream:
                row = json.loads(line)
                timings[row["panel"]].append(row["seconds"])
    loads = sum(r["seconds"] for r in summary["loaded"])
    return freeze_part_a(planned, timings, load_seconds=loads,
        pilot_hash=file_hash(pilot / "manifest.json"), budget_seconds=remaining_seconds)


def arena_plan(planned, part_a, calibration):
    if part_a["status"] != "complete" or calibration["status"] != "qualified":
        raise ValueError("Publish complete Part A and qualified calibration before freezing arena")
    base = part_a["base_decision"]["base"]
    plan = deepcopy(planned)
    plan["models"] = [s for s in plan["models"] if s["strategy"] == base]
    if len(plan["models"]) != 3 or len({s["seed"] for s in plan["models"]}) != 3:
        raise ValueError("Arena requires all three selected-base lineages")
    plan.update(stage="frozen-final",root=202610020803,phase="arena",base=base,
        part_a_summary_sha256=digest(part_a),calibration_sha256=digest(calibration),
        selected_search_config=calibration["selected"]["config"],
        expected_hands=12*sum(p["blocks"] for p in plan["panels"]))
    # The arena inherits original scientific counts, not timing-reduced Part A counts.
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planned",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--pilot",type=Path);p.add_argument("--remaining-part-a-seconds",type=float)
    p.add_argument("--part-a",type=Path);p.add_argument("--calibration",type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("Never overwrite a protocol freeze")
    planned=json.loads(a.planned.read_text())
    if a.pilot:
        if a.remaining_part_a_seconds is None:p.error("Deduct pilots/failures from six-hour allowance")
        result=freeze_timing(planned,a.pilot,a.remaining_part_a_seconds)
    else:
        if not a.part_a or not a.calibration:p.error("Supply timing pilot or Part A/calibration summaries")
        result=arena_plan(planned,json.loads(a.part_a.read_text()),json.loads(a.calibration.read_text()))
    atomic_json(a.out,result)
    print(json.dumps({"sha256":file_hash(a.out),"hands":result["expected_hands"],"stage":result["stage"]}))


if __name__ == "__main__":main()
