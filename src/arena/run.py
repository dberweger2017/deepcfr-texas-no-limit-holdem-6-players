"""Create a new artifact directory; existing experiments are never overwritten."""

import json
from pathlib import Path
from time import perf_counter

from src.arena.artifacts import manifest, validate_manifest, write_json
from src.arena.registry import PolicyRegistry
from src.arena.report import markdown, performance, summarize
from src.arena.runner import run_schedule
from src.arena.schedule import Plan, build_schedule, canonical, schedule_document


def run(plan: Plan, output: Path, *, registry: PolicyRegistry | None = None) -> dict:
    registry = registry or PolicyRegistry(plan)
    inputs = manifest(plan, registry)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", inputs)
    write_json(output / "schedule.json", schedule_document(plan))
    registry.snapshot(output)
    rows, timings = [], []
    started = perf_counter()
    try:
        with (
            (output / "hands.jsonl").open("w", encoding="utf-8") as hands_file,
            (output / "timings.jsonl").open("w", encoding="utf-8") as times_file,
        ):

            def emit(row, timing):
                hands_file.write(canonical(row) + "\n")
                hands_file.flush()
                times_file.write(canonical(timing) + "\n")
                times_file.flush()
                rows.append(row)
                timings.append(timing)

            with registry.runtime():
                run_schedule(
                    plan, build_schedule(plan), emit, factory=registry.make_policy
                )
    finally:
        report = summarize(plan, rows)
        report["policies"] = inputs["policies"]
        report["performance"] = performance(timings, perf_counter() - started)
        write_json(output / "report.json", report)
        (output / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def reproduce(original: Path, output: Path) -> dict:
    inputs = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    registry = PolicyRegistry(
        Plan.from_dict(inputs["plan"]), artifact_dir=original / "models"
    )
    plan = validate_manifest(inputs, registry)
    expected_schedule = json.loads(
        (original / "schedule.json").read_text(encoding="utf-8")
    )
    if canonical(expected_schedule) != canonical(schedule_document(plan)):
        raise ValueError("Stored schedule does not match the manifest")
    result = run(plan, output, registry=registry)
    if result["status"] != "valid":
        raise ValueError("Reproduction did not complete successfully")
    if (original / "hands.jsonl").read_bytes() != (output / "hands.jsonl").read_bytes():
        raise ValueError("Reproduced outcomes differ; both runs have been retained")
    original_report = json.loads((original / "report.json").read_text(encoding="utf-8"))
    summary = {k: v for k, v in result.items() if k != "performance"}
    original_summary = {k: v for k, v in original_report.items() if k != "performance"}
    if summary != original_summary:
        raise ValueError("Reproduced report differs; both runs have been retained")
    return result
