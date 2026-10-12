"""PR236 sequential M4 training with indexed exact gates and stage admission."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import time

from scripts import run_hu100_4b_seed_ladder as inherited
from scripts.evaluate_hu100_direct import put
from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/hu100-4b-seed-replication"
BINARY = ROOT / "build/native-trainer/release/hu20-trainer"
MODULE = "scripts.run_hu100_4b_seed_replication"
PR = 236
SEEDS = (2026100901, 2026100902)
TARGETS = (1_000_000_000, 2_000_000_000, 4_000_000_000)
GIB = 1024**3
INDEX = ROOT / "docs/reports/hu100-4b-seed-ladder-artifacts/model-index.json"


def read(path):
    return json.loads(Path(path).read_text())


def configure():
    # Reuse reviewed family accounting/creation-safe cleanup only; no historical
    # controller, schedules, sample-selection or archive worker is launched.
    inherited.ROOT = ROOT
    inherited.OUT = OUT
    inherited.BINARY = BINARY
    inherited.MODULE = MODULE


def source():
    configure()
    if (OUT / "campaign-stopped.json").exists() or (OUT / "guards/campaign-failure.json").exists():
        raise ValueError("Campaign stopped; no failed science readmission")
    return inherited.source()


def guarded(name, command, seconds=None, stop=None):
    configure()
    inherited.guarded(name, command, seconds=seconds, stop=stop)


def reference(seed, target):
    return {a["kind"]: a for a in read(INDEX)["assets"]
            if a["seed"] == seed and a["requested_nodes"] == target}


def cap_for(target):
    if target <= TARGETS[0]:
        return 57_658_644
    if target == TARGETS[1]:
        return 72_437_552
    return read(OUT / "training-freeze.json")["max_entries"]


def command(seed, target, dest, resume=None):
    args = [BINARY, "train", "--stack-bb", 100, "--seed", seed,
            "--roots-per-seat", 1, "--average-rule", "opponent-sampled",
            "--nodes", target, "--max-entries", cap_for(target),
            "--out", dest / "checkpoint.gz", "--telemetry", dest / "telemetry.jsonl",
            "--stop-file", dest / "stop.json"]
    if resume is not None:
        args += ["--resume", resume, "--resume-sha256", file_hash(resume)]
    return args


def stage_admission(seed, target, dest, projected):
    free = shutil.disk_usage(ROOT).free
    required = 16 * GIB + math.ceil(1.25 * projected) + GIB
    receipt = {"seed": seed, "target": target, "at": time(),
               "free_bytes": free, "required_bytes": required,
               "forecast_model_set_bytes": projected,
               "status": "admitted" if free >= required else "awaiting-storage",
               "whole_campaign_storage_not_yet_admitted": True}
    put(dest / "stage-admission.json", receipt)
    if free < required:
        return False
    return True


def telemetry(dest):
    rows = [json.loads(line) for line in (dest / "telemetry.jsonl").read_text().splitlines()]
    if len(rows) != 1:
        raise ValueError("One complete-iteration terminal telemetry row required")
    row = rows[0]
    if row["binary_sha256"] != file_hash(BINARY):
        raise ValueError("STOP telemetry binary identity mismatch")
    if row["checkpoint_sha256"] != file_hash(dest / "checkpoint.gz"):
        raise ValueError("STOP checkpoint telemetry mismatch")
    return row


def checkpoint_gate(seed, target, dest):
    row = telemetry(dest)
    expected = reference(seed, target).get("checkpoint.gz")
    actual = row["checkpoint_sha256"]
    entries = row["diagnostics"]["entries"]
    if expected is not None and (actual != expected["sha256"] or
                                 (dest / "checkpoint.gz").stat().st_size != expected["bytes"]):
        put(dest / "exactness-mismatch.json", {"expected": expected, "actual": actual})
        raise ValueError("STOP exact checkpoint mismatch; no scientific retry")
    terminal = row["completed_nodes"] < target
    if terminal and (target != TARGETS[2] or not
                     (entries >= cap_for(target) or row["stop_requested"])):
        raise ValueError("STOP missing reference milestone or unexplained incomplete target")
    if not terminal and row["status"] != "saved":
        raise ValueError("STOP unexpected target status")
    put(dest / "gate.json", {"status": "passed", "sha256": actual, "entries": entries,
                            "actual_nodes": row["completed_nodes"],
                            "requested_nodes": target, "max_entries": cap_for(target),
                            "terminal_capacity_stop": terminal,
                            "stop_requested": row["stop_requested"]})
    return row


def export_gate(seed, target, dest):
    pins = reference(seed, target)
    verified = {}
    for name, pin in pins.items():
        path = dest / name
        actual = file_hash(path)
        if actual != pin["sha256"] or path.stat().st_size != pin["bytes"]:
            put(dest / "export-mismatch.json", {"file": name, "expected": pin, "actual": actual})
            raise ValueError("STOP indexed current/average mismatch")
        verified[name] = {"sha256": actual, "bytes": path.stat().st_size}
    put(dest / "indexed-reference.json", {"status": "byte-exact" if pins else "new",
                                        "index": str(INDEX.relative_to(ROOT)), "files": verified})


def timing():
    revision = source()
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "timing-training"
    dest.mkdir(exist_ok=False)
    if not stage_admission(SEEDS[0], 1_000_000, dest, 100_000_000):
        return
    guarded("timing-training", command(SEEDS[0], 1_000_000, dest), stop=dest / "stop.json")
    row = telemetry(dest)
    receipt = read(OUT / "guards/operations/timing-training/receipt.json")
    history = read(ROOT / "docs/reports/hu100-4b-seed-ladder-artifacts/campaign-resources.json")
    prior = next(r for r in history["scientific_operations"]
                 if r["name"] == "train-2026100601-4000000000")
    overhead = max(0, receipt["peak_family_rss_bytes"] - receipt["kernel_command_peak_rss_bytes"])
    previous_overhead = prior["peak_family_rss_bytes"] - prior["kernel_command_peak_rss_bytes"]
    additional = max(0, overhead - previous_overhead)
    per_entry = prior["peak_family_rss_bytes"] / 70_466_972
    cap = math.floor((7 * GIB - 1024**2 - additional) / (1.10 * per_entry))
    operation_quotes = {}
    for seed in SEEDS:
        for target in TARGETS:
            prior_seed = 2026100601 if target == TARGETS[2] else seed
            for phase in ("train", "export", "audit"):
                name = f"{phase}-{prior_seed}-{target}"
                old = next(r for r in history["scientific_operations"] if r["name"] == name)
                operation_quotes[f"{phase}-{seed}-{target}"] = 2 * old["seconds"]
    put(OUT / "training-freeze.json", {
        "source": revision, "binary_sha256": file_hash(BINARY),
        "native_source_tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD:native/hu20-trainer"], text=True).strip(),
        "pilot_actual_nodes": row["completed_nodes"],
        "fresh_pilot_elapsed_seconds_including_writes": row["elapsed_seconds_including_writes"],
        "fresh_pilot_write_seconds": row["write_seconds"],
        "fresh_pilot_training_nodes_per_second": row["completed_nodes"] /
            (row["elapsed_seconds_including_writes"] - row["write_seconds"]),
        "prior_4b_family_peak_bytes": prior["peak_family_rss_bytes"],
        "prior_4b_entries": 70_466_972, "headroom_multiplier": 1.10,
        "measured_pilot_family_overhead_bytes": overhead,
        "prior_4b_family_overhead_bytes": previous_overhead,
        "additional_fixed_overhead_reserve_bytes": additional,
        "complete_iteration_reserve_bytes": 1024**2, "max_entries": cap,
        "operation_quotes_seconds": operation_quotes,
        "quote_basis": "2x measured same-M4 PR226 operation costs; fresh1M timing recorded",
        "outcomes_used_for_budget": False,
        "stagewise_start_authorization": "Owner: yes im workign on it, you can start already",
        "evaluation_not_frozen": True})


def train_seed(seed):
    revision = source()
    freeze = read(OUT / "training-freeze.json")
    if freeze["source"] != revision or freeze["binary_sha256"] != file_hash(BINARY):
        raise ValueError("Frozen training source/binary required")
    previous = None
    for target in TARGETS:
        dest = OUT / "training" / str(seed) / str(target)
        if dest.exists():
            raise ValueError("Existing stage requires explicit inspected continuation; never overwrite or retry")
        dest.mkdir(parents=True, exist_ok=False)
        pins = reference(seed, target)
        projected = sum(p["bytes"] for p in pins.values()) if pins else math.ceil(
            7_179_057_138 * freeze["max_entries"] / 70_466_972)
        if not stage_admission(seed, target, dest, projected):
            put(OUT / f"seed-{seed}-awaiting-storage.json", {"target": target, "at": time()})
            return
        quotes = freeze["operation_quotes_seconds"]
        guarded(f"train-{seed}-{target}", command(seed, target, dest, previous),
                quotes[f"train-{seed}-{target}"], dest / "stop.json")
        row = checkpoint_gate(seed, target, dest)
        guarded(f"export-{seed}-{target}", [BINARY, "export", dest / "checkpoint.gz",
                "--current", dest / "current.gz", "--average", dest / "average.gz",
                "--zero-mass", "uniform"], quotes[f"export-{seed}-{target}"])
        guarded(f"audit-{seed}-{target}", [sys.executable, "-m", "scripts.audit_native_hu_checkpoint",
                "--checkpoint", dest / "checkpoint.gz", "--current", dest / "current.gz",
                "--average", dest / "average.gz", "--stack-bb", 100,
                "--target-nodes", row["completed_nodes"], "--out", dest / "audit.json"],
                quotes[f"audit-{seed}-{target}"])
        export_gate(seed, target, dest)
        previous = dest / "checkpoint.gz"
        if row["completed_nodes"] < target:
            break
    put(OUT / f"seed-{seed}-complete.json", {"status": "verified",
                                          "terminal": str(previous), "at": time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("guard", "timing", "train-seed"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args, extra = parser.parse_known_args()
    configure()
    if args.operation == "guard":
        # The inherited guard parser receives its original arguments verbatim.
        sys.argv = [sys.argv[0], "guard", *sys.argv[2:]]
        inherited.guard_main()
        return
    if extra:
        parser.error("Unexpected arguments")
    try:
        if args.operation == "timing":
            timing()
        elif args.seed is None:
            parser.error("--seed required")
        else:
            train_seed(args.seed)
    except BaseException as exc:
        marker = OUT / "campaign-stopped.json"
        if not marker.exists():
            put(marker, {"at": time(), "operation": args.operation, "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
