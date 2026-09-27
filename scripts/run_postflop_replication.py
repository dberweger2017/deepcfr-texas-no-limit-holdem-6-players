"""One hash-pinned, completed-iteration continuation from the 5.83M blueprint."""

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import monotonic, time

from src.arena.artifacts import environment, git, write_json
from src.blueprint.abstraction import SCHEMA
from src.blueprint.artifact import load_training, save_training
from src.blueprint.lookup import (DESCENDANT_LINEAGE_SCHEMA, REPLICATION_PARENT,
                                  SAMPLERS)


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def system(command):
    if sys.platform != "darwin":
        return None
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def run(plan, parent, out, mode, seed, *, campaign_deadline):
    if out.exists():
        raise FileExistsError(out)
    if file_hash(parent) != REPLICATION_PARENT:
        raise ValueError("Parent checkpoint hash mismatch")
    if mode not in (1, 4) or seed not in plan["continuation_seeds"]:
        raise ValueError("Arm or seed differs from the frozen plan")
    if git("status", "--porcelain"):
        raise ValueError("Training source must be committed and clean")
    out.mkdir(parents=True)
    start = monotonic()
    started_at = time()
    metadata = {
        "schema": "postflop-replication-training-v1", "plan_sha256":
            sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "parent_checkpoint_sha256": REPLICATION_PARENT, "source_revision": git("rev-parse", "HEAD"),
        "source_dirty": False, "seed": seed, "replicates": mode,
        "environment": environment(), "started_unix_seconds": started_at,
        "swap_before": system(["sysctl", "vm.swapusage"]),
        "memory_pressure_before": system(["memory_pressure", "-Q"]),
    }
    write_json(out / "manifest.json", metadata)
    total_nodes = total_prefixes = total_continuations = total_visits = 0
    total_mass = total_replay = 0.0
    total_new = total_contributing = 0
    visits_by_street = Counter()
    mass_by_street = Counter()
    contribution_counts = Counter()
    milestone = 0
    stop_reason = None
    trainer = None
    discarded_work = 0

    def guard():
        if time() >= campaign_deadline:
            raise TimeoutError("Ten-hour campaign deadline reached")
        if rss() >= plan["limits"]["max_rss_gib"] * 1024**3:
            raise MemoryError("Process RSS limit reached")
        if shutil.disk_usage(out).free < plan["limits"]["min_free_gib"] * 1024**3:
            raise RuntimeError("Free-disk limit reached")

    try:
        guard()
        trainer = load_training(parent)
        load_seconds = monotonic() - start
        parent_iteration = trainer.iteration
        if trainer.config.abstraction != SCHEMA or trainer.config.raise_cap != 2 or trainer.table.button != 0:
            raise ValueError("Parent training frame changed")
        trainer.config = replace(
            trainer.config, seed=seed, postflop_replicates=mode,
            max_nodes=plan["limits"]["max_nodes_per_iteration"],
            max_entries=plan["limits"]["max_entries"],
            max_seconds=plan["limits"]["max_seconds_per_iteration"],
        )
        guard()
        while total_nodes < plan["additional_nodes_per_run"]:
            guard()
            try:
                report = trainer.step()
            except Exception:
                # The sampler publishes an outer iteration only after all roots complete.
                # Its internal partial traversal count is unavailable on failure.
                discarded_work = None
                raise
            total_nodes += report.nodes
            total_prefixes += report.sampled_postflop_prefixes
            total_continuations += report.continuation_samples
            total_visits += report.raw_traverser_visits
            total_mass += report.normalized_update_mass
            total_replay += report.replay_seconds
            total_new += report.new_entries
            total_contributing += report.contributing_infosets
            visits_by_street.update(report.traverser_visits_by_street)
            mass_by_street.update(report.normalized_mass_by_street)
            contribution_counts.update(report.updated_keys)
            row = asdict(report)
            row["updated_keys"] = len(report.updated_keys)
            row.update(additional_nodes=total_nodes,
                       additional_outer_iterations=trainer.iteration-parent_iteration,
                       process_rss_bytes=rss(), elapsed_total_seconds=monotonic()-start)
            append(out / "iterations.jsonl", row)
            while milestone < len(plan["work_milestones"]) and total_nodes >= plan["work_milestones"][milestone]:
                append(out / "milestones.jsonl", {
                    "requested_nodes": plan["work_milestones"][milestone],
                    "completed_nodes": total_nodes, "outer_iterations": trainer.iteration-parent_iteration,
                    "entries": len(trainer.nodes), "new_entries": total_new,
                    "raw_visits": total_visits, "normalized_update_mass": total_mass,
                    "sampled_prefixes": total_prefixes, "continuation_samples": total_continuations,
                    "visits_by_street": dict(visits_by_street),
                    "normalized_mass_by_street": dict(mass_by_street),
                    "contributing_infosets": total_contributing,
                    "elapsed_seconds": monotonic()-start, "rss_bytes": rss(),
                })
                milestone += 1
            guard()
            if total_nodes > plan["additional_nodes_per_run"] + plan["limits"]["max_overshoot_nodes"]:
                raise RuntimeError("Completed iteration exceeded permitted node overshoot")
    except Exception as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
        load_seconds = locals().get("load_seconds")
        parent_iteration = locals().get("parent_iteration", 8733)

    checkpoint_hash = None
    save_error = None
    if trainer is not None and total_nodes:
        try:
            checkpoint_hash = save_training(trainer, out / "checkpoint.json.gz")
        except Exception as exc:
            save_error = f"{type(exc).__name__}: {exc}"
            stop_reason = stop_reason or save_error
    lineage = None
    if checkpoint_hash is not None:
        table = trainer.table
        lineage = {
            "schema": DESCENDANT_LINEAGE_SCHEMA,
            "parent_checkpoint_sha256": REPLICATION_PARENT,
            "output_checkpoint_sha256": checkpoint_hash,
            "source_revision": metadata["source_revision"], "source_dirty": False,
            "key_schema": SCHEMA, "action_menu_raise_cap": trainer.config.raise_cap,
            "sampler_version": SAMPLERS[mode], "continuation_seed": seed,
            "parent_iteration": parent_iteration, "output_iteration": trainer.iteration,
            "completed_nodes": total_nodes,
            "completed_outer_iterations": trainer.iteration-parent_iteration,
            "training_table": {
                "player_ids": list(table.player_ids), "stacks": list(table.stacks),
                "button": table.button, "small_blind": table.small_blind,
                "big_blind": table.big_blind, "chip_unit": table.chip_unit,
            },
        }
        write_json(out / "lineage.json", lineage)
    write_json(out / "outer-contributions.json", dict(contribution_counts))
    result = {
        "status": "complete" if stop_reason is None else "incomplete",
        "stop_reason": stop_reason, "save_error": save_error,
        "parent_checkpoint_sha256": REPLICATION_PARENT,
        "output_checkpoint_sha256": checkpoint_hash,
        "completed_nodes": total_nodes,
        "target_nodes": plan["additional_nodes_per_run"],
        "overshoot_nodes": max(0, total_nodes-plan["additional_nodes_per_run"]),
        "discarded_nodes": discarded_work,
        "completed_outer_iterations": trainer.iteration-parent_iteration if trainer else 0,
        "new_entries": total_new, "sampled_postflop_prefixes": total_prefixes,
        "continuation_samples": total_continuations,
        "raw_traverser_visits": total_visits,
        "normalized_update_mass": total_mass,
        "contributing_infosets_sum": total_contributing,
        "visits_by_street": dict(visits_by_street),
        "normalized_mass_by_street": dict(mass_by_street),
        "replay_seconds": total_replay,
        "load_seconds": load_seconds, "elapsed_seconds": monotonic()-start,
        "peak_process_rss_bytes": rss(),
        "swap_after": system(["sysctl", "vm.swapusage"]),
        "memory_pressure_after": system(["memory_pressure", "-Q"]),
    }
    write_json(out / "result.json", result)
    write_json(out / "checksums.json", {str(p.relative_to(out)): file_hash(p)
               for p in out.rglob("*") if p.is_file() and p.name != "checksums.json"})
    print(json.dumps({k: result[k] for k in ("status", "completed_nodes", "completed_outer_iterations", "elapsed_seconds", "peak_process_rss_bytes", "stop_reason")}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, choices=(1, 4), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--campaign-deadline", type=float, required=True)
    args = parser.parse_args(argv)
    result = run(json.loads(args.plan.read_text()), args.parent, args.out,
                 args.replicates, args.seed, campaign_deadline=args.campaign_deadline)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
