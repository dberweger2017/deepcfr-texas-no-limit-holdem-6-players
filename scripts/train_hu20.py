"""Train one fresh heads-up 20BB K1 seed and capture a windowed policy."""

import argparse
import gzip
import json
import os
import resource
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import monotonic, time

from src.blueprint.abstraction import HU20_CARD_VERSION, HU20_MENU_VERSION, HU20_SCHEMA
from src.arena.schedule import digest
from src.blueprint.artifact import export_policy, save_training
from src.blueprint.solver import BlueprintTrainer, HU20_GAME, PilotConfig
from src.blueprint.windowed import EXTRACTION, _hash, build_index, collect_preflop, write_snapshot
from src.game.hand import Table


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with path.open("a") as target:
        target.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())


def rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def system(command):
    if sys.platform != "darwin":
        return None
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def independent_preflop_density(trainer, counters):
    """Weight visit counts by decisions from collector deals outside training."""
    histogram = Counter()
    for key, (_, action_counts) in counters.items():
        node = trainer.nodes.get(key)
        histogram[node.visits if node else 0] += sum(action_counts)
    total = sum(histogram.values())

    def quantile(fraction):
        threshold = max(1, int(total * fraction + 0.999999))
        seen = 0
        for visits, count in sorted(histogram.items()):
            seen += count
            if seen >= threshold:
                return visits
        return None

    return {"decisions": total, "trained_decisions": total-histogram[0],
            "revisited_decisions": sum(count for visits, count in histogram.items()
                                       if visits > 1),
            "visit_quantiles": {"p25": quantile(.25), "p50": quantile(.5),
                                "p75": quantile(.75), "p90": quantile(.9)},
            "mean_visits": (sum(visits*count for visits, count in histogram.items())/total
                            if total else None)}


def train(plan, seed, out, deadline, *, preflight=False):
    if out.exists():
        raise FileExistsError(out)
    if seed not in plan["training_seeds"]:
        raise ValueError("Seed differs from fixed plan")
    out.mkdir(parents=True)
    started = monotonic()
    table = Table(("player-0", "player-1"), (2000, 2000), button=0)
    config = PilotConfig(seed=seed, abstraction=HU20_SCHEMA, game=HU20_GAME,
                         raise_cap=2, roots_per_seat=1,
                         max_nodes=plan["limits"]["max_nodes_per_iteration"],
                         max_entries=plan["limits"]["max_entries"],
                         max_seconds=plan["limits"]["max_seconds_per_iteration"])
    trainer = BlueprintTrainer(table, config)
    metadata = {"schema": "hu20-training-v2", "seed": seed, "preflight": preflight,
                "game": HU20_GAME, "abstraction": HU20_SCHEMA,
                "plan_sha256": digest(plan), "started_unix_seconds": time(),
                "swap_before": system(["sysctl", "vm.swapusage"]),
                "memory_pressure_before": system(["memory_pressure", "-Q"])}
    write_json(out / "manifest.json", metadata)
    result = {"status": "incomplete", "stop_reason": None}
    total_nodes = 0
    captured = []
    counters = {}
    early_index = 0
    capture_index = 0
    visits = Counter()
    new_by_street = Counter()
    revisited_by_street = Counter()
    try:
        target = plan["preflight_nodes"] if preflight else plan["training_nodes"]
        while total_nodes < target:
            if time() >= deadline:
                raise TimeoutError("HU20 campaign deadline")
            if rss() >= plan["limits"]["max_rss_gib"] * 1024**3:
                raise MemoryError("HU20 process RSS ceiling")
            if shutil.disk_usage(out).free < plan["limits"]["min_free_gib"] * 1024**3:
                raise RuntimeError("HU20 free-disk ceiling")
            report = trainer.step()
            total_nodes += report.nodes
            visits.update(report.traverser_visits_by_street)
            new_by_street.update(report.new_entries_by_street)
            revisited_by_street.update(report.revisited_keys_by_street)
            append(out / "iterations.jsonl", {"iteration": trainer.iteration,
                "additional_nodes": total_nodes, "nodes": report.nodes,
                "entries": report.entries, "new_entries": report.new_entries,
                "traverser_visits_by_street": report.traverser_visits_by_street,
                "new_entries_by_street": report.new_entries_by_street,
                "revisited_keys_by_street": report.revisited_keys_by_street,
                "coverage": report.coverage, "elapsed_seconds": monotonic()-started,
                "rss_bytes": rss()})
            if preflight:
                continue
            if (early_index < len(plan["checkpoints"]) and
                    total_nodes >= plan["checkpoints"][early_index]):
                checkpoint = out / f"checkpoint-{early_index}.json.gz"
                checkpoint_hash = save_training(trainer, checkpoint)
                export_hash = (export_policy(trainer, out / f"policy-{early_index}.json.gz")
                               if early_index < 3 else None)
                append(out / "checkpoints.jsonl", {"index": early_index,
                    "requested_nodes": plan["checkpoints"][early_index],
                    "completed_nodes": total_nodes, "iteration": trainer.iteration,
                    "checkpoint_sha256": checkpoint_hash,
                    "policy_sha256": export_hash,
                    "bytes": checkpoint.stat().st_size, "rss_bytes": rss()})
                early_index += 1
            if (capture_index < 8 and total_nodes >= plan["capture_nodes"][capture_index]):
                start_capture = monotonic()
                path = out / f"snapshot-{capture_index}.jsonl.gz"
                snapshot_digest = write_snapshot(trainer, path)
                collection = collect_preflop(trainer, capture_index,
                    plan["collector_roots_per_seat"], plan["collector_seed"]+seed,
                    counters, max_visited=plan["limits"]["max_collector_states"])
                row = {"index": capture_index,
                       "requested_nodes": plan["capture_nodes"][capture_index],
                       "completed_nodes": total_nodes, "iteration": trainer.iteration,
                       "snapshot_sha256": snapshot_digest, "snapshot_bytes": path.stat().st_size,
                       "collector": collection, "capture_seconds": monotonic()-start_capture,
                       "rss_bytes": rss()}
                append(out / "captures.jsonl", row)
                captured.append(path)
                capture_index += 1
        result.update(completed_nodes=total_nodes, iterations=trainer.iteration,
                      entries=len(trainer.nodes), visits_by_street=dict(visits),
                      new_entries_by_street=dict(new_by_street),
                      revisited_keys_by_street=dict(revisited_by_street))
        if preflight:
            capture_start = monotonic()
            result["snapshot_sha256"] = write_snapshot(trainer, out / "resource-snapshot.jsonl.gz")
            result["collector"] = collect_preflop(trainer, 0,
                plan["collector_roots_per_seat"], plan["collector_seed"]+seed,
                counters, max_visited=plan["limits"]["max_collector_states"])
            result["independent_preflop_density"] = independent_preflop_density(
                trainer, counters)
            result["current_export_sha256"] = export_policy(trainer, out / "resource-current.json.gz")
            result["capture_export_seconds"] = monotonic()-capture_start
        else:
            if early_index != 4 or capture_index != 8:
                raise ValueError("Missing declared checkpoint or capture")
            final_path = out / "checkpoint-3.json.gz"
            result["final_checkpoint_sha256"] = _hash(final_path)
            result["current_export_sha256"] = export_policy(trainer, out / "current.json.gz")
            with gzip.open(out / "preflop-counters.json.gz", "wt", encoding="utf-8") as saved:
                json.dump(counters, saved, sort_keys=True, separators=(",", ":"))
            index_path = out / "policy-index.sqlite"
            index_stats = build_index(captured, counters, index_path)
            capture_rows = [json.loads(line) for line in (out / "captures.jsonl").read_text().splitlines()]
            identity = {"schema": EXTRACTION, "game": HU20_GAME,
                "action_menu": HU20_MENU_VERSION,
                "card_descriptor": HU20_CARD_VERSION,
                "source_checkpoint_sha256": result["final_checkpoint_sha256"],
                "snapshot_sha256": [row["snapshot_sha256"] for row in capture_rows],
                "requested_nodes": plan["capture_nodes"],
                "completed_nodes": [row["completed_nodes"] for row in capture_rows],
                "iterations": [row["iteration"] for row in capture_rows],
                "snapshot_weights": [0.125]*8,
                "collector_seed": plan["collector_seed"]+seed,
                "collector_roots_per_seat": plan["collector_roots_per_seat"],
                "preflop_fallback": "final-current when no collected mass",
                "postflop_fallback": "uniform per absent profile",
                "abstraction": HU20_SCHEMA, "raise_cap": 2,
                "lookup_mode": "native-button-relative-v2",
                "artifact_sha256": index_stats["artifact_sha256"],
                "index_stats": index_stats}
            write_json(out / "policy-manifest.json", identity)
            result["index_stats"] = index_stats
        if time() >= deadline or rss() >= plan["limits"]["max_rss_gib"]*1024**3:
            raise RuntimeError("HU20 extraction exceeded the time or RSS ceiling")
        if shutil.disk_usage(out).free < plan["limits"]["min_free_gib"]*1024**3:
            raise RuntimeError("HU20 extraction exceeded the free-disk ceiling")
        result["status"] = "complete"
    except Exception as exc:
        result["stop_reason"] = f"{type(exc).__name__}: {exc}"
    result.update(elapsed_seconds=monotonic()-started,
                  peak_process_rss_bytes=rss(),
                  swap_after=system(["sysctl", "vm.swapusage"]),
                  memory_pressure_after=system(["memory_pressure", "-Q"]))
    write_json(out / "result.json", result)
    write_json(out / "checksums.json", {str(path.relative_to(out)): _hash(path)
        for path in out.rglob("*") if path.is_file() and path.name != "checksums.json"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    result = train(plan, args.seed, args.out, args.deadline, preflight=args.preflight)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
