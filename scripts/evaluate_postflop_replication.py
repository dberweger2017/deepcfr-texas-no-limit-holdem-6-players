"""One-arm sequential evaluation on the common frozen paired schedule."""

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from random import Random
from time import monotonic, time

from src.arena.artifacts import environment, git, write_json
from src.arena.policies import make_policy
from src.arena.runner import _fixed
from src.arena.schedule import Plan, build_schedule, digest, schedule_document
from src.blueprint.abstraction import BUTTON_ZERO_COMPAT_LOOKUP, choices, information_key
from src.blueprint.artifact import load_training
from src.blueprint.lookup import NoFreeFoldDistribution, REPLICATION_PARENT, TableDistribution
from src.game.types import ActionKind


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
    with Path(path).open("a", encoding="utf-8") as target:
        target.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())


def suite_plan(suite):
    return Plan.from_dict({
        "scenarios": [{"name": "six-handed-100bb", "stacks": [10_000]*6}],
        "candidate": "check_call", "baseline": "tight_aggressive",
        "opponents": suite["opponents"], "blocks": suite["blocks"],
        "root_seed": suite["root_seed"], "split": "validation",
        "max_decisions": suite["max_decisions"],
    })


class Hero:
    def __init__(self, arm, source, seed, coverage):
        self.arm = arm
        self.source = source
        self.random = Random(seed)
        self.style = make_policy("tight_aggressive", seed) if arm == "TAG" else None
        self.coverage = coverage

    def choose_action(self, view):
        if self.style is not None:
            action = self.style.choose_action(view)
        else:
            menu, probabilities, trained = self.source.distribution(view)
            key = information_key(view, menu, schema=self.source.source.trainer.config.abstraction,
                                  lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP)
            node = self.source.source.trainer.nodes.get(key)
            self.coverage[(view.street.value, "decisions")] += 1
            self.coverage[(view.street.value, "trained" if trained else "missing")] += 1
            if node is not None:
                self.coverage[(view.street.value, "visit_sum")] += node.visits
                if node.visits == 1:
                    self.coverage[(view.street.value, "one_visit")] += 1
            self.coverage[(view.street.value, "distinct", key)] += 1
            action = self.random.choices(menu, weights=probabilities, k=1)[0].action
        self.coverage[(view.street.value, "actions", action.kind.value)] += 1
        if ActionKind.CHECK in view.legal_actions.kinds:
            self.coverage[(view.street.value, "free_check_eligible")] += 1
            if action.kind == ActionKind.FOLD:
                self.coverage[(view.street.value, "free_fold")] += 1
        return action


def run(plan, arm, checkpoint, lineage_file, out, *, campaign_deadline, resource_only=False):
    if out.exists():
        raise FileExistsError(out)
    if arm not in plan["arms"]:
        raise ValueError("Arm differs from frozen plan")
    if git("status", "--porcelain"):
        raise ValueError("Evaluation source must be committed and clean")
    out.mkdir(parents=True)
    started = monotonic()
    metadata = {
        "schema": "postflop-replication-evaluation-v1", "arm": arm,
        "resource_only": resource_only, "source_revision": git("rev-parse", "HEAD"),
        "source_dirty": False, "environment": environment(),
        "started_unix_seconds": time(), "checkpoint_sha256": None,
        "swap_before": system(["sysctl", "vm.swapusage"]),
        "memory_pressure_before": system(["memory_pressure", "-Q"]),
    }
    write_json(out / "manifest.json", metadata)
    attempts = 0
    completed_blocks = Counter()
    coverage = defaultdict(Counter)
    stop_reason = None
    wrapper = None

    def guard():
        if time() >= campaign_deadline:
            raise TimeoutError("Ten-hour campaign deadline reached")
        if rss() >= plan["limits"]["max_rss_gib"] * 1024**3:
            raise MemoryError("Evaluation process RSS limit reached")
        if shutil.disk_usage(out).free < plan["limits"]["min_free_gib"] * 1024**3:
            raise RuntimeError("Evaluation free-disk limit reached")

    try:
        guard()
        if arm == "TAG":
            source = None
        else:
            if checkpoint is None or file_hash(checkpoint) != (
                    REPLICATION_PARENT if arm in ("U_safe", "parent")
                    else json.loads(lineage_file.read_text())["output_checkpoint_sha256"]):
                raise ValueError("Checkpoint differs from pinned parent or lineage")
            trainer = load_training(checkpoint)
            guard()
            lineage = json.loads(lineage_file.read_text()) if lineage_file else None
            table = TableDistribution(
                trainer, lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP,
                checkpoint_sha256=file_hash(checkpoint), lineage=lineage,
                uniform=arm == "U_safe",
            )
            source = NoFreeFoldDistribution(table)
            wrapper = source
            metadata["checkpoint_sha256"] = file_hash(checkpoint)
            metadata["lineage_sha256"] = file_hash(lineage_file) if lineage_file else None
            write_json(out / "manifest.json", metadata)
        for suite_name, suite in plan["suites"].items():
            if resource_only:
                suite = {**suite, "blocks": min(4, suite["blocks"])}
            arena_plan = suite_plan(suite)
            schedule = build_schedule(arena_plan)
            metadata.setdefault("schedule_sha256", {})[suite_name] = digest(schedule_document(arena_plan))
            write_json(out / "manifest.json", metadata)
            scenario = arena_plan.scenarios[0]
            for block in schedule:
                guard()
                for rotation in range(6):
                    guard()
                    ids = tuple(f"player-{(seat-rotation)%6}" for seat in range(6))
                    trace, timings = {"events": [], "reloads": {}}, []
                    row = {"suite": suite_name, "block": block.index,
                           "rotation": rotation, "arm": arm, "status": "started",
                           "candidate_chips": None, "big_blind": scenario.big_blind,
                           "deal_seed": block.deal_seeds[0], "button": block.button,
                           "opponents": list(block.opponents)}
                    begin = monotonic()
                    try:
                        hero = Hero(arm, source, block.action_seeds[0], coverage[suite_name])
                        policies = {"player-0": hero}
                        policies.update({f"player-{index}": make_policy(name, block.action_seeds[index])
                                         for index, name in enumerate(block.opponents, 1)})
                        net, _ = _fixed(scenario, block, rotation, ids, policies,
                                        arena_plan.max_decisions, trace, timings,
                                        f"postflop-{suite_name}")
                        if sum(net) != 0:
                            raise ValueError("Hand did not conserve chips")
                        if not resource_only:
                            row["candidate_chips"] = net[rotation]
                        row["status"] = "completed"
                    except Exception as exc:
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                                   public_events=trace["events"])
                    row["seconds"] = monotonic()-begin
                    row["hero_decisions"] = sum(t["player_id"] == "player-0" for t in timings)
                    row["public_events_sha256"] = digest(trace["events"])
                    append(out / "hands.jsonl", row)
                    attempts += 1
                    if row["status"] != "completed":
                        raise RuntimeError(row["error"])
                completed_blocks[suite_name] += 1
                if completed_blocks[suite_name] % 64 == 0:
                    append(out / "progress.jsonl", {"suite": suite_name,
                           "blocks": completed_blocks[suite_name], "seconds": monotonic()-started,
                           "rss_bytes": rss()})
    except Exception as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
    coverage_doc = {}
    for suite_name, counts in coverage.items():
        coverage_doc[suite_name] = {
            "counts": [{"coordinates": list(key), "count": value}
                       for key, value in sorted(counts.items()) if "distinct" not in key],
            "distinct": [{"street": key[0], "key": key[2], "reached": value}
                         for key, value in counts.items()
                         if len(key) == 3 and key[1] == "distinct"],
        }
    write_json(out / "coverage.json", coverage_doc)
    result = {
        "status": "complete" if stop_reason is None else "incomplete",
        "stop_reason": stop_reason, "attempts": attempts,
        "completed_blocks": dict(completed_blocks),
        "elapsed_seconds": monotonic()-started, "peak_process_rss_bytes": rss(),
        "swap_after": system(["sysctl", "vm.swapusage"]),
        "memory_pressure_after": system(["memory_pressure", "-Q"]),
        "wrapper_interventions": dict(wrapper.interventions) if wrapper else {},
    }
    write_json(out / "result.json", result)
    write_json(out / "checksums.json", {str(p.relative_to(out)): file_hash(p)
               for p in out.rglob("*") if p.is_file() and p.name != "checksums.json"})
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--lineage", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--campaign-deadline", type=float, required=True)
    parser.add_argument("--resource-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(json.loads(args.plan.read_text()), args.arm, args.checkpoint,
                 args.lineage, args.out, campaign_deadline=args.campaign_deadline,
                 resource_only=args.resource_only)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
