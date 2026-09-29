"""Evaluate one frozen extracted blueprint arm on coupled six-seat blocks."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from random import Random
from time import monotonic, time

from src.arena.policies import make_policy
from src.arena.runner import _fixed
from src.arena.schedule import build_schedule, digest, schedule_document
from src.blueprint.lookup import NoFreeFoldDistribution
from src.blueprint.windowed import WindowedDistribution, _hash
from src.game.types import ActionKind
from scripts.evaluate_postflop_replication import append, rss, suite_plan, system


class Hero:
    def __init__(self, source, seed, telemetry):
        self.source = source
        self.random = Random(seed)
        self.telemetry = telemetry

    def choose_action(self, view):
        menu, probabilities, trained = self.source.distribution(view)
        street = view.street.value
        self.telemetry[(street, "decisions")] += 1
        self.telemetry[(street, "trained" if trained else "missing")] += 1
        if ActionKind.CHECK in view.legal_actions.kinds:
            self.telemetry[(street, "free_check_eligible")] += 1
        action = self.random.choices(menu, weights=probabilities, k=1)[0].action
        self.telemetry[(street, "action", action.kind.value)] += 1
        if ActionKind.CHECK in view.legal_actions.kinds and action.kind == ActionKind.FOLD:
            self.telemetry[(street, "free_fold")] += 1
        return action


def run(plan, arm, index, manifest_path, out, deadline, *, resource_only=False):
    if out.exists():
        raise FileExistsError(out)
    if arm not in plan["arms"] or arm[0] not in "CPFA":
        raise ValueError("Arm is not in frozen plan")
    seed = int(arm[1:])
    if seed not in plan["continuation_seeds"]:
        raise ValueError("Arm seed differs from frozen plan")
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_checkpoint_sha256"] != plan["source_checkpoints"][str(seed)]:
        raise ValueError("Source checkpoint differs from frozen plan")
    started = monotonic()
    out.mkdir(parents=True)
    metadata = {"arm": arm, "resource_only": resource_only,
                "plan_sha256": digest(plan), "index_sha256": _hash(index),
                "manifest_sha256": _hash(manifest_path),
                "started_unix_seconds": time(),
                "swap_before": system(["sysctl", "vm.swapusage"]),
                "memory_pressure_before": system(["memory_pressure", "-Q"])}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    source = wrapper = None
    attempts = 0
    blocks = Counter()
    telemetry = Counter()
    stop_reason = None

    def guard():
        if time() >= deadline:
            raise TimeoutError("Ten-hour campaign deadline")
        if rss() >= plan["limits"]["max_rss_gib"] * 1024**3:
            raise MemoryError("Evaluation RSS ceiling")
        if shutil.disk_usage(out).free < plan["limits"]["min_free_gib"] * 1024**3:
            raise RuntimeError("Evaluation disk ceiling")

    try:
        guard()
        source = WindowedDistribution(index, manifest, arm[0])
        wrapper = NoFreeFoldDistribution(source)
        for suite_name, suite in plan["suites"].items():
            if resource_only:
                suite = {**suite, "blocks": min(4, suite["blocks"])}
            arena_plan = suite_plan(suite)
            schedule = build_schedule(arena_plan)
            metadata.setdefault("schedule_sha256", {})[suite_name] = digest(schedule_document(arena_plan))
            (out / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            scenario = arena_plan.scenarios[0]
            for block in schedule:
                guard()
                for rotation in range(6):
                    ids = tuple(f"player-{(seat-rotation)%6}" for seat in range(6))
                    trace, timings = {"events": [], "reloads": {}}, []
                    row = {"suite": suite_name, "block": block.index, "rotation": rotation,
                           "arm": arm, "status": "started", "candidate_chips": None,
                           "big_blind": scenario.big_blind, "deal_seed": block.deal_seeds[0],
                           "button": block.button, "opponents": list(block.opponents)}
                    begin = monotonic()
                    try:
                        hero = Hero(wrapper, block.action_seeds[0], telemetry)
                        policies = {"player-0": hero}
                        policies.update({f"player-{i}": make_policy(name, block.action_seeds[i])
                                         for i, name in enumerate(block.opponents, 1)})
                        net, _ = _fixed(scenario, block, rotation, ids, policies,
                                        arena_plan.max_decisions, trace, timings,
                                        f"extract-{suite_name}")
                        if sum(net) != 0:
                            raise ValueError("Hand did not conserve chips")
                        if not resource_only:
                            row["candidate_chips"] = net[rotation]
                        row["status"] = "completed"
                    except Exception as exc:
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                                   public_events=trace["events"])
                    row["seconds"] = monotonic() - begin
                    row["hero_decisions"] = sum(t["player_id"] == "player-0" for t in timings)
                    row["public_events_sha256"] = digest(trace["events"])
                    append(out / "hands.jsonl", row)
                    attempts += 1
                    if row["status"] != "completed":
                        raise RuntimeError(row["error"])
                blocks[suite_name] += 1
                if blocks[suite_name] % 64 == 0:
                    append(out / "progress.jsonl", {"suite": suite_name,
                           "blocks": blocks[suite_name], "seconds": monotonic()-started,
                           "rss_bytes": rss()})
    except Exception as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
    finally:
        if source is not None:
            source.close()
    result = {"status": "complete" if stop_reason is None else "incomplete",
              "stop_reason": stop_reason, "attempts": attempts, "completed_blocks": dict(blocks),
              "elapsed_seconds": monotonic()-started, "peak_process_rss_bytes": rss(),
              "distribution_coverage": dict(source.coverage) if source else {},
              "wrapper_interventions": dict(wrapper.interventions) if wrapper else {},
              "hero_telemetry": [{"coordinates": list(key), "count": value}
                                 for key, value in sorted(telemetry.items())],
              "swap_after": system(["sysctl", "vm.swapusage"]),
              "memory_pressure_after": system(["memory_pressure", "-Q"])}
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "checksums.json").write_text(json.dumps({str(p.relative_to(out)): _hash(p)
        for p in out.rglob("*") if p.is_file() and p.name != "checksums.json"},
        indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--resource-only", action="store_true")
    args = parser.parse_args()
    result = run(json.loads(args.plan.read_text()), args.arm, args.index,
                 args.manifest, args.out, args.deadline, resource_only=args.resource_only)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
