"""Paired, two-position fixed-stack evaluation for versioned HU20 policies."""

import argparse
import json
import resource
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from random import Random
from time import monotonic, perf_counter, time

from src.arena.catalog import Checkpoint
from src.arena.policies import make_policy
from src.arena.runner import _fixed
from src.arena.schedule import Plan, build_schedule, digest, schedule_document
from src.blueprint.abstraction import HU20_SCHEMA, choices, information_key
from src.blueprint.artifact import HU20_FORMAT, FrozenBlueprint
from src.blueprint.solver import HU20_GAME
from src.blueprint.windowed import WindowedDistribution, _hash
from src.game.types import ActionKind


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with path.open("a") as target:
        target.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        target.flush()


def rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def system(command):
    if sys.platform != "darwin":
        return None
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


class UniformHU20:
    def distribution(self, view):
        menu = choices(view, free_fold=False)
        return menu, (1 / len(menu),) * len(menu), False


class Player:
    def __init__(self, source, seed, telemetry=None, reached=None):
        self.source = source
        self.random = Random(seed)
        self.telemetry = telemetry
        self.reached = reached

    def choose_action(self, view):
        started = perf_counter()
        menu, probabilities, trained = self.source.distribution(view)
        if self.reached is not None:
            key = information_key(view, menu, schema=HU20_SCHEMA)
            self.reached[(view.street.value, key)] += 1
        if self.telemetry is not None:
            self.telemetry[(view.street.value, "decisions")] += 1
            self.telemetry[(view.street.value, "trained" if trained else "fallback")] += 1
            if ActionKind.CHECK in view.legal_actions.kinds and menu:
                self.telemetry[(view.street.value, "free_check")] += 1
            if not probabilities or abs(sum(probabilities)-1) > 1e-8:
                raise ValueError("HU20 inference probabilities are not normalized")
        action = self.random.choices(menu, weights=probabilities, k=1)[0].action
        if self.telemetry is not None:
            elapsed = perf_counter()-started
            self.telemetry[(view.street.value, "decision_seconds_sum")] += elapsed
            self.telemetry[(view.street.value, "decision_seconds_max")] = max(
                self.telemetry[(view.street.value, "decision_seconds_max")], elapsed)
        return action


def load_source(arm, training_root):
    if arm == "uniform":
        return UniformHU20(), {"arm": arm, "game": HU20_GAME}
    if arm[0] in "CA" and arm[1:].isdigit():
        seed = int(arm[1:])
        run = training_root / str(seed)
        manifest_path = run / "policy-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("game") != HU20_GAME or manifest.get("abstraction") != HU20_SCHEMA:
            raise ValueError("HU20 extracted artifact identity mismatch")
        source = WindowedDistribution(run / "policy-index.sqlite", manifest, arm[0])
        return source, {"arm": arm, "index_sha256": _hash(run / "policy-index.sqlite"),
                        "manifest_sha256": _hash(manifest_path), "game": HU20_GAME}
    if arm.startswith("E"):
        seed, milestone = arm[1:].split("-")
        path = training_root / seed / f"policy-{milestone}.json.gz"
        spec = Checkpoint(arm, str(path), _hash(path), HU20_FORMAT)
        return FrozenBlueprint(spec, path), {"arm": arm, "policy_sha256": spec.sha256,
                                             "game": HU20_GAME}
    raise ValueError("Unknown HU20 arm")


def run(plan, training_root, arm, opponent, phase, out, deadline, *, blocks=None,
        resource_only=False):
    if out.exists():
        raise FileExistsError(out)
    if phase not in ("development", "confirmation", "crossplay"):
        raise ValueError("Unknown evaluation phase")
    if opponent not in plan["opponents"] and not (opponent[0] in "CAE" and opponent[1:]):
        raise ValueError("Unknown opponent")
    if arm != "uniform" and not (arm[0] in "CAE" and arm[1:]):
        raise ValueError("Unknown candidate")
    out.mkdir(parents=True)
    started = monotonic()
    definition = plan[phase]
    count = (blocks or definition["blocks_per_opponent"])
    if resource_only:
        count = min(count, 8)
    opponent_index = (plan["opponents"].index(opponent) if opponent in plan["opponents"]
                      else 100 + int(opponent[1:].split("-")[0]) % 100)
    schedule_plan = Plan.from_dict({"scenarios": [{"name": "hu20", "stacks": [2000, 2000]}],
        "candidate": "check_call", "baseline": "check_call", "opponents": [opponent],
        "blocks": count, "root_seed": definition["root_seed"]+opponent_index,
        "split": "validation" if phase == "development" else "test",
        "max_decisions": 1000})
    schedule = build_schedule(schedule_plan)
    manifest = {"schema": "hu20-evaluation-v2", "phase": phase, "arm": arm,
        "opponent": opponent, "game": HU20_GAME, "plan_sha256": digest(plan),
        "schedule_sha256": digest(schedule_document(schedule_plan)),
        "resource_only": resource_only, "blocks": count, "started_unix_seconds": time(),
        "swap_before": system(["sysctl", "vm.swapusage"]),
        "memory_pressure_before": system(["memory_pressure", "-Q"])}
    write_json(out / "manifest.json", manifest)
    source = opponent_source = None
    telemetry = Counter()
    reached = Counter()
    attempts = completed_blocks = 0
    stop_reason = None
    try:
        load_start = monotonic()
        source, source_identity = load_source(arm, training_root)
        if opponent == "hu20_uniform":
            opponent_source = UniformHU20()
        elif opponent[0] in "CAE" and opponent[1:]:
            opponent_source, _ = load_source(opponent, training_root)
        manifest["source_identity"] = source_identity
        manifest["load_seconds"] = monotonic()-load_start
        write_json(out / "manifest.json", manifest)
        scenario = schedule_plan.scenarios[0]
        for block in schedule:
            if time() >= deadline:
                raise TimeoutError("HU20 campaign deadline")
            if rss() >= plan["limits"]["max_rss_gib"]*1024**3:
                raise MemoryError("HU20 evaluation RSS ceiling")
            if shutil.disk_usage(out).free < plan["limits"]["min_free_gib"]*1024**3:
                raise RuntimeError("HU20 evaluation disk ceiling")
            for rotation in range(2):
                ids = tuple(f"player-{(seat-rotation)%2}" for seat in range(2))
                trace, timings = {"events": [], "reloads": {}}, []
                row = {"arm": arm, "opponent": opponent, "phase": phase,
                    "block": block.index, "rotation": rotation, "deal_seed": block.deal_seeds[0],
                    "button": block.button, "candidate_chips": None, "status": "started"}
                begin = monotonic()
                try:
                    if opponent_source is not None:
                        rival = Player(opponent_source, block.action_seeds[1])
                    else:
                        rival = make_policy(opponent, block.action_seeds[1])
                    policies = {"player-0": Player(source, block.action_seeds[0], telemetry, reached),
                                "player-1": rival}
                    net, _ = _fixed(scenario, block, rotation, ids, policies, 1000,
                                    trace, timings, f"hu20-{phase}-{opponent}")
                    if sum(net) != 0:
                        raise ValueError("HU20 hand did not conserve chips")
                    row["status"] = "completed"
                    if not resource_only:
                        row["candidate_chips"] = net[rotation]
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
            completed_blocks += 1
            if completed_blocks % 128 == 0:
                append(out / "progress.jsonl", {"blocks": completed_blocks,
                    "elapsed_seconds": monotonic()-started, "rss_bytes": rss()})
    except Exception as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
    finally:
        if isinstance(source, WindowedDistribution):
            source.close()
        if isinstance(opponent_source, WindowedDistribution):
            opponent_source.close()
    result = {"status": "complete" if stop_reason is None else "incomplete",
              "stop_reason": stop_reason, "attempts": attempts,
              "completed_blocks": completed_blocks,
              "peak_process_rss_bytes": rss(), "elapsed_seconds": monotonic()-started,
              "coverage": dict(source.coverage) if isinstance(source, WindowedDistribution) else {},
              "hero_telemetry": [{"coordinates": list(key), "count": value}
                                 for key, value in sorted(telemetry.items())],
              "swap_after": system(["sysctl", "vm.swapusage"]),
              "memory_pressure_after": system(["memory_pressure", "-Q"])}
    write_json(out / "reached.json", [{"street": street, "key": key, "decisions": count}
        for (street, key), count in sorted(reached.items())])
    write_json(out / "result.json", result)
    write_json(out / "checksums.json", {str(path.relative_to(out)): _hash(path)
        for path in out.rglob("*") if path.is_file() and path.name != "checksums.json"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--phase", choices=("development", "confirmation", "crossplay"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--resource-only", action="store_true")
    args = parser.parse_args()
    result = run(json.loads(args.plan.read_text()), args.training_root, args.arm,
                 args.opponent, args.phase, args.out, args.deadline,
                 blocks=args.blocks, resource_only=args.resource_only)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
