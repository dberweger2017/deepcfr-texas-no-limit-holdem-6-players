"""Paired three-position TP20 evaluation; independent rival random streams."""

import argparse
import json
from collections import Counter
from pathlib import Path
from random import Random
from time import monotonic, perf_counter, time

from scripts.tp20_common import (append, guard, interruptible, rss, schedule, seal,
                                  system, validate, write_json)
from src.arena.catalog import Checkpoint
from src.arena.policies import make_policy
from src.arena.runner import _fixed
from src.arena.schedule import digest
from src.blueprint.abstraction import TP20_SCHEMA, choices, information_key
from src.blueprint.artifact import TP20_FORMAT, FrozenBlueprint
from src.blueprint.solver import TP20_GAME
from src.blueprint.windowed import _hash


class UniformTP20:
    def distribution(self, view):
        menu = choices(view, free_fold=False)
        information_key(view, menu, schema=TP20_SCHEMA)
        return menu, (1/len(menu),)*len(menu), False


class Player:
    def __init__(self, source, seed, telemetry=None, reached=None):
        self.source, self.random = source, Random(seed)
        self.telemetry, self.reached = telemetry, reached

    def choose_action(self, view):
        started = perf_counter()
        menu, probabilities, trained = self.source.distribution(view)
        if not probabilities or abs(sum(probabilities)-1) > 1e-8:
            raise ValueError("TP20 probabilities are not normalized")
        if self.reached is not None:
            self.reached[(view.street.value, information_key(view, menu, schema=TP20_SCHEMA))] += 1
        action = self.random.choices(menu, weights=probabilities, k=1)[0].action
        view.legal_actions.validate(action)
        if self.telemetry is not None:
            street = view.street.value
            self.telemetry[(street, "decisions")] += 1
            self.telemetry[(street, "trained" if trained else "fallback")] += 1
            elapsed = perf_counter()-started
            self.telemetry[(street, "seconds_sum")] += elapsed
            self.telemetry[(street, "seconds_max")] = max(
                self.telemetry[(street, "seconds_max")], elapsed)
            upper = next((b for b in (.0001,.00025,.0005,.001,.0025,.005,.01,.025,.05,.1,.25,.5,1,5)
                          if elapsed <= b), float("inf"))
            self.telemetry[(street, f"latency_bin_upper_{upper}")] += 1
        return action


def load_source(arm, training_root):
    if arm in ("uniform", "tp20_uniform"):
        return UniformTP20(), {"arm": arm, "game": TP20_GAME}
    if arm.startswith("C") and arm[1:].isdigit():
        path = training_root / arm[1:] / "current.json.gz"
    elif arm.startswith("E"):
        seed, index = arm[1:].split("-")
        path = training_root / seed / f"policy-{int(index)}.json.gz"
    else:
        raise ValueError("Unknown TP20 current-policy arm")
    spec = Checkpoint(arm, str(path), _hash(path), TP20_FORMAT)
    source = FrozenBlueprint(spec, path)
    if source.abstraction != TP20_SCHEMA or source.description["strategy"] != "current":
        raise ValueError("TP20 candidate must use fixed current extraction")
    return source, {"arm": arm, "game": TP20_GAME, "policy_sha256": spec.sha256,
                    "path": str(path), "training_seed": source.description["training_seed"]}


def run(plan, training_root, arm, lineup, phase, out, deadline, *, blocks=None,
        resource_only=False):
    validate(plan)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started = monotonic()
    scenario, schedule_blocks, document = schedule(plan, phase, lineup, blocks)
    write_json(out / "schedule.json", document)
    manifest = {"schema": "tp20-evaluation-v1", "phase": phase, "arm": arm,
        "lineup": lineup, "game": TP20_GAME, "plan_sha256": digest(plan),
        "schedule_sha256": digest(document), "resource_only": resource_only,
        "blocks": len(schedule_blocks), "started_unix_seconds": time(),
        "swap_before": system(["sysctl", "vm.swapusage"]),
        "memory_pressure_before": system(["memory_pressure", "-Q"])}
    write_json(out / "manifest.json", manifest)
    telemetry, reached = Counter(), Counter()
    attempts = completed_blocks = 0
    stop_reason = None
    try:
        source, identity = load_source(arm, training_root)
        opponents = {}
        opponent_identities = {}
        for name in sorted({o for b in schedule_blocks for o in b.opponents}):
            if name == "tp20_uniform" or name.startswith(("C", "E")):
                opponents[name], opponent_identities[name] = load_source(name, training_root)
        manifest.update(source_identity=identity, opponent_identities=opponent_identities,
                        load_seconds=monotonic()-started)
        write_json(out / "manifest.json", manifest)
        for block in schedule_blocks:
            guard(plan, out, deadline)
            for rotation in range(3):
                ids = tuple(f"player-{(seat-rotation)%3}" for seat in range(3))
                trace, timings = {"events": [], "reloads": {}}, []
                row = {"arm": arm, "lineup": lineup, "phase": phase,
                    "block": block.index, "rotation": rotation,
                    "deal_seed": block.deal_seeds[0], "button": block.button,
                    "opponents": block.opponents, "action_seeds": block.action_seeds,
                    "candidate_chips": None, "status": "started"}
                begin = monotonic()
                try:
                    policies = {"player-0": Player(source, block.action_seeds[0], telemetry, reached)}
                    for index, name in enumerate(block.opponents, 1):
                        policies[f"player-{index}"] = (
                            Player(opponents[name], block.action_seeds[index]) if name in opponents
                            else make_policy(name, block.action_seeds[index]))
                    net, _ = _fixed(scenario, block, rotation, ids, policies, 1000,
                                    trace, timings, f"tp20-{phase}-{lineup}")
                    if sum(net) != 0:
                        raise ValueError("TP20 native hand did not conserve chips")
                    row["status"] = "completed"
                    if not resource_only:
                        row.update(candidate_chips=net[rotation], net_chips=net)
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                               public_events=trace["events"])
                row.update(seconds=monotonic()-begin,
                           public_events_sha256=digest(trace["events"]),
                           hero_decisions=sum(t["player_id"] == "player-0" for t in timings))
                append(out / "hands.jsonl", row)
                attempts += 1
                if row["status"] != "completed":
                    raise RuntimeError(row["error"])
            completed_blocks += 1
            if completed_blocks % 128 == 0:
                append(out / "progress.jsonl", {"blocks": completed_blocks,
                    "elapsed_seconds": monotonic()-started, "rss_bytes": rss()})
        guard(plan, out, deadline)
    except Exception as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
    result = {"status": "complete" if stop_reason is None else "incomplete",
        "stop_reason": stop_reason, "attempts": attempts, "completed_blocks": completed_blocks,
        "elapsed_seconds": monotonic()-started, "peak_process_rss_bytes": rss(),
        "hero_telemetry": [{"coordinates": list(k), "count": v} for k,v in sorted(telemetry.items())],
        "swap_after": system(["sysctl", "vm.swapusage"]),
        "memory_pressure_after": system(["memory_pressure", "-Q"])}
    write_json(out / "reached.json", [{"street": s, "key": k, "decisions": c}
        for (s,k),c in sorted(reached.items())])
    write_json(out / "result.json", result)
    seal(out)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "training-root", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("arm", "lineup", "phase"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--blocks", type=int)
    p.add_argument("--resource-only", action="store_true")
    a = p.parse_args()
    interruptible()
    result = run(json.loads(a.plan.read_text()), a.training_root, a.arm, a.lineup, a.phase,
                 a.out, a.deadline, blocks=a.blocks, resource_only=a.resource_only)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
