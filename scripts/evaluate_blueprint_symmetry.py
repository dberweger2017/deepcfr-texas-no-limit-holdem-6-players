"""Seven-arm paired blueprint lookup and free-fold audit on fixed rotations."""

import argparse
import gc
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
from src.arena.runner import _fixed, public_events
from src.arena.schedule import Plan, build_schedule, digest, schedule_document
from src.blueprint.abstraction import (
    BUTTON_ZERO_COMPAT_LOOKUP, LEGACY_LOOKUP, choices, information_key,
)
from src.blueprint.artifact import load_training
from src.blueprint.lookup import NoFreeFoldDistribution, TableDistribution
from src.game.types import ActionKind

ARMS = ("U", "U_safe", "B_legacy", "B_legacy_safe",
        "B_canonical", "B_canonical_safe", "TAG")
MODES = (LEGACY_LOOKUP, BUTTON_ZERO_COMPAT_LOOKUP)


def _hash(path):
    h = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _swap():
    if sys.platform != "darwin":
        return None
    command = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                             text=True, check=False)
    return command.stdout.strip() if command.returncode == 0 else command.stderr.strip()


def _pressure():
    if sys.platform != "darwin":
        return None
    command = subprocess.run(["memory_pressure", "-Q"], capture_output=True,
                             text=True, check=False)
    return command.stdout.strip() if command.returncode == 0 else command.stderr.strip()


def _append(path, row):
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


class CoverageProbe:
    def __init__(self, trainer):
        self.trainer = trainer
        self.weighted = Counter()
        self.weighted_visits = Counter()
        self.distinct = defaultdict(dict)
        self.same_decision = Counter()
        self.actions = Counter()
        self.decisions = 0

    def record(self, view, arm):
        menu = choices(view, raise_cap=self.trainer.config.raise_cap)
        street, button = view.street.value, view.button
        found = {}
        for mode in MODES:
            key = information_key(view, menu,
                                  schema=self.trainer.config.abstraction,
                                  lookup_mode=mode)
            node = self.trainer.nodes.get(key)
            if node is not None and node.names != tuple(item.name for item in menu):
                raise ValueError("Coverage probe found mismatched action labels")
            group = (arm, street, button, mode)
            visits = node.visits if node is not None else None
            label = "missing" if visits is None else "found"
            self.weighted[(*group, label)] += 1
            if visits is not None:
                self.weighted_visits[(*group, visits)] += 1
            self.distinct[group][key] = visits
            found[mode] = node is not None
        category = ("both" if all(found.values()) else
                    "legacy_only" if found[LEGACY_LOOKUP] else
                    "canonical_only" if found[BUTTON_ZERO_COMPAT_LOOKUP] else "neither")
        self.same_decision[(arm, street, button, category)] += 1
        self.decisions += 1

    def action(self, view, arm, action):
        self.actions[(arm, view.street.value, view.button, action.kind.value)] += 1

    def summary(self):
        groups = sorted(self.distinct)
        rows = []
        for group in groups:
            values = self.distinct[group].values()
            distinct_hist = Counter(value for value in values if value is not None)
            weighted_hist = {str(visits): count
                             for (*prefix, visits), count in self.weighted_visits.items()
                             if tuple(prefix) == group}
            rows.append({
                "arm": group[0], "street": group[1], "button": group[2],
                "lookup_mode": group[3],
                "decision_weighted_found": self.weighted[(*group, "found")],
                "decision_weighted_missing": self.weighted[(*group, "missing")],
                "decision_weighted_visit_histogram": weighted_hist,
                "distinct_keys": len(self.distinct[group]),
                "distinct_found": sum(value is not None for value in values),
                "distinct_missing": sum(value is None for value in values),
                "distinct_found_visit_histogram": {
                    str(visits): count for visits, count in sorted(distinct_hist.items())
                },
            })
        return {
            "decisions": self.decisions,
            "coverage_rows": rows,
            "same_decision_counts": [
                {"arm": arm, "street": street, "button": button,
                 "category": category, "count": count}
                for (arm, street, button, category), count in sorted(self.same_decision.items())
            ],
            "action_counts": [
                {"arm": arm, "street": street, "button": button,
                 "action": action, "count": count}
                for (arm, street, button, action), count in sorted(self.actions.items())
            ],
        }


class AuditedPlayer:
    def __init__(self, arm, source, seed, probe):
        self.arm = arm
        self.source = source
        self.random = Random(seed)
        self.probe = probe
        self.style = make_policy("tight_aggressive", seed) if arm == "TAG" else None

    def choose_action(self, view):
        self.probe.record(view, self.arm)
        if self.style is not None:
            action = self.style.choose_action(view)
        else:
            menu, probabilities, _ = self.source.distribution(view)
            action = self.random.choices(menu, weights=probabilities, k=1)[0].action
        self.probe.action(view, self.arm, action)
        return action


def _guard(plan, out, started):
    if monotonic() - started >= plan["limits"]["max_wall_seconds"]:
        raise TimeoutError("Blueprint audit reached its overall wall limit")
    if _rss() >= plan["limits"]["max_rss_gib"] * 1024**3:
        raise MemoryError("Blueprint audit reached its process RSS limit")
    if shutil.disk_usage(out).free < plan["limits"]["min_free_gib"] * 1024**3:
        raise RuntimeError("Blueprint audit reached its free-disk limit")


def _arena_plan(suite, name):
    return Plan.from_dict({
        "scenarios": [{"name": "six-handed-100bb", "stacks": [10_000] * 6}],
        "candidate": "U", "baseline": "TAG",
        "opponents": suite["opponents"], "blocks": suite["blocks"],
        "root_seed": suite["root_seed"], "split": "validation",
        "max_decisions": suite["max_decisions"],
    })


def run(plan, checkpoint, out):
    if out.exists():
        raise FileExistsError(out)
    if tuple(plan["arms"]) != ARMS:
        raise ValueError("The seven-arm decomposition is frozen")
    if _hash(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < plan["limits"]["min_free_gib"] * 1024**3:
        raise RuntimeError("Insufficient free disk before the audit")
    out.mkdir()
    started = monotonic()
    manifest = {
        "schema": "blueprint-symmetry-audit-run-v1",
        "plan": plan,
        "plan_sha256": digest(plan),
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "source_revision": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain")),
        "environment": environment(),
        "started_unix_seconds": time(),
        "swap_before": _swap(), "memory_pressure_before": _pressure(),
        "lookup_modes": list(MODES),
        "status": "started",
    }
    write_json(out / "manifest.json", manifest)
    attempts = 0
    completed_blocks = Counter()
    stop_reason = None
    phase = "checkpoint_load"
    probe = None
    wrappers = {}
    try:
        trainer = load_training(checkpoint)
        _guard(plan, out, started)
        probe = CoverageProbe(trainer)
        legacy = TableDistribution(trainer)
        canonical = TableDistribution(
            trainer, lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP,
            checkpoint_sha256=plan["checkpoint_sha256"],
        )
        uniform = TableDistribution(trainer, uniform=True)
        sources = {
            "U": uniform,
            "U_safe": NoFreeFoldDistribution(uniform),
            "B_legacy": legacy,
            "B_legacy_safe": NoFreeFoldDistribution(legacy),
            "B_canonical": canonical,
            "B_canonical_safe": NoFreeFoldDistribution(canonical),
            "TAG": None,
        }
        wrappers = {arm: source for arm, source in sources.items()
                    if isinstance(source, NoFreeFoldDistribution)}
        phase = "arena"
        for suite_name, suite in plan["suites"].items():
            arena_plan = _arena_plan(suite, suite_name)
            schedule = build_schedule(arena_plan)
            manifest.setdefault("schedule_sha256", {})[suite_name] = digest(
                schedule_document(arena_plan))
            write_json(out / "manifest.json", manifest)
            scenario = arena_plan.scenarios[0]
            for block in schedule:
                _guard(plan, out, started)
                for rotation in range(6):
                    ids = tuple(f"player-{(seat - rotation) % 6}" for seat in range(6))
                    for arm in ARMS:
                        _guard(plan, out, started)
                        trace = {"events": [], "reloads": {}}
                        timings = []
                        row = {
                            "suite": suite_name, "block": block.index,
                            "rotation": rotation, "arm": arm,
                            "button": block.button, "deal_seed": block.deal_seeds[0],
                            "opponents": list(block.opponents),
                            "status": "started", "candidate_chips": None,
                            "big_blind": scenario.big_blind,
                        }
                        begun = monotonic()
                        try:
                            policies = {"player-0": AuditedPlayer(
                                arm, sources[arm], block.action_seeds[0], probe,
                            )}
                            policies.update({f"player-{index}": make_policy(
                                name, block.action_seeds[index])
                                for index, name in enumerate(block.opponents, 1)})
                            net, participants = _fixed(
                                scenario, block, rotation, ids, policies,
                                arena_plan.max_decisions, trace, timings,
                                f"symmetry-{suite_name}",
                            )
                            if sum(net) != 0:
                                raise RuntimeError("Hand did not conserve chips")
                            if not plan["resource_only"]:
                                row.update(candidate_chips=net[rotation],
                                           net_chips=net, participants=participants)
                            row["status"] = "completed"
                        except Exception as exc:
                            row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                                       events=trace["events"])
                        row["seconds"] = monotonic() - begun
                        row["hero_decisions"] = sum(item["player_id"] == "player-0"
                                                    for item in timings)
                        row["outcome_sha256"] = digest({k: v for k, v in row.items()
                                                        if k != "outcome_sha256"})
                        _append(out / "hands.jsonl", row)
                        attempts += 1
                        if row["status"] != "completed":
                            raise RuntimeError(row["error"])
                completed_blocks[suite_name] += 1
                _append(out / "block-progress.jsonl", {
                    "suite": suite_name, "completed_blocks": completed_blocks[suite_name],
                    "elapsed_seconds": monotonic() - started,
                    "peak_process_rss_bytes": _rss(),
                    "decisions": probe.decisions,
                    "wrapper_interventions": {name: dict(item.interventions)
                                               for name, item in wrappers.items()},
                })
        write_json(out / "reached-decisions.json", probe.summary())
    except Exception as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
        if probe is not None:
            write_json(out / "reached-decisions.json", probe.summary())
    result = {
        "status": "complete" if stop_reason is None else "incomplete",
        "stop_reason": stop_reason,
        "attempts": attempts,
        "completed_blocks": dict(completed_blocks),
        "planned_attempts": sum(suite["blocks"] for suite in plan["suites"].values()) * 6 * len(ARMS),
        "elapsed_seconds": monotonic() - started,
        "peak_process_rss_bytes": _rss(),
        "swap_after": _swap(), "memory_pressure_after": _pressure(),
        "wrapper_interventions": {name: dict(item.interventions)
                                   for name, item in wrappers.items()},
        "decision_count": probe.decisions if probe is not None else 0,
    }
    write_json(out / "result.json", result)
    write_json(out / "checksums.json", {
        str(path.relative_to(out)): _hash(path) for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    })
    del probe
    gc.collect()
    print(json.dumps({key: result[key] for key in
                      ("status", "attempts", "elapsed_seconds", "peak_process_rss_bytes",
                       "stop_reason")}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(json.loads(args.plan.read_text()), args.checkpoint, args.out)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
