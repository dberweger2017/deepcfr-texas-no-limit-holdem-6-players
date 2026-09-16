"""Measure extra own-decision branching and coverage across archived profiles."""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np

from scripts.check_persistent_critic import (
    baseline_profiles,
    full_root,
    moments,
    seed_for,
)
from src.arena.artifacts import environment, git, source_fingerprint, write_json
from src.arena.heuristics import STYLES
from src.arena.policies import make_policy
from src.arena.schedule import digest
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street
from src.holdem.checkpoint import load_training
from src.holdem.critic import PersistentCritic, check_limit
from src.holdem.outcome_sampling import collect_outcome
from src.holdem.river_reference import ReferenceProfile, contexts, enumerate_reference
from src.solver.neural.network import deterministic_cpu

POSITIONS = ("button", "small_blind", "big_blind", "utg", "hijack", "cutoff")
STREETS = (Street.FLOP, Street.TURN, Street.RIVER)


def select_profiles(trainer, iterations):
    result = {}
    for iteration in iterations:
        if type(iteration) is not int or not 0 <= iteration <= trainer.iteration:
            raise ValueError("Requested profile is outside the saved training history")
        profile = (
            trainer.current_profile()
            if iteration == trainer.iteration
            else trainer._state.archive[iteration]
        )
        if (
            iteration
            and profile.fingerprint != trainer.reports[iteration - 1].fitted_profile
        ):
            raise ValueError("Archive profile does not match the completed fit")
        if (
            iteration < trainer.iteration
            and profile.fingerprint != trainer.reports[iteration].collection_profile
        ):
            raise ValueError("Archive profile does not match the next collection")
        profile.assert_unchanged()
        result[iteration] = profile
    return result


def sample_cell(
    plan, suite, name, seed, worlds, profiles, raw, deadline, reference=None
):
    count = (
        plan["river_replicates"] if reference is not None else plan["full_replicates"]
    )
    arms = [(depth, arm) for depth in (1, 2) for arm in profiles]
    samples = {arm: [] for arm in arms}
    for replicate in range(count):
        action_seed = seed_for(plan["root_seed"], suite, name, seed, replicate)
        world = Random(seed_for(action_seed, "world")).randrange(len(worlds))
        root = worlds[world]
        paired = defaultdict(list)
        order = arms[replicate % len(arms) :] + arms[: replicate % len(arms)]
        for depth, arm in order:
            before = perf_counter()
            result = collect_outcome(
                root,
                profiles[arm],
                root.actor,
                iteration=1,
                action_seed=action_seed,
                exploration=plan["exploration"],
                baseline="zero" if arm == "zero" else "frozen",
                branch_first=True,
                branch_second=depth == 2,
                max_nodes=plan["max_tree_nodes"],
                deadline=deadline,
            )
            seconds = perf_counter() - before
            decisions = [
                d
                for d in result.decisions
                if d.candidates.decision.source == result.root
            ]
            if len(decisions) != 1 or decisions[0].own_sample_reach != 1:
                raise ValueError(
                    "Missing root target or incorrect own-prefix correction"
                )
            row = {
                "suite": suite,
                "context": name,
                "seed": seed,
                "replicate": replicate,
                "action_seed": action_seed,
                "world": world,
                "arm": arm,
                "depth": depth,
                "regrets_bb": decisions[0].regrets_bb,
                "values_bb": decisions[0].values_bb,
                "nodes": result.nodes,
                "seconds": seconds,
                "path_sha256": digest([asdict(e.event) for e in result.executions]),
            }
            raw.write(json.dumps(row, allow_nan=False) + "\n")
            samples[depth, arm].append(row)
            paired[depth].append(row)
        for rows in paired.values():
            if len({(r["path_sha256"], r["nodes"]) for r in rows}) != 1:
                raise ValueError("Changing the baseline changed the sampled path")
        if (
            suite == "river-reference"
            and name.endswith("facing")
            and (
                len({tuple(r["values_bb"]) for rows in paired.values() for r in rows})
                != 1
            )
        ):
            raise ValueError("A final expanded decision changed across arms")
        raw.flush()
    return [
        {
            "suite": suite,
            "context": name,
            "seed": seed,
            "arm": arm,
            "depth": depth,
            **moments(rows, reference),
        }
        for (depth, arm), rows in samples.items()
    ]


def summarize_sampling(plan, cells):
    output = []
    for seed in plan["sampling_seeds"]:
        metrics = []
        for metric in ("variance_times_nodes", "variance_times_seconds"):
            streets = []
            for street in STREETS:
                values = [
                    float(
                        np.mean(
                            [
                                c[metric]
                                for c in cells
                                if c["seed"] == seed
                                and c["suite"] == f"full-{street.value}"
                                and c["arm"] == "historical"
                                and c["depth"] == depth
                            ]
                        )
                    )
                    for depth in (1, 2)
                ]
                control, candidate = values
                streets.append(
                    {
                        "street": street.value,
                        "control": control,
                        "candidate": candidate,
                        "ratio": candidate / control if control else None,
                        "pass": candidate <= plan["maximum_street_ratio"] * control,
                    }
                )
            control, candidate = (
                float(np.mean([s[key] for s in streets]))
                for key in ("control", "candidate")
            )
            metrics.append(
                {
                    "metric": metric,
                    "streets": streets,
                    "control": control,
                    "candidate": candidate,
                    "ratio": candidate / control if control else None,
                    "pass": control > 0
                    and candidate <= plan["required_ratio"] * control
                    and all(s["pass"] for s in streets),
                }
            )
        output.append(
            {"seed": seed, "metrics": metrics, "pass": all(m["pass"] for m in metrics)}
        )
    return {"pass": all(s["pass"] for s in output), "seeds": output}


def natural_hand(root, profile, plan, deadline, coordinates, hero=None):
    players = []
    for seat in range(6):
        seed = seed_for(plan["root_seed"], "natural-actions", coordinates, hero, seat)
        players.append(
            profile.player(seed)
            if hero is None or seat == hero
            else make_policy(tuple(STYLES)[(seat - hero - 1) % 6], seed)
        )
    hand, actions = root, []
    while not hand.finished:
        check_limit(deadline)
        if len(actions) >= plan["max_tree_nodes"]:
            raise RuntimeError("Natural coverage hand exceeded its decision budget")
        view = hand.observe(hand.actor)
        next_hand = hand.apply(players[hand.actor].choose_action(view))
        event = next_hand.events[len(hand.events)]
        actions.append(
            {
                **asdict(event),
                "stack_before": view.players[view.seat].stack,
                "all_in": event.paid > 0
                and event.paid == view.players[view.seat].stack,
            }
        )
        hand = next_hand
    final = hand.events[-1].stacks
    if sum(final) != sum(root.table.stacks):
        raise ValueError("Natural coverage hand did not settle to zero sum")
    return {
        "actions": actions,
        "final_stacks": final,
        "net_chips": tuple(a - b for a, b in zip(final, root.table.stacks)),
        "public_events": [
            {"event": type(e).__name__, **asdict(e)} for e in hand.events
        ],
    }


def natural_view(result, hero):
    counts = Counter(a["street"] for a in result["actions"] if a["seat"] == hero)
    return {
        "streets": dict(counts),
        "unique_observations": sum(counts.values()),
        "postflop": any(s != "preflop" for s in counts),
        "hero_preflop_all_in": any(
            a["seat"] == hero and a["street"] == "preflop" and a["all_in"]
            for a in result["actions"]
        ),
        "any_preflop_all_in": any(
            a["street"] == "preflop" and a["all_in"] for a in result["actions"]
        ),
        **result,
    }


def collected_view(result):
    records = [
        {
            "street": d.candidates.decision.source.street.value,
            "history_sha256": digest(
                [asdict(e) for e in d.candidates.decision.source.history]
            ),
            "expanded": d.sampled_action is None,
            "own_sample_reach": d.own_sample_reach,
        }
        for d in result.decisions
    ]
    counts = Counter(r["street"] for r in records)
    executions = [
        {
            "event": asdict(e.event),
            "history_sha256": digest(
                [asdict(h) for h in e.candidates.decision.source.history]
            ),
            "stack_before": e.candidates.decision.source.players[e.event.seat].stack,
            "all_in": e.event.paid > 0
            and e.event.paid
            == e.candidates.decision.source.players[e.event.seat].stack,
        }
        for e in result.executions
    ]
    return {
        "streets": dict(counts),
        "unique_observations": len({r["history_sha256"] for r in records}),
        "postflop": any(s != "preflop" for s in counts),
        "nodes": result.nodes,
        "expanded_decisions": sum(r["expanded"] for r in records),
        "records": records,
        "executions": executions,
        "all_in_executions": sum(e["all_in"] for e in executions),
    }


def summarize_coverage(rows):
    result = []
    for iteration in sorted({r["iteration"] for r in rows}):
        for position in POSITIONS:
            cases = [
                r
                for r in rows
                if r["iteration"] == iteration and r["position"] == position
            ]
            for mode in ("first", "second", "selfplay", "styles"):
                counts = Counter()
                for row in cases:
                    counts.update(row[mode]["streets"])
                item = {
                    "iteration": iteration,
                    "position": position,
                    "mode": mode,
                    "hero_hand_cases": len(cases),
                    "streets": dict(counts),
                    "cases_with_postflop": sum(r[mode]["postflop"] for r in cases),
                    "unique_observations_sum": sum(
                        r[mode]["unique_observations"] for r in cases
                    ),
                }
                if mode in ("selfplay", "styles"):
                    item["hero_preflop_all_in_cases"] = sum(
                        r[mode]["hero_preflop_all_in"] for r in cases
                    )
                    item["any_preflop_all_in_cases"] = sum(
                        r[mode]["any_preflop_all_in"] for r in cases
                    )
                else:
                    item["nodes"] = sum(r[mode]["nodes"] for r in cases)
                    item["all_in_executions"] = sum(
                        r[mode]["all_in_executions"] for r in cases
                    )
                    if mode == "second":
                        item["cases_with_extra_expansion"] = sum(
                            r[mode]["expanded_decisions"] > 1 for r in cases
                        )
                result.append(item)
    return result


def coverage(plan, profiles, out, deadline):
    rows = []
    with (out / "coverage.jsonl").open("w") as raw:
        for iteration, profile in profiles.items():
            for button in range(6):
                for block in range(plan["coverage_blocks"]):
                    deal_seed = seed_for(
                        plan["root_seed"], "coverage-deal", button, block
                    )
                    root = full_root(deal_seed, button=button)
                    selfplay = natural_hand(
                        root, profile, plan, deadline, (iteration, button, block)
                    )
                    for hero in range(6):
                        row = {
                            "iteration": iteration,
                            "profile": profile.fingerprint,
                            "button": button,
                            "block": block,
                            "hero": hero,
                            "position": POSITIONS[(hero - button) % 6],
                            "deal_seed": deal_seed,
                            "hand_id": root.observe(hero).hand_id,
                            "table": {
                                key: getattr(root.table, key)
                                for key in (
                                    "player_ids",
                                    "stacks",
                                    "button",
                                    "small_blind",
                                    "big_blind",
                                    "chip_unit",
                                )
                            },
                        }
                        for depth, label in ((1, "first"), (2, "second")):
                            before = perf_counter()
                            result = collect_outcome(
                                root,
                                profile,
                                hero,
                                iteration=1,
                                action_seed=seed_for(
                                    plan["root_seed"],
                                    "coverage-actions",
                                    iteration,
                                    button,
                                    block,
                                    hero,
                                ),
                                exploration=plan["exploration"],
                                baseline="frozen",
                                branch_first=True,
                                branch_second=depth == 2,
                                max_nodes=plan["max_tree_nodes"],
                                deadline=deadline,
                            )
                            row[label] = {
                                "seconds": perf_counter() - before,
                                **collected_view(result),
                            }
                        row["selfplay"] = natural_view(selfplay, hero)
                        row["styles"] = natural_view(
                            natural_hand(
                                root,
                                profile,
                                plan,
                                deadline,
                                (iteration, button, block),
                                hero,
                            ),
                            hero,
                        )
                        raw.write(json.dumps(row, allow_nan=False) + "\n")
                        raw.flush()
                        rows.append(row)
            write_json(out / "coverage-progress.json", summarize_coverage(rows))
            print(f"Coverage complete: after fit {iteration}", flush=True)
    return summarize_coverage(rows)


def run(plan, out):
    started = perf_counter()
    deadline = started + plan["max_seconds"]
    out.mkdir(parents=True, exist_ok=False)
    river_plan = json.loads(Path(plan["river_plan"]).read_text())
    report = {
        "format": plan["format"],
        "plan": plan,
        "plan_sha256": digest(plan),
        "river_plan": river_plan,
        "revision": git("rev-parse", "HEAD"),
        "source_sha256": source_fingerprint(),
        "environment": environment(),
        "status": "running",
        "sampling": [],
        "references": [],
        "coverage": [],
        "profiles": {},
    }
    write_json(out / "report.json", report)
    try:
        with deterministic_cpu(), (out / "samples.jsonl").open("w") as raw:
            trainer = load_training(
                Path(plan["historical_checkpoint"]),
                plan["historical_sha256"],
                manifest=json.loads(Path(plan["historical_manifest"]).read_text()),
            )
            if trainer.iteration != 256 or trainer.config.seed != 307:
                raise ValueError("Wrong historical training checkpoint")
            profiles = select_profiles(trainer, plan["coverage_iterations"])
            historical = trainer.current_profile()
            del trainer
            report["profiles"] = {k: p.fingerprint for k, p in profiles.items()}
            report["coverage"] = coverage(plan, profiles, out, deadline)
            write_json(out / "report.json", report)
            del profiles
            policy = ReferenceProfile("increasing")
            spec = plan["critics"]["river-reference"]
            critic = PersistentCritic.load(Path(spec["path"]), spec["sha256"])
            arms = baseline_profiles(policy, critic, historical)
            for context in contexts(river_plan):
                reference = enumerate_reference(
                    context, policy, max_nodes=plan["max_tree_nodes"], deadline=deadline
                )
                report["references"].append(
                    {
                        "context": context.name,
                        "values_bb": reference.target.values_bb,
                        "regrets_bb": reference.target.regrets_bb,
                        "nodes": reference.nodes,
                        "seconds": reference.seconds,
                    }
                )
                reference_arms = {
                    **arms,
                    "oracle": ReferenceProfile("increasing", reference.baselines),
                }
                for seed in plan["sampling_seeds"]:
                    report["sampling"].extend(
                        sample_cell(
                            plan,
                            "river-reference",
                            context.name,
                            seed,
                            context.worlds,
                            reference_arms,
                            raw,
                            deadline,
                            reference,
                        )
                    )
                write_json(out / "report.json", report)
            for street in STREETS:
                suite = f"full-{street.value}"
                spec = plan["critics"][suite]
                critic = PersistentCritic.load(Path(spec["path"]), spec["sha256"])
                arms = baseline_profiles(historical, critic, historical)
                for index in range(plan["full_roots_per_street"]):
                    deal_seed = seed_for(
                        plan["probe_plan"]["root_seed"], "full-evaluation", suite, index
                    )
                    root = full_root(deal_seed, street=street, button=index % 6)
                    for seed in plan["sampling_seeds"]:
                        report["sampling"].extend(
                            sample_cell(
                                plan,
                                suite,
                                f"root-{index}",
                                seed,
                                (root,),
                                arms,
                                raw,
                                deadline,
                            )
                        )
                    write_json(out / "report.json", report)
                print(f"Probe comparison complete: {suite}", flush=True)
            check_limit(deadline)
            report["screen"] = summarize_sampling(plan, report["sampling"])
            report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = perf_counter() - started
        report["expected_cells"] = len(plan["sampling_seeds"]) * (
            len(contexts(river_plan)) * 10
            + len(STREETS) * plan["full_roots_per_street"] * 8
        )
        report["expected_coverage_cases"] = (
            len(plan["coverage_iterations"]) * 36 * plan["coverage_blocks"]
        )
        report["artifacts"] = {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in out.iterdir()
            if p.is_file() and p.name != "report.json"
        }
        write_json(out / "report.json", report)
    return report


def verify_natural(row, mode):
    saved = row[mode]
    hand = Hand.start(
        Table(**row["table"]), hand_id=row["hand_id"], seed=row["deal_seed"]
    )
    for action in saved["actions"]:
        view = hand.observe(hand.actor)
        if (
            view.seat != action["seat"]
            or view.street != action["street"]
            or view.players[view.seat].stack != action["stack_before"]
        ):
            raise ValueError("Natural action observation does not reproduce")
        chosen = Action(
            ActionKind(action["action"]["kind"]), action["action"]["raise_to"]
        )
        following = hand.apply(chosen)
        event = following.events[len(hand.events)]
        if event.paid != action["paid"] or action["all_in"] != (
            event.paid > 0 and event.paid == action["stack_before"]
        ):
            raise ValueError("Natural payment or all-in classification is wrong")
        hand = following
    if not hand.finished or list(hand.events[-1].stacks) != saved["final_stacks"]:
        raise ValueError("Natural hand settlement does not reproduce")
    if [a - b for a, b in zip(saved["final_stacks"], row["table"]["stacks"])] != saved[
        "net_chips"
    ] or sum(saved["net_chips"]) != 0:
        raise ValueError("Natural hand chip accounting is wrong")
    events = [{"event": type(e).__name__, **asdict(e)} for e in hand.events]
    if digest(events) != digest(saved["public_events"]):
        raise ValueError("Natural public history does not reproduce")
    derived = natural_view(
        {
            key: saved[key]
            for key in ("actions", "final_stacks", "net_chips", "public_events")
        },
        row["hero"],
    )
    for key in (
        "streets",
        "unique_observations",
        "postflop",
        "hero_preflop_all_in",
        "any_preflop_all_in",
    ):
        if derived[key] != saved[key]:
            raise ValueError("Natural visitation summary is wrong")


def verify(out):
    report = json.loads((out / "report.json").read_text())
    if (
        report["status"] != "completed"
        or digest(report["plan"]) != report["plan_sha256"]
    ):
        raise ValueError("A complete unaltered study is required")
    for name, expected in report["artifacts"].items():
        if sha256((out / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Artifact hash mismatch")
    grouped, paired = defaultdict(list), defaultdict(list)
    for line in (out / "samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        grouped[
            row["suite"], row["context"], row["seed"], row["depth"], row["arm"]
        ].append(row)
        paired[
            row["suite"], row["context"], row["seed"], row["replicate"], row["depth"]
        ].append(row)
    if len(grouped) != report["expected_cells"] or len(grouped) != len(
        report["sampling"]
    ):
        raise ValueError("Incomplete sampling cells")
    for rows in paired.values():
        expected = {"zero", "accounting", "historical", "learned"}
        if rows[0]["suite"] == "river-reference":
            expected.add("oracle")
        if (
            {r["arm"] for r in rows} != expected
            or len(rows) != len(expected)
            or len(
                {
                    (r["path_sha256"], r["nodes"], r["world"], r["action_seed"])
                    for r in rows
                }
            )
            != 1
        ):
            raise ValueError("Paired baseline paths differ")
    for cell in report["sampling"]:
        rows = grouped[
            cell["suite"], cell["context"], cell["seed"], cell["depth"], cell["arm"]
        ]
        count = report["plan"][
            "river_replicates"
            if cell["suite"] == "river-reference"
            else "full_replicates"
        ]
        if len(rows) != count or {r["replicate"] for r in rows} != set(range(count)):
            raise ValueError("Incomplete replicate schedule")
        for key, value in moments(rows).items():
            if not np.allclose(value, cell[key], rtol=1e-12, atol=1e-12):
                raise ValueError("Sampling moments do not reproduce")
    coverage_rows = [
        json.loads(line) for line in (out / "coverage.jsonl").read_text().splitlines()
    ]
    expected_cases = {
        (i, b, k, h)
        for i in report["plan"]["coverage_iterations"]
        for b in range(6)
        for k in range(report["plan"]["coverage_blocks"])
        for h in range(6)
    }
    if {
        (r["iteration"], r["button"], r["block"], r["hero"]) for r in coverage_rows
    } != expected_cases:
        raise ValueError("Coverage schedule does not match the plan")
    for row in coverage_rows:
        if (
            row["position"] != POSITIONS[(row["hero"] - row["button"]) % 6]
            or row["profile"] != report["profiles"][str(row["iteration"])]
        ):
            raise ValueError("Coverage position or profile is wrong")
        for mode in ("selfplay", "styles"):
            verify_natural(row, mode)
        for mode in ("first", "second"):
            view = row[mode]
            counts = Counter(r["street"] for r in view["records"])
            if (
                dict(counts) != view["streets"]
                or len(view["executions"]) + 1 != view["nodes"]
            ):
                raise ValueError("Collector counts disagree with raw records")
            if any(
                e["all_in"]
                != (e["event"]["paid"] > 0 and e["event"]["paid"] == e["stack_before"])
                for e in view["executions"]
            ):
                raise ValueError("Collector all-in classification is wrong")
    if (
        len(coverage_rows) != report["expected_coverage_cases"]
        or summarize_coverage(coverage_rows) != report["coverage"]
    ):
        raise ValueError("Coverage summary does not reproduce")
    if summarize_sampling(report["plan"], report["sampling"]) != report["screen"]:
        raise ValueError("Cost screen does not reproduce")
    return {
        "verified": True,
        "cells": len(grouped),
        "coverage_cases": len(coverage_rows),
        "screen_pass": report["screen"]["pass"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("configs/holdem/collector-branching.json")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
    else:
        run(json.loads(args.plan.read_text()), args.out)


if __name__ == "__main__":
    main()
