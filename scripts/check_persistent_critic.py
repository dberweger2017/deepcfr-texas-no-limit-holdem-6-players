"""Compare a persistent Monte Carlo critic on retained river and full-stack probes."""

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np

from src.arena.artifacts import environment, git, source_fingerprint, write_json
from src.arena.heuristics import STYLES
from src.arena.policies import make_policy
from src.arena.schedule import digest
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street
from src.holdem.checkpoint import load_training
from src.holdem.critic import (
    BaselineProfile,
    CriticConfig,
    PersistentCritic,
    check_limit,
    model_digest,
)
from src.holdem.outcome_sampling import collect_outcome
from src.holdem.river_reference import ReferenceProfile, contexts, enumerate_reference
from src.solver.neural.network import deterministic_cpu

STREETS = (Street.FLOP, Street.TURN, Street.RIVER)


def seed_for(*coordinates):
    return int(digest(coordinates)[:15], 16)


def full_root(seed, *, street=None, button=0):
    table = Table(tuple(f"player-{i}" for i in range(6)), (10000,) * 6, button=button)
    hand = Hand.start(table, hand_id="critic-probe", seed=seed)
    if street is not None:
        while not hand.finished and hand.observe(hand.actor).street != street:
            view = hand.observe(hand.actor)
            kind = (
                ActionKind.CHECK
                if ActionKind.CHECK in view.legal_actions.kinds
                else ActionKind.CALL
            )
            hand = hand.apply(Action(kind))
        if hand.finished:
            raise ValueError("Forced prefix failed to reach the probe street")
    return hand


def street_counts(records):
    return dict(
        sorted(
            Counter(r.candidates.decision.source.street.value for r in records).items()
        )
    )


def train_critic(plan, suite, seed, profiles, roots, out, deadline):
    config = CriticConfig(
        **{
            k: plan[k]
            for k in (
                "width",
                "learning_rate",
                "gradient_clip",
                "replay_capacity",
                "batch_size",
            )
        },
        seed=seed,
    )
    critic = PersistentCritic(config)
    phases = []
    for phase, profile in enumerate(profiles):
        critic.begin_phase(profile, suite)
        before = perf_counter()
        collected = 0
        for root in roots(phase):
            collected += critic.collect(
                root, profile, deadline=deadline, max_nodes=plan["max_tree_nodes"]
            )
        data_seconds = perf_counter() - before
        before = perf_counter()
        losses = critic.fit(plan["fit_steps_per_phase"], deadline=deadline)
        fit_seconds = perf_counter() - before
        path = out / f"{suite}-{seed}-phase-{phase + 1}.pt"
        fingerprint = critic.save(path)
        # Check one further update from both states without changing the measured fit.
        before = perf_counter()
        original = deepcopy(critic)
        restored = PersistentCritic.load(path, fingerprint)
        if original.replay != restored.replay or original.identity != restored.identity:
            raise ValueError("Critic replay recovery mismatch")
        a = original.fit(1, deadline=deadline)
        b = restored.fit(1, deadline=deadline)
        if (
            a != b
            or model_digest(original.model) != model_digest(restored.model)
            or original.random.getstate() != restored.random.getstate()
        ):
            raise ValueError("Critic fitting recovery mismatch")
        phases.append(
            {
                "phase": phase + 1,
                "profile": profile.fingerprint,
                "records_collected": collected,
                "replay_records": len(critic.replay),
                "replay_streets": street_counts(critic.replay),
                "data_seconds": data_seconds,
                "fit_seconds": fit_seconds,
                "mean_minibatch_loss": float(np.mean(losses)),
                "steps_total": critic.steps,
                "model_sha256": model_digest(critic.model),
                "checkpoint": path.name,
                "checkpoint_sha256": fingerprint,
                "checkpoint_bytes": path.stat().st_size,
                "recovery_seconds": perf_counter() - before,
                "recovery_verified": True,
            }
        )
        with (out / "fit-phases.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {"suite": suite, "seed": seed, **phases[-1]}, allow_nan=False
                )
                + "\n"
            )
        print(f"Fit complete: {suite} / seed {seed} / phase {phase + 1}", flush=True)
    return critic, {
        "suite": suite,
        "seed": seed,
        "parameters": sum(p.numel() for p in critic.model.parameters()),
        "phases": phases,
        "training_seconds": sum(p["data_seconds"] + p["fit_seconds"] for p in phases),
    }


def moments(rows, exact=None):
    values = np.asarray([r["regrets_bb"] for r in rows])
    variance = float(np.var(values, axis=0, ddof=1).sum())
    seconds = float(np.mean([r["seconds"] for r in rows]))
    nodes = float(np.mean([r["nodes"] for r in rows]))
    result = {
        "replicates": len(rows),
        "mean_regrets_bb": values.mean(0).tolist(),
        "trace_variance": variance,
        "mean_nodes": nodes,
        "mean_seconds": seconds,
        "variance_times_nodes": variance * nodes,
        "variance_times_seconds": variance * seconds,
    }
    if exact is not None:
        result["mean_error_bb"] = (values.mean(0) - exact.target.regrets_bb).tolist()
        result["reference_mse"] = float(
            np.mean((values - exact.target.regrets_bb) ** 2)
        )
    return result


def sample_context(
    plan, suite, name, seed, worlds, profiles, raw, deadline, exact=None
):
    samples = {arm: [] for arm in profiles}
    for replicate in range(plan["replicates"]):
        action_seed = seed_for(
            plan["root_seed"], "evaluation-actions", suite, name, seed, replicate
        )
        world = Random(seed_for(action_seed, "world")).randrange(len(worlds))
        root = worlds[world]
        paths, values = [], []
        for arm, profile in profiles.items():
            before = perf_counter()
            result = collect_outcome(
                root,
                profile,
                root.actor,
                iteration=1,
                action_seed=action_seed,
                exploration=plan["exploration"],
                baseline="zero" if arm == "zero" else "frozen",
                branch_first=True,
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
                raise ValueError("Missing or incorrectly weighted root target")
            decision = decisions[0]
            row = {
                "suite": suite,
                "context": name,
                "seed": seed,
                "replicate": replicate,
                "world": world,
                "action_seed": action_seed,
                "arm": arm,
                "values_bb": decision.values_bb,
                "regrets_bb": decision.regrets_bb,
                "nodes": result.nodes,
                "seconds": seconds,
                "path_sha256": digest([asdict(e.event) for e in result.executions]),
            }
            samples[arm].append(row)
            raw.write(json.dumps(row, allow_nan=False) + "\n")
            paths.append((row["path_sha256"], result.nodes))
            values.append(decision.values_bb)
        if len(set(paths)) != 1:
            raise ValueError("Changing the baseline changed sampling paths or nodes")
        if (
            suite == "river-reference"
            and name.endswith("facing")
            and len(set(values)) != 1
        ):
            raise ValueError("Baseline changed the final expanded decision")
    raw.flush()
    return [
        {
            "suite": suite,
            "context": name,
            "seed": seed,
            "arm": arm,
            **moments(rows, exact),
        }
        for arm, rows in samples.items()
    ]


def value_errors(context, exact, profiles, seed):
    candidates = exact.target.candidates
    truth = np.asarray(exact.target.values_bb)
    results = []
    for arm, profile in profiles.items():
        prediction = np.asarray(profile.action_values(candidates))
        chosen = int(prediction.argmax())
        results.append(
            {
                "context": context.name,
                "split": context.split,
                "seed": seed,
                "arm": arm,
                "hole_cards": candidates.decision.source.hole_cards,
                "values_bb": prediction.tolist(),
                "reference_values_bb": truth.tolist(),
                "q_mse": float(np.mean((prediction - truth) ** 2)),
                "greedy_action": asdict(candidates.actions[chosen]),
                "greedy_decision_cost_bb": float(truth.max() - truth[chosen]),
            }
        )
    return results


def baseline_profiles(policy, critic, historical):
    return {
        "zero": BaselineProfile(policy, kind="zero"),
        "accounting": BaselineProfile(policy, kind="accounting"),
        "historical": BaselineProfile(policy, kind="historical", historical=historical),
        "learned": BaselineProfile(policy, kind="learned", critic=critic),
    }


def cost_screen(plan, fits, sampling):
    results = []
    for seed in plan["seeds"]:
        comparisons = []
        for street in STREETS:
            suite = f"full-{street.value}"
            fit = next(f for f in fits if f["suite"] == suite and f["seed"] == seed)
            rows = [r for r in sampling if r["suite"] == suite and r["seed"] == seed]
            means = {}
            for arm in ("zero", "accounting", "historical", "learned"):
                selected = [r for r in rows if r["arm"] == arm]
                means[arm] = {
                    "variance_times_seconds": float(
                        np.mean([r["variance_times_seconds"] for r in selected])
                    ),
                    "trace_variance": float(
                        np.mean([r["trace_variance"] for r in selected])
                    ),
                }
            learned = means["learned"]
            overhead = learned["trace_variance"] * fit["training_seconds"]
            amortized = (
                learned["variance_times_seconds"]
                + overhead / plan["amortization_traversals"]
            )
            for comparator in ("accounting", "historical"):
                base = means[comparator]["variance_times_seconds"]
                savings = base - learned["variance_times_seconds"]
                comparisons.append(
                    {
                        "street": street.value,
                        "comparator": comparator,
                        "baseline_cost_variance": base,
                        "learned_cost_variance": amortized,
                        "learned_without_training_cost": learned[
                            "variance_times_seconds"
                        ],
                        "ratio": amortized / base if base else None,
                        "break_even_traversals": overhead / savings
                        if savings > 0
                        else None,
                        "street_pass": amortized <= plan["maximum_street_ratio"] * base,
                    }
                )
        aggregate = []
        for comparator in ("accounting", "historical"):
            selected = [c for c in comparisons if c["comparator"] == comparator]
            base = float(np.mean([c["baseline_cost_variance"] for c in selected]))
            learned = float(np.mean([c["learned_cost_variance"] for c in selected]))
            aggregate.append(
                {
                    "comparator": comparator,
                    "baseline": base,
                    "learned": learned,
                    "ratio": learned / base if base else None,
                    "pass": learned <= plan["required_cost_variance_ratio"] * base
                    and all(c["street_pass"] for c in selected),
                }
            )
        results.append(
            {
                "seed": seed,
                "streets": comparisons,
                "aggregate": aggregate,
                "pass": all(a["pass"] for a in aggregate),
            }
        )
    return {"seeds": results, "pass": all(r["pass"] for r in results)}


def coverage(plan, historical, out, deadline):
    collector, unique, arena = Counter(), set(), Counter()
    with (out / "coverage.jsonl").open("w") as raw:
        for index in range(plan["coverage_hands"]):
            check_limit(deadline)
            root = full_root(
                seed_for(plan["root_seed"], "coverage-deal", index), button=index % 6
            )
            hero = index % 6
            result = collect_outcome(
                root,
                historical,
                hero,
                iteration=1,
                action_seed=seed_for(plan["root_seed"], "coverage-collection", index),
                exploration=plan["exploration"],
                baseline="frozen",
                branch_first=True,
                max_nodes=plan["max_tree_nodes"],
                deadline=deadline,
            )
            local = street_counts(result.decisions)
            collector.update(local)
            for d in result.decisions:
                unique.add((index, d.candidates.decision.source))
            players = [None] * 6
            for seat in range(6):
                action_seed = seed_for(plan["root_seed"], "coverage-arena", index, seat)
                players[seat] = (
                    historical.player(action_seed)
                    if seat == hero
                    else make_policy(tuple(STYLES)[(seat - hero - 1) % 6], action_seed)
                )
            hand, visited, actions = root, Counter(), []
            while not hand.finished:
                check_limit(deadline)
                if len(actions) >= plan["max_tree_nodes"]:
                    raise ValueError("Coverage arena exceeded decision limit")
                view = hand.observe(hand.actor)
                if view.seat == hero:
                    visited[view.street.value] += 1
                action = players[hand.actor].choose_action(view)
                actions.append(
                    {
                        "seat": hand.actor,
                        "street": view.street.value,
                        "action": asdict(action),
                    }
                )
                hand = hand.apply(action)
            final = hand.events[-1].stacks
            if sum(final) != sum(root.table.stacks):
                raise ValueError("Coverage arena settlement failed")
            arena.update(visited)
            raw.write(
                json.dumps(
                    {
                        "hand": index,
                        "hero": hero,
                        "collector": local,
                        "collector_nodes": result.nodes,
                        "collector_path_sha256": digest(
                            [asdict(e.event) for e in result.executions]
                        ),
                        "arena_hero_decisions": dict(visited),
                        "arena_actions": actions,
                        "arena_net_chips": [
                            a - b for a, b in zip(final, root.table.stacks)
                        ],
                    }
                )
                + "\n"
            )
            raw.flush()
    return {
        "hands_per_distribution": plan["coverage_hands"],
        "collector_records": dict(collector),
        "collector_unique_observations": len(unique),
        "arena_hero_decisions": dict(arena),
    }


def run(plan, out):
    started = perf_counter()
    deadline = started + plan["max_seconds"]
    out.mkdir(parents=True, exist_ok=False)
    river_plan = json.loads(Path(plan["river_plan"]).read_text())
    report = {
        "format": plan["format"],
        "plan": plan,
        "river_plan": river_plan,
        "revision": git("rev-parse", "HEAD"),
        "source_sha256": source_fingerprint(),
        "plan_sha256": digest(plan),
        "environment": environment(),
        "status": "running",
        "fits": [],
        "sampling": [],
        "value_errors": [],
        "references": [],
        "full_roots": [],
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
                raise ValueError("Wrong historical profile")
            historical = trainer.current_profile()
            del trainer
            report["historical_profile"] = historical.fingerprint
            retained = contexts(river_plan)
            policy = ReferenceProfile("increasing")
            exact = {}
            for context in retained:
                exact[context.name] = enumerate_reference(
                    context, policy, max_nodes=plan["max_tree_nodes"], deadline=deadline
                )
                reference = exact[context.name]
                report["references"].append(
                    {
                        "context": context.name,
                        "values_bb": reference.target.values_bb,
                        "regrets_bb": reference.target.regrets_bb,
                        "assignments": context.assignments,
                        "nodes": reference.nodes,
                        "seconds": reference.seconds,
                    }
                )
            for seed in plan["seeds"]:
                train = [c for c in retained if c.split == "train"]

                def river_roots(phase, seed=seed, train=train):
                    random = Random(
                        seed_for(plan["root_seed"], "river-training", seed, phase)
                    )
                    for _ in range(plan["river_rollouts_per_phase"]):
                        context = random.choice(train)
                        yield random.choice(context.worlds)

                critic, fit = train_critic(
                    plan,
                    "river-reference",
                    seed,
                    (ReferenceProfile("uniform"), policy),
                    river_roots,
                    out,
                    deadline,
                )
                report["fits"].append(fit)
                profiles = baseline_profiles(policy, critic, historical)
                for context in retained:
                    reference = exact[context.name]
                    arms = {
                        **profiles,
                        "oracle": ReferenceProfile("increasing", reference.baselines),
                    }
                    report["sampling"].extend(
                        sample_context(
                            plan,
                            "river-reference",
                            context.name,
                            seed,
                            context.worlds,
                            arms,
                            raw,
                            deadline,
                            reference,
                        )
                    )
                    report["value_errors"].extend(
                        value_errors(context, reference, profiles, seed)
                    )
                write_json(out / "report.json", report)
                for street in STREETS:
                    suite = f"full-{street.value}"

                    def training_roots(phase, suite=suite, seed=seed, street=street):
                        for index in range(plan["full_rollouts_per_phase"]):
                            yield full_root(
                                seed_for(
                                    plan["root_seed"],
                                    "full-training",
                                    suite,
                                    seed,
                                    phase,
                                    index,
                                ),
                                street=street,
                                button=index % 6,
                            )

                    critic, fit = train_critic(
                        plan,
                        suite,
                        seed,
                        (historical, historical),
                        training_roots,
                        out,
                        deadline,
                    )
                    report["fits"].append(fit)
                    profiles = baseline_profiles(historical, critic, historical)
                    for index in range(plan["full_evaluation_roots_per_street"]):
                        deal_seed = seed_for(
                            plan["root_seed"], "full-evaluation", suite, index
                        )
                        root = full_root(deal_seed, street=street, button=index % 6)
                        name = f"root-{index}"
                        report["full_roots"].append(
                            {
                                "suite": suite,
                                "seed": seed,
                                "context": name,
                                "deal_seed": deal_seed,
                                "root": asdict(root.observe(root.actor)),
                            }
                        )
                        report["sampling"].extend(
                            sample_context(
                                plan,
                                suite,
                                name,
                                seed,
                                (root,),
                                profiles,
                                raw,
                                deadline,
                            )
                        )
                    write_json(out / "report.json", report)
                    print(f"Sampling complete: {suite} / seed {seed}", flush=True)
            report["coverage"] = coverage(plan, historical, out, deadline)
            report["screen"] = cost_screen(plan, report["fits"], report["sampling"])
            report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = perf_counter() - started
        report["expected_fits"] = len(plan["seeds"]) * 4
        report["expected_sampling_cells"] = len(plan["seeds"]) * (
            len(contexts(river_plan)) * 5
            + len(STREETS) * plan["full_evaluation_roots_per_street"] * 4
        )
        report["artifacts"] = {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in out.iterdir()
            if p.name != "report.json" and p.is_file()
        }
        write_json(out / "report.json", report)
    return report


def verify(out):
    report = json.loads((out / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("Only a completed study can pass verification")
    if digest(report["plan"]) != report["plan_sha256"]:
        raise ValueError("Study plan hash mismatch")
    for name, expected in report["artifacts"].items():
        if sha256((out / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    grouped, paired = defaultdict(list), defaultdict(list)
    for line in (out / "samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        grouped[row["suite"], row["context"], row["seed"], row["arm"]].append(row)
        paired[row["suite"], row["context"], row["seed"], row["replicate"]].append(row)
    if len(grouped) != report["expected_sampling_cells"] or len(
        report["sampling"]
    ) != len(grouped):
        raise ValueError("Incomplete sampling cells")
    for rows in paired.values():
        arms = {r["arm"] for r in rows}
        expected = {"zero", "accounting", "historical", "learned"}
        if rows[0]["suite"] == "river-reference":
            expected.add("oracle")
        if (
            arms != expected
            or len(rows) != len(expected)
            or len(
                {
                    (r["path_sha256"], r["nodes"], r["world"], r["action_seed"])
                    for r in rows
                }
            )
            != 1
        ):
            raise ValueError("Paired traversal mismatch")
    for cell in report["sampling"]:
        rows = grouped[cell["suite"], cell["context"], cell["seed"], cell["arm"]]
        if len(rows) != report["plan"]["replicates"] or len(
            {r["replicate"] for r in rows}
        ) != len(rows):
            raise ValueError("Incomplete or duplicate replicates")
        for key, value in moments(rows).items():
            if not np.allclose(value, cell[key], rtol=1e-12, atol=1e-12):
                raise ValueError(f"Sampling summary mismatch: {key}")
    if len(report["fits"]) != report["expected_fits"]:
        raise ValueError("Incomplete critic fits")
    for fit in report["fits"]:
        if len(fit["phases"]) != 2:
            raise ValueError("Incomplete fitting phases")
        for phase in fit["phases"]:
            critic = PersistentCritic.load(
                out / phase["checkpoint"], phase["checkpoint_sha256"]
            )
            if (
                model_digest(critic.model) != phase["model_sha256"]
                or critic.steps != phase["steps_total"]
                or not phase["recovery_verified"]
            ):
                raise ValueError("Checkpoint disagrees with phase report")
    if (
        cost_screen(report["plan"], report["fits"], report["sampling"])
        != report["screen"]
    ):
        raise ValueError("Cost screen does not reproduce")
    return {
        "verified": True,
        "sampling_cells": len(grouped),
        "paired_replicates": len(paired),
        "checkpoints": len(report["fits"]) * 2,
        "screen_pass": report["screen"]["pass"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("configs/holdem/persistent-critic.json")
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
