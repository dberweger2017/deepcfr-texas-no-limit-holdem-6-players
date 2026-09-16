"""Bounded training and frozen-arena reports with independent save/evaluation cadence."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

from src.arena.artifacts import environment, git, source_fingerprint, write_json
from src.arena.policies import make_policy
from src.arena.report import performance, summarize
from src.arena.runner import run_schedule
from src.arena.schedule import Plan, Scenario, build_schedule, canonical, digest
from src.game.hand import Table
from src.game.observation import RULES_PROFILE, SCHEMA_VERSION
from src.holdem.checkpoint import (
    CONTRACT,
    load_policy,
    load_training,
    save_policy,
    save_training,
)
from src.holdem.collection import collection_seed
from src.holdem.fitting import FitConfig
from src.holdem.training import HoldemTrainer, SampledTrainConfig, TrainConfig
from src.solver.neural.checkpoint import atomic_write
from src.solver.neural.network import deterministic_cpu

FORMAT = "holdem-baseline-experiment-v1"


@dataclass(frozen=True)
class Experiment:
    seeds: tuple[int, ...]
    scenarios: tuple[Scenario, ...]
    training: TrainConfig
    iterations: int
    save_every: int
    evaluate_every: int
    blocks: int
    evaluation_seed: int
    opponents: tuple[str, ...]
    max_seconds: float

    def __post_init__(self):
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(s) is not int or s < 0 for s in self.seeds)
        ):
            raise ValueError("Provide distinct nonnegative training seeds")
        if not self.scenarios or len({s.name for s in self.scenarios}) != len(
            self.scenarios
        ):
            raise ValueError("Provide distinct training scenarios")
        if any(
            type(n) is not int or n < 1
            for n in (
                self.iterations,
                self.save_every,
                self.evaluate_every,
                self.blocks,
            )
        ) or not 0 < self.max_seconds < float("inf"):
            raise ValueError("Invalid experiment budgets or schedules")
        if self.training.seed != 0:
            raise ValueError("Training seed is assigned by the independent seed list")
        if any(
            s.mode != "fixed" or not 4 <= len(s.stacks) <= 6 for s in self.scenarios
        ):
            raise ValueError(
                "Training scenarios require four to six fixed-stack players"
            )
        for name in self.opponents:
            make_policy(name, 0)
        for scenario in self.scenarios:
            plan = self.arena(scenario)
            evaluation_deals = {d for b in build_schedule(plan) for d in b.deal_seeds}
            training_deals = {
                collection_seed(seed, iteration, role, sample, "deal")
                for seed in self.seeds
                for iteration in range(1, self.iterations + 1)
                for role in range(len(scenario.stacks))
                for sample in range(self.training.traversals_per_player)
            }
            if evaluation_deals & training_deals:
                raise ValueError("Training and evaluation deal schedules overlap")

    def arena(self, scenario):
        return Plan(
            (scenario,),
            candidate="snapshot_average",
            baseline="uniform_candidates",
            opponents=self.opponents,
            blocks=self.blocks,
            root_seed=self.evaluation_seed,
            split="validation",
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            **{
                **data,
                "seeds": tuple(data["seeds"]),
                "opponents": tuple(data["opponents"]),
                "scenarios": tuple(Scenario(**s) for s in data["scenarios"]),
                "training": (
                    SampledTrainConfig if "sampler" in data["training"] else TrainConfig
                )(
                    **{
                        **data["training"],
                        "fit": FitConfig(**data["training"]["fit"]),
                    }
                ),
            }
        )


def manifest(plan):
    return {
        "format": FORMAT,
        "plan": asdict(plan),
        "contract": CONTRACT,
        "source_sha256": source_fingerprint(),
        "environment": environment(),
        "rules": RULES_PROFILE,
        "observation_schema": SCHEMA_VERSION,
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "purpose": "implementation report; no model promotion or readiness claim",
    }


def _verify_manifest(saved, current):
    for key in current:
        if key not in ("revision", "dirty") and canonical(saved[key]) != canonical(
            current[key]
        ):
            raise ValueError(f"Experiment reproduction mismatch: {key}")


def evaluate(trainer, scenario, plan, out, provenance, deadline):
    from src.holdem.average import AveragePolicy
    from src.holdem.policy import FrozenProfile

    artifact = out / f"average-{trainer.iteration}.pt"
    artifact_hash = save_policy(trainer, artifact, manifest=provenance)
    policy, _ = load_policy(artifact, artifact_hash)
    uniform = AveragePolicy((FrozenProfile([None] * trainer.table.capacity),))
    arena = plan.arena(scenario)
    rows, timings = [], []

    class BoundedPolicy:
        def __init__(self, inner):
            self.inner = inner

        def choose_action(self, view):
            if perf_counter() >= deadline:
                raise TimeoutError("Experiment deadline expired during evaluation")
            return self.inner.choose_action(view)

    def factory(name, seed):
        if name == arena.candidate:
            selected = policy.player(seed)
        elif name == arena.baseline:
            selected = uniform.player(seed)
        else:
            selected = make_policy(name, seed)
        return BoundedPolicy(selected)

    started = perf_counter()
    with deterministic_cpu():
        valid = run_schedule(
            arena,
            build_schedule(arena),
            lambda row, timing: (rows.append(row), timings.append(timing)),
            factory=factory,
        )
    report = summarize(arena, rows)
    report.update(
        iteration=trainer.iteration,
        training_seed=trainer.config.seed,
        archive_profiles=policy.fingerprints,
        policy_sha256=artifact_hash,
    )
    write_json(out / f"evaluation-{trainer.iteration}.json", report)
    write_json(out / f"outcomes-{trainer.iteration}.json", rows)
    write_json(
        out / f"timing-{trainer.iteration}.json",
        performance(timings, perf_counter() - started),
    )
    if not valid or report["status"] != "valid":
        raise RuntimeError(
            "Evaluation failed; retained incomplete report, no strength estimate"
        )
    return report


def _checkpoint(trainer, directory, provenance):
    iteration = trainer.iteration
    path = directory / f"training-{iteration}.pt"
    fingerprint = save_training(trainer, path, manifest=provenance)
    record = {"iteration": iteration, "sha256": fingerprint}
    atomic_write(directory / f"training-{iteration}.json", canonical(record).encode())


def _restore(directory, provenance):
    records = []
    for path in directory.glob("training-*.json"):
        record = json.loads(path.read_text())
        if path.name != f"training-{record['iteration']}.json":
            raise ValueError("Checkpoint marker name disagrees with its iteration")
        records.append(record)
    if not records:
        return None
    latest = max(records, key=lambda r: r["iteration"])
    trainer = load_training(
        directory / f"training-{latest['iteration']}.pt",
        latest["sha256"],
        manifest=provenance,
    )
    if trainer.iteration != latest["iteration"]:
        raise ValueError("Checkpoint marker refers to another iteration")
    return trainer


def _run(
    plan: Experiment,
    out: Path,
    *,
    resume: Path | None = None,
    reproduce: Path | None = None,
    stop_after: int | None = None,
):
    if resume and reproduce:
        raise ValueError("Choose resume or reproduction")
    if stop_after is not None and (
        type(stop_after) is not int or not 1 <= stop_after <= plan.iterations
    ):
        raise ValueError("Stop boundary must belong to the declared iteration budget")
    provenance = manifest(plan)
    previous = resume or reproduce
    if previous:
        saved = json.loads((previous / "manifest.json").read_text())
        _verify_manifest(saved, provenance)
        provenance = saved
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", provenance)
    started = perf_counter()
    deadline = started + plan.max_seconds
    summaries = []
    target = stop_after or plan.iterations
    for scenario in plan.scenarios:
        for seed in plan.seeds:
            # Directory names come from ordinal positions, never arbitrary scenario text.
            name = f"scenario-{plan.scenarios.index(scenario)}-seed-{seed}"
            directory = out / name
            directory.mkdir()
            table = Table(
                tuple(f"player-{i}" for i in range(len(scenario.stacks))),
                scenario.stacks,
                small_blind=scenario.small_blind,
                big_blind=scenario.big_blind,
                chip_unit=scenario.chip_unit,
            )
            config = replace(plan.training, seed=seed)
            trainer = _restore(resume / name, provenance) if resume else None
            trainer = trainer or HoldemTrainer(table, config)
            if (
                trainer.table != table
                or trainer.config != config
                or trainer.iteration > target
            ):
                raise ValueError("Checkpoint does not match the scheduled job")
            if trainer.iteration:
                _checkpoint(trainer, directory, provenance)
            evaluations = []
            for iteration in range(trainer.iteration + 1, target + 1):
                if perf_counter() + config.max_seconds > deadline:
                    raise TimeoutError(
                        "Insufficient remaining budget for another iteration"
                    )
                trainer.step()
                if iteration % plan.save_every == 0 or iteration == target:
                    _checkpoint(trainer, directory, provenance)
                if iteration % plan.evaluate_every == 0 or iteration == plan.iterations:
                    evaluations.append(
                        evaluate(
                            trainer, scenario, plan, directory, provenance, deadline
                        )
                    )
            # A recovered final boundary still needs an exported, evaluated policy.
            if trainer.iteration == plan.iterations and not evaluations:
                evaluations.append(
                    evaluate(trainer, scenario, plan, directory, provenance, deadline)
                )
            result = {
                "scenario": scenario.name,
                "seed": seed,
                "iteration": trainer.iteration,
                "reports": [asdict(r) for r in trainer.reports],
                "current_profile": trainer.current_profile().fingerprint,
                "archive_profiles": trainer.average_policy().fingerprints,
                "replay": [m.fingerprint() for m in trainer.memories],
                "final_evaluation": evaluations[-1]
                if evaluations and trainer.iteration == plan.iterations
                else None,
            }
            write_json(directory / "result.json", result)
            summaries.append(result)
    result = {
        "format": FORMAT,
        "complete": target == plan.iterations,
        "jobs": summaries,
        "promoted": False,
    }
    if reproduce:
        reference = json.loads((reproduce / "result.json").read_text())
        if digest(reference) != digest(result):
            raise RuntimeError(
                "Fresh training or evaluation differs from the recorded run"
            )
    write_json(out / "result.json", result)
    write_json(out / "runtime.json", {"seconds": perf_counter() - started})
    return result


def run(
    plan: Experiment,
    out: Path,
    *,
    resume: Path | None = None,
    reproduce: Path | None = None,
    stop_after: int | None = None,
):
    """Keep a machine-readable failure beside any committed recovery artifacts."""
    existed = out.exists()
    started = perf_counter()
    try:
        return _run(
            plan, out, resume=resume, reproduce=reproduce, stop_after=stop_after
        )
    except Exception as error:
        if not existed and out.is_dir():
            completed = sorted(out.glob("*/result.json"))
            directories = [
                out / f"scenario-{index}-seed-{seed}"
                for index, _ in enumerate(plan.scenarios)
                for seed in plan.seeds
            ]
            unfinished = [
                d
                for d in directories
                if d.is_dir() and not (d / "result.json").exists()
            ]
            record = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": perf_counter() - started,
                "completed_jobs": [p.parent.name for p in completed],
                "unfinished_jobs": [d.name for d in unfinished],
                "unattempted_jobs": [d.name for d in directories if not d.exists()],
                "promoted": False,
            }
            atomic_write(out / "failure.json", canonical(record).encode())
        raise
