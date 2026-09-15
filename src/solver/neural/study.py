"""Shared-collection strategy comparisons followed by an independent seed gate."""

import json
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from src.solver.evaluate import evaluate
from src.solver.experiment import canonical, write_json
from src.solver.neural.campaign import Campaign, run_seed
from src.solver.neural.checkpoint import load_training
from src.solver.neural.experiment import provenance
from src.solver.neural.network import deterministic_cpu


@dataclass(frozen=True)
class Variant:
    name: str
    hidden: int
    steps: int

    def __post_init__(self):
        if not self.name or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in self.name
        ):
            raise ValueError("Invalid strategy variant name")
        for value, upper in ((self.hidden, 512), (self.steps, 48000)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("Invalid strategy fitting budget")


@dataclass(frozen=True)
class Study:
    exploration: Campaign
    confirmation_seeds: tuple[int, ...]
    kuhn: Campaign
    checkpoints: tuple[int, ...]
    variants: tuple[Variant, ...]
    worker_seconds: int = 7200
    maximum_seconds: int = 21600
    version: int = 1

    def __post_init__(self):
        for field in ("confirmation_seeds", "checkpoints", "variants"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if (
            self.version != 1
            or self.exploration.training.game != "leduc"
            or self.kuhn.training.game != "kuhn"
        ):
            raise ValueError("Expected a Leduc study with a Kuhn regression gate")
        seed_set = set(self.confirmation_seeds)
        if (
            len(seed_set) < 3
            or len(seed_set) != len(self.confirmation_seeds)
            or seed_set.intersection(self.exploration.seeds)
            or any(type(seed) is not int or seed < 0 for seed in seed_set)
        ):
            raise ValueError("Confirmation seeds must be distinct from exploration")
        plan = self.exploration.training
        if (
            not self.checkpoints
            or tuple(sorted(set(self.checkpoints))) != self.checkpoints
            or self.checkpoints[-1] != plan.iterations
            or any(
                type(t) is not int or t < 1 or t % plan.evaluation_interval
                for t in self.checkpoints
            )
        ):
            raise ValueError(
                "Study checkpoints must be scheduled and include the final iteration"
            )
        if (
            not self.variants
            or len(self.variants) > 8
            or len({v.name for v in self.variants}) != len(self.variants)
        ):
            raise ValueError("Expected distinct strategy variants")
        first = self.variants[0]
        if (first.name, first.hidden, first.steps) != (
            "baseline",
            plan.training.strategy_width,
            plan.training.strategy_steps,
        ):
            raise ValueError("The first variant must describe the collection baseline")
        if (
            type(self.worker_seconds) is not int
            or not 1 <= self.worker_seconds <= 7200
            or self.worker_seconds < plan.maximum_seconds
            or type(self.maximum_seconds) is not int
            or not 1 <= self.maximum_seconds <= 21600
        ):
            raise ValueError("Invalid study runtime limits")

    @classmethod
    def from_dict(cls, value):
        return cls(
            **{
                **value,
                "exploration": Campaign.from_dict(value["exploration"]),
                "kuhn": Campaign.from_dict(value["kuhn"]),
                "variants": tuple(Variant(**v) for v in value["variants"]),
            }
        )

    def confirmation(self, name: str) -> Campaign:
        variant = next(v for v in self.variants if v.name == name)
        plan = self.exploration.training
        return replace(
            self.exploration,
            seeds=self.confirmation_seeds,
            training=replace(
                plan,
                training=replace(
                    plan.training,
                    strategy_hidden=variant.hidden,
                    strategy_steps=variant.steps,
                ),
            ),
        )


def explore_seed(study: Study, seed: int, output: Path) -> dict:
    plan = study.exploration.for_seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {**provenance(asdict(study)), "seed": seed})
    started = perf_counter()
    report = {"status": "running", "seed": seed, "fits": []}
    try:
        baseline = run_seed(study.exploration, seed, output / "baseline")
        if baseline["training_status"] != "completed":
            raise TimeoutError(
                "Exploration training did not finish its declared iterations"
            )
        report["oracle"] = baseline["oracle"]
        training = output / "baseline/training"
        with deterministic_cpu():
            for iteration in study.checkpoints:
                path = training / f"iteration-{iteration:06d}.pt"
                digest = sha256(path.read_bytes()).hexdigest()
                solver, progress = load_training(
                    path, digest, manifest=provenance(asdict(plan))
                )
                original = progress["report"]["evaluations"][-1]
                if solver.iterations != iteration or original["iteration"] != iteration:
                    raise ValueError("Unexpected study checkpoint iteration")
                if (
                    evaluate(solver.tree, solver.average_policy()).to_dict()
                    != original["neural_average"]
                ):
                    raise ArithmeticError(
                        "Reloaded strategy differs from its training evaluation"
                    )
                for variant in study.variants:
                    began = perf_counter()
                    if variant.name == "baseline":
                        metrics, evaluation = (
                            original["strategy_fit"],
                            original["neural_average"],
                        )
                    else:
                        solver.config = replace(
                            plan.training,
                            strategy_hidden=variant.hidden,
                            strategy_steps=variant.steps,
                        )
                        metrics = solver.fit_strategy(started + study.worker_seconds)
                        evaluation = evaluate(
                            solver.tree, solver.average_policy()
                        ).to_dict()
                    policy = solver.average_policy()
                    report["fits"].append(
                        {
                            "iteration": iteration,
                            "variant": variant.name,
                            "fit": metrics,
                            "evaluation": evaluation,
                            "value_error": abs(
                                evaluation["value_player0"]
                                - baseline["oracle"]["value"]
                            ),
                            "policy_sha256": sha256(
                                canonical(policy.tolist()).encode()
                            ).hexdigest(),
                            "checkpoint_sha256": digest,
                            "additional_seconds": perf_counter() - began,
                        }
                    )
                    write_json(output / "report.json", report)
        report["status"] = "completed"
    except BaseException as exc:
        report["status"], report["error"] = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = perf_counter() - started
        write_json(output / "report.json", report)
    return report


def select(study: Study, reports: list[dict]) -> dict:
    seeds = [r["seed"] for r in reports]
    if len(seeds) != len(set(seeds)) or set(seeds) != set(study.exploration.seeds):
        raise ValueError("Selection requires every exploration seed exactly once")
    expected = {(t, v.name) for t in study.checkpoints for v in study.variants}
    for report in reports:
        keys = [(row["iteration"], row["variant"]) for row in report["fits"]]
        if (
            report["status"] != "completed"
            or len(keys) != len(expected)
            or set(keys) != expected
        ):
            raise ValueError(
                "Selection requires every planned fit, including poor results"
            )
    summaries = []
    for variant in study.variants:
        rows = [
            row
            for report in reports
            for row in report["fits"]
            if row["variant"] == variant.name
            and row["iteration"] == study.checkpoints[-1]
        ]
        values = [row["evaluation"]["exploitability"] for row in rows]
        summaries.append(
            {
                "variant": variant.name,
                "worst_exploitability": max(values),
                "mean_exploitability": sum(values) / len(values),
                "eligible": all(
                    row["evaluation"]["exploitability"]
                    <= study.exploration.maximum_exploitability
                    and row["value_error"] <= study.exploration.maximum_value_error
                    for row in rows
                ),
            }
        )
    eligible = [row for row in summaries if row["eligible"]]
    winner = (
        min(
            eligible,
            key=lambda row: (
                row["worst_exploitability"],
                row["mean_exploitability"],
                row["variant"],
            ),
        )
        if eligible
        else None
    )
    return {
        "status": "selected" if winner else "no_candidate",
        "variants": summaries,
        "selected": None if winner is None else winner["variant"],
        "confirmation_seeds": list(study.confirmation_seeds),
        "model_promoted": False,
    }


def load_exploration(study: Study, paths: list[Path]) -> tuple[list[dict], dict]:
    expected = provenance(asdict(study))
    reports, hashes = [], {}
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text())
        for field in ("plan_sha256", "source_sha256", "environment", "protocol"):
            if canonical(manifest[field]) != canonical(expected[field]):
                raise ValueError(f"Study manifest mismatch: {field}")
        data = (path / "report.json").read_bytes()
        report = json.loads(data)
        if report["seed"] != manifest["seed"]:
            raise ValueError("Study seed differs from its manifest")
        hashes[str(report["seed"])] = sha256(data).hexdigest()
        reports.append(report)
    return reports, hashes
