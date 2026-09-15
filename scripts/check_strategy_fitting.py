"""Compare strategy fitting budgets on one completed campaign's frozen replay."""

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from src.solver.evaluate import evaluate
from src.solver.experiment import write_json
from src.solver.neural.checkpoint import load_training
from src.solver.neural.experiment import Plan, provenance
from src.solver.neural.network import deterministic_cpu, fit, predict, stream_seed


def check(settings: dict, training: Path, output: Path) -> dict:
    if (
        settings["version"] != 1
        or settings["game"] != "leduc"
        or settings["seed"] != 11
        or settings["iteration"] != 120
        or settings["hidden_sizes"] != [64, 128]
        or settings["steps"] != 24000
        or settings["maximum_seconds"] != 180
        or settings["comparison_exploitability"] != 0.15
    ):
        raise ValueError("Expected the declared frozen-strategy-refit-v1 protocol")
    original = json.loads((training / "manifest.json").read_text())
    plan = Plan.from_dict(original["plan"])
    descriptor = json.loads((training / "checkpoint.json").read_text())
    name = descriptor["file"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Invalid checkpoint filename")
    solver, progress = load_training(
        training / name, descriptor["sha256"], manifest=provenance(asdict(plan))
    )
    if (
        solver.tree.game != settings["game"]
        or solver.config.seed != settings["seed"]
        or solver.iterations != settings["iteration"]
        or solver.strategy is None
    ):
        raise ValueError(
            "The diagnostic requires the completed declared seed's strategy"
        )
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "manifest.json",
        {
            **provenance(settings),
            "diagnostic_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint_sha256": descriptor["sha256"],
            "training_manifest": original,
        },
    )
    started = perf_counter()
    report = {"status": "running", "refits": []}
    memory = solver.strategy_memory

    def replay_hash():
        return sha256(
            b"".join(
                array[: memory.size].tobytes()
                for array in (memory.infos, memory.iterations, memory.targets)
            )
        ).hexdigest()

    try:
        with deterministic_cpu():
            before = replay_hash()
            report["memory_sha256"] = before
            report["baseline"] = progress["report"]["evaluations"][-1]
            baseline = evaluate(solver.tree, solver.average_policy()).to_dict()
            if baseline != report["baseline"]["neural_average"]:
                raise ArithmeticError(
                    "Loaded baseline differs from the retained evaluation"
                )
            for hidden in settings["hidden_sizes"]:
                model, metrics = fit(
                    memory,
                    solver.features,
                    solver.mask,
                    hidden=hidden,
                    steps=settings["steps"],
                    batch_size=solver.config.batch_size,
                    learning_rate=solver.config.learning_rate,
                    iteration=solver.iterations,
                    seed=stream_seed(
                        solver.config.seed, "strategy-fit", solver.iterations, 0
                    ),
                    strategy=True,
                    deadline=started + settings["maximum_seconds"],
                )
                evaluation = evaluate(
                    solver.tree,
                    solver.local_policy(
                        predict(model, solver.features, solver.mask, strategy=True)
                    ),
                )
                if replay_hash() != before:
                    raise ArithmeticError("Strategy refitting mutated frozen replay")
                report["refits"].append(
                    {
                        "hidden": hidden,
                        "fit": metrics,
                        "evaluation": evaluation.to_dict(),
                        "meets_original_seed_limit": evaluation.exploitability
                        <= settings["comparison_exploitability"],
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = check(json.loads(args.plan.read_text()), args.training, args.out)
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        ArithmeticError,
    ) as exc:
        parser.exit(2, f"Strategy fitting check failed: {exc}\n")
    print(
        json.dumps(
            {"status": result["status"], "report": str(args.out / "report.json")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
