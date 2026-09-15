"""One isolated, single-threaded CPU pilot job."""

import argparse
import json
import resource
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.pilot_replay import load_replay
from src.solver.evaluate import evaluate
from src.solver.experiment import canonical, write_json
from src.solver.neural.experiment import Plan, run
from src.solver.neural.network import deterministic_cpu, fit, predict, stream_seed


def refit(job: dict, output: Path) -> dict:
    started = perf_counter()
    solver, memory, source = load_replay(Path(job["replay"]))
    spec = job["fit"]
    with deterministic_cpu():
        model, metrics = fit(
            memory,
            solver.features,
            solver.mask,
            **spec,
            iteration=source["iteration"],
            strategy=True,
            seed=stream_seed(source["seed"], "strategy-fit", source["iteration"], 0),
            deadline=started + job["maximum_seconds"],
        )
        policy = solver.local_policy(
            predict(model, solver.features, solver.mask, strategy=True)
        )
        means, weights = memory.means(len(solver.features))
        probabilities = predict(model, solver.features, solver.mask, strategy=True)
        counts = np.bincount(memory.infos[: memory.size], minlength=len(weights))
        rows = []
        for index, info in enumerate(solver.tree.information_sets):
            rows.append(
                {
                    "information_set": asdict(info),
                    "samples": int(counts[index]),
                    "weight": float(weights[index]),
                    "target": means[index].tolist() if counts[index] else None,
                    "prediction": probabilities[index].tolist(),
                    "squared_error": float(
                        ((means[index] - probabilities[index]) ** 2).sum()
                    )
                    if counts[index]
                    else None,
                }
            )
        write_json(output / "information_sets.json", rows)
        return {
            "status": "completed",
            "fit": metrics,
            "evaluation": evaluate(solver.tree, policy).to_dict(),
            "policy_sha256": sha256(canonical(policy.tolist()).encode()).hexdigest(),
            "replay_sha256": source["replay_sha256"],
            "fit_and_evaluation_seconds": perf_counter() - started,
        }


def execute(job: dict, output: Path) -> dict:
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "error"}
    try:
        if job["kind"] == "training":
            result = run(Plan.from_dict(job["plan"]), output / "training")
            report = {
                "status": result["status"],
                "result": {k: v for k, v in result.items() if k != "wall_seconds"},
                "training_seconds": result["wall_seconds"],
            }
        elif job["kind"] == "refit":
            report = refit(job, output)
        else:
            raise ValueError("Unknown pilot job")
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        ArithmeticError,
    ) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(
            wall_seconds=perf_counter() - started,
            cpu_seconds=usage.ru_utime + usage.ru_stime,
            peak_rss_bytes=int(
                usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            ),
        )
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = execute(json.loads(args.job.read_text()), args.out)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
