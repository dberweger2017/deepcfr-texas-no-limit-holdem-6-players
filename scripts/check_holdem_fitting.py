"""Refit frozen Hold'em replay with matched initialization and minibatch streams."""

import argparse
import json
import subprocess
import sys
import tarfile
from dataclasses import asdict, replace
from hashlib import file_digest
from math import fsum, isfinite
from pathlib import Path
from time import perf_counter

import torch

from src.arena.artifacts import environment, source_fingerprint
from src.holdem.checkpoint import load_training
from src.holdem.fitting import fit_role
from src.holdem.timing import peak_rss_bytes
from src.holdem.training import SampledTrainConfig
from src.solver.experiment import write_json
from src.solver.neural.network import deterministic_cpu

ARMS = ((1.0, 64), (1.0, 256), (None, 64), (None, 256))


def checksum(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def replay_metrics(model, memory, iteration, *, deadline=float("inf")):
    """Measure the full retained empirical objective, not held-out prediction error."""
    regrets, values = [], []
    nonpositive = 0
    items = memory.items
    total_weight = iteration * (iteration + 1) / 2
    with torch.inference_mode():
        for start in range(0, len(items), 32):
            if perf_counter() >= deadline:
                raise TimeoutError("Replay scoring exceeded the deadline")
            batch = items[start : start + 32]
            scores = model([s.target.candidates for s in batch])
            for sample, score in zip(batch, scores):
                r, v = score.regrets.tolist(), score.values.tolist()
                if not all(isfinite(x) for x in (*r, *v)):
                    raise FloatingPointError("Non-finite replay predictions")
                weight = (
                    memory.seen
                    / len(items)
                    * sample.iteration
                    / total_weight
                    / sample.roots
                    / sample.target.own_sample_reach
                )
                regrets.append(
                    weight
                    * fsum((p - y) ** 2 for p, y in zip(r, sample.target.regrets_bb))
                )
                values.append(
                    weight
                    * fsum((p - y) ** 2 for p, y in zip(v, sample.target.values_bb))
                )
                nonpositive += max(r) <= 0
    regret, value = fsum(regrets), fsum(values)
    if not isfinite(regret + value):
        raise FloatingPointError("Non-finite replay objective")
    return {
        "regret_loss": regret,
        "value_loss": value,
        "loss": regret + value,
        "nonpositive_regret_fraction": nonpositive / len(items),
        "records": len(items),
        "population": memory.seen,
    }


def refit(memory, config, *, iteration, seed, gradient_clip, steps, deadline):
    started = perf_counter()
    model, metrics = fit_role(
        memory,
        replace(config, steps=steps),
        iteration=iteration,
        seed=seed,
        gradient_clip=gradient_clip,
        deadline=deadline,
    )
    fitting_seconds = perf_counter() - started
    empirical = replay_metrics(model, memory, iteration, deadline=deadline)
    return model, {
        "gradient_clip": gradient_clip,
        "steps": steps,
        "fit": asdict(metrics),
        "replay": empirical,
        "fitting_seconds": fitting_seconds,
        "total_seconds": perf_counter() - started,
    }


def worker(settings, source, output, seed):
    started = perf_counter()
    deadline = started + settings["maximum_seconds_per_seed"]
    spec = next(s for s in settings["checkpoints"] if s["seed"] == seed)
    report = {"seed": seed, "status": "running", "fits": []}
    output.mkdir(parents=True, exist_ok=False)
    try:
        manifest = json.loads((source / f"manifest-{seed}.json").read_text())
        with deterministic_cpu():
            trainer = load_training(
                source / f"training-{seed}.pt", spec["sha256"], manifest=manifest
            )
            if (
                trainer.iteration != settings["iteration"]
                or trainer.config.seed != seed
                or not isinstance(trainer.config, SampledTrainConfig)
                or trainer.table.capacity != 6
                or asdict(trainer.config.fit) != settings["fit"]
            ):
                raise ValueError("Checkpoint does not match the frozen protocol")
            report["loading_seconds"] = perf_counter() - started
            write_json(output / "report.json", report)
            for memory in trainer.memories:
                original = trainer.current_profile()._models[memory.role]
                if original is None or not memory:
                    raise ValueError(
                        "Expected trained models and replay for all six roles"
                    )
                for clip, steps in ARMS:
                    model, result = refit(
                        memory,
                        trainer.config.fit,
                        iteration=trainer.iteration,
                        seed=seed,
                        gradient_clip=clip,
                        steps=steps,
                        deadline=deadline,
                    )
                    result["role"] = memory.role
                    result["max_parameter_difference_from_checkpoint"] = max(
                        float((p - q).abs().max())
                        for p, q in zip(model.parameters(), original.parameters())
                    )
                    name = f"role-{memory.role}-clip-{clip}-steps-{steps}.pt"
                    torch.save(model.state_dict(), output / name)
                    result["weights"] = {
                        "path": name,
                        "sha256": checksum(output / name),
                    }
                    report["fits"].append(result)
                    report["elapsed_seconds"] = perf_counter() - started
                    write_json(output / "report.json", report)
                    print(json.dumps({"seed": seed, **result}), flush=True)
            report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wall_seconds"] = perf_counter() - started
        report["peak_process_rss_bytes"] = peak_rss_bytes()
        write_json(output / "report.json", report)


def extract_inputs(archive, specs, destination, *, deadline=float("inf")):
    destination.mkdir()
    wanted = {}
    for spec in specs:
        root = f"results/longer-05-seed-{spec['seed']}"
        wanted[f"{root}/scenario-0-seed-{spec['seed']}/training-256.pt"] = (
            destination / f"training-{spec['seed']}.pt",
            spec["sha256"],
        )
        wanted[f"{root}/manifest.json"] = (
            destination / f"manifest-{spec['seed']}.json",
            None,
        )
    found = set()
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            if perf_counter() >= deadline:
                raise TimeoutError("Archive preparation exceeded the deadline")
            name = member.name.removeprefix("./")
            if name not in wanted:
                continue
            if name in found or not member.isfile() or member.size > 1_000_000_000:
                raise ValueError("Invalid or duplicate checkpoint archive member")
            path, expected = wanted[name]
            with bundle.extractfile(member) as stream, path.open("xb") as target:
                while chunk := stream.read(1024 * 1024):
                    if perf_counter() >= deadline:
                        raise TimeoutError("Archive preparation exceeded the deadline")
                    target.write(chunk)
            if expected is not None and checksum(path) != expected:
                raise ValueError("Checkpoint checksum mismatch")
            found.add(name)
    if found != set(wanted):
        raise ValueError("Missing checkpoint or manifest")


def run(plan, archive, output):
    settings = json.loads(plan.read_text())
    if (
        settings["version"] != 1
        or settings["iteration"] != 256
        or [s["seed"] for s in settings["checkpoints"]] != [307, 311, 313]
        or settings["maximum_seconds_per_seed"] != 900
        or settings["maximum_total_seconds"] != 2700
    ):
        raise ValueError("Expected the frozen iteration-256 fitting protocol")
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    summary = {
        "status": "running",
        "jobs": [
            {"seed": spec["seed"], "status": "unattempted"}
            for spec in settings["checkpoints"]
        ],
    }
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(
        output / "manifest.json",
        {
            "settings": settings,
            "plan_sha256": checksum(plan),
            "revision": revision,
            "script_sha256": checksum(Path(__file__)),
            "fitting_sha256": checksum(Path("src/holdem/fitting.py")),
            "python": sys.version,
            "torch": torch.__version__,
            "platform": sys.platform,
            "environment": environment(),
            "source_fingerprint": source_fingerprint(),
            "archive": str(archive.resolve()),
        },
    )
    try:
        extract_inputs(
            archive,
            settings["checkpoints"],
            output / "inputs",
            deadline=started + settings["maximum_total_seconds"],
        )
        for job in summary["jobs"]:
            remaining = settings["maximum_total_seconds"] - (perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("Campaign deadline expired")
            job["status"] = "running"
            write_json(output / "result.json", summary)
            with (output / f"seed-{job['seed']}.log").open("x") as log:
                command = [
                    sys.executable,
                    "-m",
                    "scripts.check_holdem_fitting",
                    "--plan",
                    str(plan),
                    "--out",
                    str(output),
                    "--worker",
                    str(job["seed"]),
                ]
                try:
                    completed = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=min(settings["maximum_seconds_per_seed"], remaining),
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    job["status"] = "timeout"
                    raise
            job.update(
                exit_code=completed.returncode,
                status="completed" if completed.returncode == 0 else "failed",
            )
            if completed.returncode:
                raise RuntimeError(
                    f"Seed {job['seed']} failed; remaining jobs unattempted"
                )
        summary["status"] = "completed"
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["wall_seconds"] = perf_counter() - started
        write_json(output / "result.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=(307, 311, 313))
    args = parser.parse_args()
    if args.worker is not None:
        worker(
            json.loads(args.plan.read_text()),
            args.out / "inputs",
            args.out / f"seed-{args.worker}",
            args.worker,
        )
    else:
        if args.archive is None:
            parser.error("--archive is required")
        run(args.plan, args.archive, args.out)


if __name__ == "__main__":
    main()
