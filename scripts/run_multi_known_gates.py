"""Execute the fixed, bounded native Kuhn/Leduc protocol under an on-host guard."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from scripts.check_multi_known_reference import check


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    binary = Path("native/hu20-trainer/target/release/examples/multi_known").resolve()
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip(), "commit the tested source before main runs"
    admission = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "native_tree": subprocess.check_output(["git", "rev-parse", "HEAD:native"], text=True).strip(),
                 "binary_sha256": digest(binary), "seeds": [2026101501, 2026101502, 2026101503],
                 "endpoints": {"kuhn": 2_000_000, "leduc": 500_000},
                 "per_player_gain_limits": {"kuhn": 0.005, "leduc": 0.025}}
    (out / "admission.json").write_text(json.dumps(admission, indent=2) + "\n")
    results = []

    def run(game: str, seed: int, iterations: int, name: str, resume: Path | None = None) -> Path:
        folder = out / name
        command = [str(binary), game, "3", str(iterations), str(seed), str(folder)]
        if resume:
            command.append(str(resume))
        (out / f"{name}.command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (out / f"{name}.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800)
        return folder

    started = time.monotonic()
    for game, iterations in admission["endpoints"].items():
        for seed in admission["seeds"]:
            folder = run(game, seed, iterations, f"{game}-{seed}")
            reference = check(folder / "average.json")
            passed = all(0 <= gain + 1e-12 <= admission["per_player_gain_limits"][game] for gain in reference["gains"])
            record = {"game": game, "seed": seed, "iterations": iterations, "reference": reference,
                      "checkpoint_sha256": digest(folder / "checkpoint.gz"), "gain_gate_passed": passed}
            if game == "kuhn":
                values = reference["values"]
                beta = min(0.25, max(0, (24 * (values[2] - values[0]) - 1.5) / 2))
                expected = [-(0.5 + beta) / 24, -0.5 / 24, (1 + beta) / 24]
                error = max(abs(a-b) for a, b in zip(values, expected))
                record.update(fitted_beta=beta, published_value_max_error=error, published_value_gate_passed=error <= 0.005)
                passed &= error <= 0.005
            record["status"] = "passed" if passed else "failed"
            results.append(record)
            (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(record), flush=True)
        seed = admission["seeds"][0]
        prefix = run(game, seed, iterations // 2, f"{game}-prefix")
        resumed = run(game, seed, iterations, f"{game}-resumed", prefix / "checkpoint.gz")
        fresh = out / f"{game}-{seed}"
        recovery = {"game": game, "fresh": digest(fresh / "checkpoint.gz"), "resumed": digest(resumed / "checkpoint.gz"),
                    "prefix": digest(prefix / "checkpoint.gz"), "average_equal": digest(fresh / "average.json") == digest(resumed / "average.json")}
        (out / f"{game}-recovery.json").write_text(json.dumps(recovery, indent=2) + "\n")
        assert recovery["fresh"] == recovery["resumed"] and recovery["average_equal"], recovery
    summary = {"status": "passed" if all(r["status"] == "passed" for r in results) else "failed",
               "elapsed_seconds": time.monotonic() - started, "runs": len(results)}
    (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
