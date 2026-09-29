"""Verify saved search runs and compare corrected policies on paired blocks."""

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from src.arena.report import comparison
from src.arena.schedule import digest

def _file_sha256(path: Path) -> str:
    result = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _load_run(path: Path) -> tuple[dict, dict, dict]:
    checksums = json.loads((path / "checksums.json").read_text())
    actual = {
        str(file.relative_to(path)): _file_sha256(file)
        for file in path.rglob("*") if file.is_file() and file.name != "checksums.json"
    }
    if actual != checksums:
        raise ValueError(f"Artifact checksums differ in {path}")
    result = json.loads((path / "result.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    if result["status"] != "valid" or manifest["dirty"]:
        raise ValueError(f"Run is invalid or source is dirty: {path}")
    if result["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash differs in {path}")
    rows = {}
    for name in result["comparisons"]:
        rows[name] = [json.loads(line) for line in (path / f"{name}-hands.jsonl").read_text().splitlines()]
        if any(digest({k: v for k, v in row.items() if k != "outcome_sha256"}) != row["outcome_sha256"] for row in rows[name]):
            raise ValueError(f"Hand digest differs in {path}/{name}")
    return result, manifest, rows


def _block_rates(rows: list[dict], arm: str) -> dict[tuple, float]:
    blocks = defaultdict(list)
    for row in rows:
        if row["arm"] == arm:
            if row["status"] != "completed":
                raise ValueError("Incomplete hand in a valid run")
            blocks[(row["scenario"], row["block"])].append(row)
    return {
        key: 100 * sum(row["candidate_chips"] for row in group)
        / (len(group) * group[0]["big_blind"])
        for key, group in blocks.items()
    }


def report(runs: dict[str, Path]) -> dict:
    loaded = {label: _load_run(path) for label, path in runs.items()}
    reference = loaded["5m"][1]["comparison"]["comparisons"]
    for label, (result, manifest, _) in loaded.items():
        plans = manifest["comparison"]["comparisons"]
        if plans != reference:
            raise ValueError(f"Schedules or opponents differ for {label}")
        if result["search"] != loaded["5m"][0]["search"]:
            raise ValueError(f"Search settings differ for {label}")
    output = {"checkpoints": {}, "paired_checkpoint_comparisons": {}}
    for label, (result, manifest, _) in loaded.items():
        output["checkpoints"][label] = {
            "sha256": result["checkpoint_sha256"],
            "revision": manifest["revision"],
            "entries": manifest["checkpoint_entries"],
            "benchmarks": {
                name: {
                    "comparison": data["report"]["scenarios"]["six-handed-100bb"]["comparison"],
                    "telemetry": data["telemetry"],
                    "completed_hands": data["report"]["completed_hands"],
                    "invalid_actions": data["report"]["invalid_actions"],
                }
                for name, data in result["comparisons"].items()
            },
        }
    for name in reference:
        small = _block_rates(loaded["5m"][2][name], "candidate")
        for label in ("12m", "58m"):
            larger = _block_rates(loaded[label][2][name], "candidate")
            if small.keys() != larger.keys():
                raise ValueError(f"Paired block keys differ for {label}/{name}")
            keys = sorted(small)
            output["paired_checkpoint_comparisons"][f"{label}_minus_5m_{name}"] = comparison(
                [larger[key] for key in keys], [small[key] for key in keys]
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("5m", "12m", "58m"):
        parser.add_argument(f"--{label}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report({label: getattr(args, label) for label in ("5m", "12m", "58m")})
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
