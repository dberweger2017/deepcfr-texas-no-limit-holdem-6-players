"""Compare saved index probabilities with direct eight-snapshot arithmetic."""

import argparse
import gzip
import json
import sqlite3
from pathlib import Path

from src.blueprint.windowed import _hash


def _key(line: bytes) -> str:
    return line[2:line.index(b'"', 2)].decode("ascii")


def verify(run: Path, *, stride: int = 1_000_000):
    snapshots = [run / f"snapshot-{index}.jsonl.gz" for index in range(8)]
    manifest = json.loads((run / "policy-manifest.json").read_text())
    index_path = run / "policy-index.sqlite"
    if _hash(index_path) != manifest["artifact_sha256"]:
        raise ValueError("Index identity differs from manifest")
    if [_hash(path) for path in snapshots] != manifest["snapshot_sha256"]:
        raise ValueError("Snapshot identity differs from manifest")
    selected = []
    with gzip.open(snapshots[-1], "rb") as source:
        for position, line in enumerate(source):
            if position % stride == 0:
                selected.append(_key(line))
    if not selected:
        raise ValueError("Final snapshot has no entries")
    # Pin one key absent from the first profile to exercise sparse-window rules.
    with gzip.open(snapshots[0], "rb") as first, gzip.open(snapshots[-1], "rb") as last:
        first_line = first.readline()
        for last_line in last:
            last_key = _key(last_line)
            while first_line and _key(first_line) < last_key:
                first_line = first.readline()
            if not first_line or _key(first_line) != last_key:
                if last_key not in selected:
                    selected.append(last_key)
                break
    selected_set = set(selected)
    present = []
    for path in snapshots:
        found = {}
        with gzip.open(path, "rb") as source:
            for line in source:
                key = _key(line)
                if key in selected_set:
                    _, names, probabilities = json.loads(line)
                    found[key] = (tuple(names), tuple(probabilities))
        present.append(found)
    maximum_error = 0.0
    late_keys = 0
    db = sqlite3.connect(f"file:{index_path}?mode=ro&immutable=1", uri=True)
    try:
        for key in selected:
            final = present[-1][key]
            names = final[0]
            direct = [0.0] * len(names)
            trained = 0
            for profile in present:
                entry = profile.get(key)
                if entry is not None:
                    if entry[0] != names:
                        raise ValueError("Direct snapshots disagree on action menu")
                    probabilities = entry[1]
                    trained += 1
                else:
                    probabilities = (1/len(names),) * len(names)
                for i, probability in enumerate(probabilities):
                    direct[i] += probability/8
            row = db.execute("SELECT names,current,snapshot,trained_profiles FROM policies WHERE key=?",
                             (key,)).fetchone()
            if row is None or tuple(json.loads(row[0])) != names or row[3] != trained:
                raise ValueError("Index key, menu or trained count disagrees with snapshots")
            current = json.loads(row[1])
            average = json.loads(row[2])
            for a,b in zip(current, final[1]):
                maximum_error = max(maximum_error, abs(a-b))
            for a,b in zip(average, direct):
                maximum_error = max(maximum_error, abs(a-b))
            late_keys += trained < 8
    finally:
        db.close()
    if maximum_error > 1e-12:
        raise ValueError("Saved index differs from direct snapshot calculation")
    return {"schema": "windowed-blueprint-direct-parity-v1",
            "index_sha256": manifest["artifact_sha256"],
            "queried_keys": len(selected), "late_created_keys": late_keys,
            "max_probability_error": maximum_error,
            "snapshot_sha256": manifest["snapshot_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1_000_000)
    args = parser.parse_args()
    result = verify(args.run, stride=args.stride)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
