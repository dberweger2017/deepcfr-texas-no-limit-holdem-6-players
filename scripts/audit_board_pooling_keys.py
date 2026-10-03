"""Stream retained 500M node statistics for preselected #145 excess-fold keys."""

import argparse
from collections import defaultdict
import gzip
import json
from itertools import combinations
from math import fsum
from pathlib import Path
from time import monotonic

from src.blueprint.solver import regret_match
from src.diagnostics.cfr_average import checked_header, checked_row
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.saved_hu20 import file_hash

ARTIFACTS = Path("docs/reports/hu20-exact-turn-check-artifacts")


def selected_keys(supplement, count=20):
    return sorted({r["v1_key"] for group in ("A", "B")
                   for r in supplement["top_keys_by_excess_mass"][group][:count]})


def stream_nodes(path, spec, keys):
    if file_hash(path) != spec["checkpoint_sha256"]:
        raise ValueError("Original 500M checkpoint hash differs")
    result = {}
    with gzip.open(path, "rt") as source:
        header = json.loads(next(source)); checked_header(header, spec)
        for line in source:
            row = json.loads(line)
            if row[0] not in keys:
                continue
            key, names, regrets, average, total, visits = checked_row(row, header["iteration"])
            if key in result:
                raise ValueError("Duplicate retained selected node")
            result[key] = {"names": names, "visits": visits, "regrets": regrets,
                           "average": average, "average_mass": total,
                           "zero_average_mass": not total,
                           "current": regret_match(regrets)}
    return result


def compare(rows):
    output = []
    for a, b in combinations(sorted(rows), 2):
        left, right = rows[a], rows[b]
        if set(left["names"]) != set(right["names"]):
            raise ValueError("Selected actual key has incompatible lineage menus")
        values = {}
        for field in ("average", "current"):
            lhs = dict(zip(left["names"], left[field], strict=True))
            rhs = dict(zip(right["names"], right[field], strict=True))
            values[field + "_tv"] = fsum(abs(lhs[n] - rhs[n]) for n in lhs) / 2
        output.append({"lineages": [a, b], **values})
    return output


def audit(checkpoints, out):
    if out.exists():
        raise FileExistsError("Preserve immutable companion evidence")
    start = monotonic()
    supplement = json.loads((ARTIFACTS / "main03-supplement.json").read_text())
    keys = selected_keys(supplement)
    # Write selection before opening any checkpoint statistics.
    atomic_json(out.with_name(out.stem + "-selection.json"), {
        "keys": keys, "top_per_set": 20,
        "supplement_sha256": file_hash(ARTIFACTS / "main03-supplement.json")})
    models = json.loads(Path("configs/diagnostics/b500-cfr-average-inputs.json").read_text())["models"]
    nodes = {m["seed"]: stream_nodes(checkpoints / m["checkpoint_path"], m, set(keys))
             for m in models}
    diversity = defaultdict(set)
    boards = {}
    for group in ("A", "B"):
        for root in json.loads((ARTIFACTS / ("corpus-" + group + ".json")).read_text())["roots"]:
            boards[root["spot"]] = tuple(root["board"])
    with gzip.open(ARTIFACTS / "main03-overfold-groups.jsonl.gz", "rt") as source:
        for line in source:
            row = json.loads(line)
            if row["v1_key"] in keys and row["excess_fold_mass"] > 0:
                diversity[row["v1_key"]].add(boards[row["spot"]])
    result = {"format": "hu20-board-pooling-companion-v1", "keys": [],
              "inputs": [{"path": m["checkpoint_path"], "sha256": m["checkpoint_sha256"]}
                         for m in models], "elapsed_seconds": monotonic() - start,
              "board_diversity_scope": "distinct #145 root boards with positive excess-fold contexts; not historical training occupancy",
              "historical_training_board_occupancy": "not retained in checkpoints",
              "supplement_sha256": file_hash(ARTIFACTS / "main03-supplement.json"),
              "context_sha256": file_hash(ARTIFACTS / "main03-overfold-groups.jsonl.gz")}
    for key in keys:
        rows = {seed: lineage[key] for seed, lineage in nodes.items() if key in lineage}
        result["keys"].append({"key": key, "lineages": rows,
                               "missing_lineages": sorted(set(nodes) - set(rows)),
                               "pairwise": compare(rows),
                               "diagnostic_root_boards": sorted(diversity[key]),
                               "diagnostic_distinct_root_boards": len(diversity[key])})
    atomic_json(out, result)
    return {"keys": len(keys), "elapsed_seconds": result["elapsed_seconds"],
            "output_sha256": file_hash(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.checkpoints, args.out)))


if __name__ == "__main__":
    main()
