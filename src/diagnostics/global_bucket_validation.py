"""Data adapter for frozen global-bucket witnesses; no production key changes."""

from copy import deepcopy
from pathlib import Path
import numpy as np

from src.blueprint.equity_buckets import BucketTable
from src.diagnostics.board_pooling_features import add_pool_keys

ALIASES = {50: "eq50-fit0", 200: "eq50-fit1"}
BLOCKED = 65535


def global_labels(data, tables, *, ks=(50, 200)):
    """Resolve legal holding/runout states through the unchanged table reader."""
    boards, holdings = data["boards"], data["holdings"]
    codes = np.asarray(data["codes"])
    if codes.shape != (len(boards), len(holdings)):
        raise ValueError("Frozen holding/board matrix dimensions differ")
    output, counts = {}, {}
    for k in ks:
        matrix = np.full(codes.shape, BLOCKED, dtype=np.uint16)
        for row, board in enumerate(boards):
            street = "turn" if len(board) == 4 else "river"
            table = tables[k, street]
            if table.k != k or table.street != street:
                raise ValueError("Global table street or K differs")
            for column, hand in enumerate(holdings):
                blocked = bool(set(hand).intersection(board))
                if blocked != (codes[row, column] == 255):
                    raise ValueError("Frozen card-removal sentinel differs")
                if not blocked:
                    bucket = table.bucket(hand, board)
                    if not 0 <= bucket < k:
                        raise ValueError("Bucket outside declared K")
                    matrix[row, column] = bucket
        output[str(k)] = matrix.tolist()
        counts[str(k)] = {street: sorted(set(map(int, np.unique(matrix[rows]))) - {BLOCKED})
                          for street, rows in (("turn", slice(0, 1)), ("river", slice(1, None)))}
    return output, counts


def project_compact(request, original, labels):
    """Replace only diagnostic card labels/maps; preserve blueprint and templates."""
    data = deepcopy(original)
    data["crossfit_labels"] = {"0": labels["50"], "1": labels["200"]}
    rebuilt = add_pool_keys(request, data, data)
    if rebuilt["v1"] != original["pool_keys"]["v1"]:
        raise ValueError("V1 key changed during global projection")
    # Removing the unused corpus-fit slot avoids collecting redundant statistics.
    data["pool_keys"] = {metric: rebuilt[metric] for metric in ("v1", *ALIASES.values())}
    data["global_bucket_transport"] = {
        "aliases": {alias: f"global-k{k}" for k, alias in ALIASES.items()},
        "scope": "fixed full-deck tables; no corpus fitting"}
    return data


def load_tables(folder):
    return {(k, street): BucketTable(Path(folder) / f"{street}-k{k}.bin")
            for k in ALIASES for street in ("turn", "river")}
