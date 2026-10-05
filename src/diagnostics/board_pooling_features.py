"""One outcome-free equity codebook shared across the frozen board corpus."""

from itertools import combinations
import numpy as np

from src.blueprint.search import DECK
from src.diagnostics.flop_check import (decode_descriptor, descriptor,
                                        descriptor_code, factored_key, line_key)
from src.diagnostics.flop_check_analysis import uniform_river_equities, emd_clusters


def card_features(board, bins=20):
    board = tuple(board)
    if len(board) != 4 or len(set(board)) != 4:
        raise ValueError("Need a distinct four-card board")
    cards = [c for c in DECK if c not in board]
    holdings = list(combinations(cards, 2))
    lookup = {tuple(sorted(h)): i for i, h in enumerate(holdings)}
    boards = [board] + [board + (c,) for c in cards]
    codes = np.full((49, len(holdings)), 255, dtype=np.uint16)
    river = np.full((48, len(holdings)), np.nan)
    hist = np.zeros((len(holdings), bins), dtype=np.uint16)
    for row, public in enumerate(boards):
        for i, hand in enumerate(holdings):
            if not set(hand).intersection(public):
                codes[row, i] = descriptor_code(descriptor(hand, public))
        if row:
            hands, equity = uniform_river_equities(public)
            mapped = np.asarray([lookup[tuple(sorted(h))] for h in hands])
            river[row - 1, mapped] = equity
            np.add.at(hist, (mapped, np.minimum((equity * bins).astype(int), bins - 1)), 1)
    return {"board": list(board), "boards": [list(b) for b in boards],
            "holdings": [list(h) for h in holdings], "codes": codes,
            "histograms": hist / hist.sum(axis=1)[:, None], "river_equity": river}


def shared_codebook(features, *, k=50, seed=202610030305, training_indices=None):
    training = features if training_indices is None else [features[i] for i in training_indices]
    if not training:
        raise ValueError("No training features for the shared codebook")
    hist = np.concatenate([f["histograms"] for f in training])
    _, centers = emd_clusters(hist, k, seed=seed)
    equities = np.concatenate([f["river_equity"][np.isfinite(f["river_equity"])]
                               for f in training])
    edges = np.quantile(equities, np.arange(1, k) / k)
    outputs = []
    for f in features:
        count = len(f["holdings"]); river = f["river_equity"]
        valid = np.isfinite(river)
        rivers = np.full(river.shape, 65535, dtype=np.uint16)
        rivers[valid] = np.searchsorted(edges, river[valid], side="right")
        # Assignment never updates training centers using held-out features.
        cumulative = np.cumsum(f["histograms"], axis=1)
        labels = np.abs(cumulative[:, None, :] - np.cumsum(centers, axis=1)[None, :, :]).sum(axis=2).argmin(axis=1)
        buckets = np.vstack([labels[None, :], rivers]).tolist()
        outputs.append({"board": f["board"], "boards": f["boards"], "holdings": f["holdings"],
                        "codes": f["codes"].tolist(), "labels": {"50": buckets, "200": buckets},
                        "root_equity": np.nanmean(river, axis=0).tolist(),
                        "equity_label_scope": "one shared corpus codebook; 200 field unused",
                        "histogram_bins": hist.shape[1]})
    return outputs, {"k": k, "seed": seed, "turn_centers": centers.tolist(),
                     "river_edges": edges.tolist(), "training_indices": training_indices,
                     "scope": "uniform compatible holdings; only declared training boards fit centers/edges"}


def crossfit_codebooks(features, folds, *, k=50, seed=202610030305):
    if len(features) != len(folds) or set(folds) != {0, 1}:
        raise ValueError("Need features from both frozen halves")
    output, all_board = shared_codebook(features, k=k, seed=seed)
    books = {"all": all_board}
    for fold in (0, 1):
        indices = [i for i, f in enumerate(folds) if f == fold]
        assigned, book = shared_codebook(features, k=k, seed=seed + fold, training_indices=indices)
        books[str(fold)] = book
        for result, labels in zip(output, assigned, strict=True):
            result.setdefault("crossfit_labels", {})[str(fold)] = labels["labels"]["50"]
    return output, books


def add_pool_keys(request, data, tables):
    """Export real v1 hashes and shared equity-template hashes, never table IDs."""
    result = {"v1": {}, "eq50": {}}
    codes = np.asarray(data["codes"]); buckets = np.asarray(data["labels"]["50"])
    for node in request["nodes"]:
        if node["terminal"]:
            continue
        table = tables["node_tables"][line_key(node["line"])]
        rows = slice(0, 1) if node["street"] == "turn" else slice(1, None)
        matrices = [("v1", codes, 255), ("eq50", buckets, 65535)]
        matrices += [("eq50-fit" + fold, np.asarray(matrix), 65535)
                     for fold, matrix in data.get("crossfit_labels", {}).items()]
        for metric, matrix, blocked in matrices:
            result.setdefault(metric, {})
            values = set(map(int, np.unique(matrix[rows]))) - {blocked}
            keys = {str(code): factored_key(node["template"], decode_descriptor(code)
                                           if metric == "v1" else ["equity-bucket", code])
                    for code in sorted(values)}
            if table in result[metric] and result[metric][table] != keys:
                raise ValueError("Aliased template mapping differs")
            result[metric][table] = keys
    return result
