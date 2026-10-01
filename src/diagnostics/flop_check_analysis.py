"""Projection and reporting semantics for exact-card diagnostic profiles."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from itertools import combinations
import json

import numpy as np

from src.diagnostics.exact_ranker import exact_seven_card
from src.diagnostics.flop_check import descriptor, factored_key
from src.blueprint.search import DECK


def uniform_river_equities(board):
    """Exhaustive equity against uniform, card-compatible opponent holdings.

    Sorted rank counts minus two card-specific rank counts avoid a dense
    holding-pair matrix. The own holding is removed once, including in ties.
    """
    if len(board) != 5 or len(set(board)) != 5:
        raise ValueError("Need five distinct river cards")
    available = [c for c in DECK if c not in board]
    holdings = tuple(combinations(available, 2))
    ranks = [exact_seven_card(tuple(pair) + tuple(board)) for pair in holdings]
    ordered = sorted(ranks); by_card = defaultdict(list)
    for holding, rank in zip(holdings, ranks, strict=True):
        for card in holding:
            by_card[card].append(rank)
    for values in by_card.values():
        values.sort()
    denominator = (len(available) - 2) * (len(available) - 3) / 2
    equity = []
    for (a, b), rank in zip(holdings, ranks, strict=True):
        wins = bisect_left(ordered, rank) - bisect_left(by_card[a], rank) - bisect_left(by_card[b], rank)
        ties = (bisect_right(ordered, rank) - bisect_left(ordered, rank)
                - bisect_right(by_card[a], rank) + bisect_left(by_card[a], rank)
                - bisect_right(by_card[b], rank) + bisect_left(by_card[b], rank) + 1)
        equity.append((wins + ties / 2) / denominator)
    return holdings, np.asarray(equity)


def emd_clusters(histograms, k, *, seed, iterations=100):
    """Deterministic mean-centroid clustering with cumulative-L1 assignment.

    This follows the requested EMD assignment/mean-centroid convention; it is
    not a claim that Euclidean k-means optimizes an EMD objective. Equal feature
    vectors always receive equal labels, including at assignment ties.
    """
    features = np.asarray(histograms, dtype=np.float64)
    if (features.ndim != 2 or not len(features) or k < 1
            or not np.isfinite(features).all() or (features < 0).any()
            or not np.allclose(features.sum(axis=1), 1)):
        raise ValueError("Invalid probability histograms")
    unique = np.unique(features, axis=0); count = min(k, len(unique))
    rng = np.random.default_rng(seed)
    centers = unique[rng.choice(len(unique), count, replace=False)].copy()
    labels = np.full(len(features), -1)
    cumulative = np.cumsum(features, axis=1)
    for _ in range(iterations):
        distances = np.abs(cumulative[:, None, :] - np.cumsum(centers, axis=1)[None, :, :]).sum(axis=2)
        updated = distances.argmin(axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for bucket in range(count):
            members = features[labels == bucket]
            if len(members):
                centers[bucket] = members.mean(axis=0)
    return labels, centers


def equity_quantiles(values, k):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or k < 1:
        raise ValueError("Invalid river equity values")
    # Equal equities stay together; arbitrary holding order cannot split ties.
    edges = np.quantile(values, np.arange(1, k) / k)
    return np.searchsorted(edges, values, side="right")


def projection(records, templates, *, bucket=None, per_line=False):
    """Pool own-range/own-policy reach over every runout and aliased line.

    Actual v1 hashes may alias different betting lines. Pooling only within a
    line produces a relaxation, not a feasible full-key v1 policy. Both modes
    are exposed, and only the full-key mode is called the feasible projection.
    Chance factors are constant at a fixed street and cancel in each group.
    """
    totals = {}; mass = defaultdict(float); mapped = []
    for row in records:
        line = json.dumps(row["line"], separators=(",", ":"))
        template = templates[line]; hands = row["holdings"]
        strategy = np.asarray(row["strategy"], dtype=float).reshape(-1, len(hands)).T
        weights = np.asarray(row["own_weights"], dtype=float)
        if (strategy.shape[0] != len(hands) or weights.shape != (len(hands),)
                or not np.isfinite(strategy).all() or not np.isfinite(weights).all()
                or (strategy < 0).any() or (weights < 0).any()
                or not np.allclose(strategy.sum(axis=1), 1, atol=2e-5)):
            raise ValueError("Invalid external profile/reach")
        groups = []
        for index, holding in enumerate(hands):
            if bucket is None:
                group = factored_key(template, descriptor(holding, row["board"]))
            else:
                group = (line, len(row["board"]), bucket(row, holding))
            if per_line:
                group = (line, group)
            groups.append(group)
            if group in totals and totals[group].shape != strategy[index].shape:
                raise ValueError("Merged projection keys have different action dimensions")
            totals.setdefault(group, np.zeros(strategy.shape[1]))
            totals[group] += weights[index] * strategy[index]
            mass[group] += weights[index]
        mapped.append((row, groups, strategy.shape[1]))
    result = []
    for row, groups, actions in mapped:
        probabilities = np.stack([totals[g] / mass[g] if mass[g] else
                                  np.full(actions, 1 / actions) for g in groups])
        result.append(dict(row, strategy=probabilities.T.ravel().tolist()))
    return result


def bootstrap_spots(rows, metric, *, seed=202610010905, resamples=2000):
    """Resample independent roots, averaging repeated lineages within a root."""
    grouped = defaultdict(list); weights = {}
    for row in rows:
        value = row.get(metric)
        if value is None or not row.get("decision_eligible", False):
            continue
        if not np.isfinite(value):
            raise ValueError("Non-finite spot metric")
        grouped[row["spot"]].append(value)
        weight = row.get("reach_weight", 1.0)
        if row["spot"] in weights and weights[row["spot"]] != weight:
            raise ValueError("Conflicting common-corpus reach weights")
        weights[row["spot"]] = weight
    keys = sorted(grouped)
    if not keys:
        return {"mean": None, "ci95": None, "independent_spots": 0}
    values = np.asarray([np.mean(grouped[k]) for k in keys])
    w = np.asarray([weights[k] for k in keys], dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Need positive finite corpus weights")
    mean = float(np.average(values, weights=w))
    if len(keys) < 2:
        return {"mean": mean, "ci95": None, "independent_spots": len(keys)}
    draws = np.random.default_rng(seed).integers(0, len(keys), (resamples, len(keys)))
    estimates = (values[draws] * w[draws]).sum(axis=1) / w[draws].sum(axis=1)
    return {"mean": mean, "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "independent_spots": len(keys), "resamples": resamples, "seed": seed}


def decision_rule(mean_bp, mean_projection, flop_share, fold_gap_pp, *, eligible):
    if not eligible:
        return {"classification": "pending", "reason": "gates, exclusions or frozen protocol unmet"}
    ratio = mean_projection / mean_bp if mean_bp > 0 else None
    if flop_share < 0.10:
        classification = "H3"
    elif ratio is None:
        classification = "undefined"
    elif ratio >= 0.7 and flop_share >= 0.25:
        classification = "H1-consistent"
    elif ratio <= 0.3:
        classification = "H2-consistent"
    else:
        classification = "mixed"
    return {"classification": classification, "R": ratio,
            "H0_overfold": abs(fold_gap_pp) <= 3,
            "interpretation": "projection is an upper bound, not an abstraction lower bound"}
