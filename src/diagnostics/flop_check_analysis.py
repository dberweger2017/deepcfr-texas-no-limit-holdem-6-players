"""Projection and reporting semantics for exact-card diagnostic profiles."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from itertools import combinations
import json

import numpy as np

from src.diagnostics.exact_ranker import exact_seven_card
from src.diagnostics.flop_check import descriptor, descriptor_code, factored_key, line_key
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
        updated = np.empty(len(features), dtype=int)
        centroid_cdf = np.cumsum(centers, axis=1)
        for start in range(0, len(features), 512):
            distances = np.abs(cumulative[start:start + 512, None, :] - centroid_cdf[None, :, :]).sum(axis=2)
            updated[start:start + 512] = distances.argmin(axis=1)
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
        line = line_key(row["line"])
        template = templates[line]; hands = row["holdings"]
        strategy = np.asarray(row["strategy"], dtype=float).reshape(-1, len(hands)).T
        weights = np.asarray(row["own_weights"], dtype=float)
        sums = strategy.sum(axis=1)
        blocked = np.asarray([bool(set(h).intersection(row["board"])) for h in hands])
        if np.any(blocked & (weights > 0)):
            raise ValueError("External profile gives a blocked holding positive reach")
        strategy[blocked | ((weights == 0) & (sums == 0))] = 1 / strategy.shape[1]
        if (strategy.shape[0] != len(hands) or weights.shape != (len(hands),)
                or not np.isfinite(strategy).all() or not np.isfinite(weights).all()
                or (strategy < 0).any() or (weights < 0).any()
                or not np.allclose(strategy.sum(axis=1), 1, atol=2e-5)):
            raise ValueError("Invalid external profile/reach")
        groups = []
        for index, holding in enumerate(hands):
            if blocked[index]:
                group = ("blocked", line, tuple(row["board"]), tuple(holding))
            elif bucket is None:
                group = factored_key(template, descriptor(holding, row["board"]))
            else:
                group = factored_key(template, ["equity-bucket", bucket(row, holding)])
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
    grouped = defaultdict(list); weights = {}; strata = {}
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
        stratum = json.dumps(row.get("stratum", (row.get("kind"), row.get("button"))), sort_keys=True)
        if row["spot"] in strata and strata[row["spot"]] != stratum:
            raise ValueError("Conflicting sampling strata for one root")
        strata[row["spot"]] = stratum
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
    cells = defaultdict(list)
    for index, key in enumerate(keys):
        cells[strata[key]].append(index)
    if any(len(cell) < 2 for cell in cells.values()):
        return {"mean": mean, "ci95": None, "independent_spots": len(keys),
                "reason": "A sampling stratum has fewer than two independent roots"}
    rng = np.random.default_rng(seed)
    draws = np.concatenate([np.asarray(cell)[rng.integers(0, len(cell), (resamples, len(cell)))]
                            for _, cell in sorted(cells.items())], axis=1)
    estimates = (values[draws] * w[draws]).sum(axis=1) / w[draws].sum(axis=1)
    return {"mean": mean, "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "independent_spots": len(keys), "resamples": resamples, "seed": seed,
            "sampling_strata": len(cells)}


def blueprint_locks(records, request, tables):
    """Expand factored policy tables with explicit solver action reordering."""
    nodes = {line_key(n["line"]): n
             for n in request["nodes"] if not n["terminal"]}
    locks = []
    for row in records:
        line = line_key(row["line"]); node = nodes[line]
        table = tables["tables"][tables["node_tables"][line]]
        canonical = lambda a: json.dumps(a, sort_keys=True)
        if sorted(row["actions"], key=canonical) != sorted(node["actions"], key=canonical):
            raise ValueError("Solver lock has a different native action set")
        reorder = [node["actions"].index(a) for a in row["actions"]]
        probabilities = []
        for holding in row["holdings"]:
            if set(holding).intersection(row["board"]):
                p = [1 / len(reorder)] * len(reorder)
            else:
                code = str(descriptor_code(descriptor(holding, row["board"])))
                p = table["rows"][code]
            probabilities.append([p[i] for i in reorder])
        locks.append(dict(row, strategy=np.asarray(probabilities).T.ravel().tolist()))
    return locks


def compatible_reach(holdings, weights, opponent_holdings, opponent_weights):
    """Exact joint range marginal in O(number of hands), including blockers."""
    own = np.asarray(weights, dtype=float); other = np.asarray(opponent_weights, dtype=float)
    if (not np.isfinite(own).all() or not np.isfinite(other).all()
            or (own < 0).any() or (other < 0).any()):
        raise ValueError("Invalid range reach")
    total = float(other.sum()); marginal = defaultdict(float); same = defaultdict(float)
    for hand, weight in zip(opponent_holdings, other, strict=True):
        a, b = hand; marginal[a] += weight; marginal[b] += weight
        same[tuple(sorted(hand))] += weight
    compatible = np.asarray([total - marginal[a] - marginal[b] + same[tuple(sorted((a, b)))]
                             for a, b in holdings])
    if np.min(compatible, initial=0) < -1e-12:
        raise ValueError("Negative compatible reach")
    return own * np.maximum(compatible, 0)


def overfold_node(row, blueprint, *, equity, opponent_holdings, opponent_weights, template=None):
    """Compare policies on the same equilibrium-conditioned joint range."""
    actions = row["actions"]
    folds = [i for i, a in enumerate(actions) if a["kind"] == "Fold"]
    if not folds or len(row["board"]) != 3:
        raise ValueError("Overfold tables require a flop node facing a bet")
    hands = row["holdings"]; n = len(hands); fold = folds[0]
    equilibrium = np.asarray(row["strategy"]).reshape(-1, n)[fold]
    direct = np.asarray(blueprint["strategy"]).reshape(-1, n)[fold]
    reach = compatible_reach(hands, row["own_weights"], opponent_holdings, opponent_weights)
    if reach.sum() <= 0:
        return {"reached": False, "fold_bp": None, "fold_eq": None, "groups": []}
    groups = defaultdict(lambda: [0.0, 0.0, 0.0]); excess_hands = []
    for hand, weight, bp, eq in zip(hands, reach, direct, equilibrium, strict=True):
        decile = min(9, int(equity(hand) * 10))
        key = (decile, descriptor(tuple(hand), row["board"]))
        groups[key][0] += weight; groups[key][1] += weight * bp
        groups[key][2] += weight * max(bp - eq, 0)
        if weight > 0 and bp > eq:
            excess_hands.append({"hand": list(hand), "equity_decile": decile,
                                 "equity": float(equity(hand)), "reach_mass": float(weight),
                                 "fold_bp": float(bp), "fold_eq": float(eq),
                                 "excess_fold_probability": float(bp - eq),
                                 "v1_key": factored_key(template, key[1]) if template else None})
    return {"reached": True, "fold_bp": float(np.average(direct, weights=reach)),
            "fold_eq": float(np.average(equilibrium, weights=reach)),
            "reach_mass": float(reach.sum()), "conditioning": "common equilibrium joint reach",
            "excess_fold_hands": excess_hands,
            "groups": [{"equity_decile": key[0], "v1_descriptor": list(key[1]),
                        "v1_key": factored_key(template, key[1]) if template else None,
                        "reach_mass": values[0], "blueprint_fold_mass": values[1],
                        "excess_blueprint_fold_mass": values[2]} for key, values in sorted(groups.items())]}


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
