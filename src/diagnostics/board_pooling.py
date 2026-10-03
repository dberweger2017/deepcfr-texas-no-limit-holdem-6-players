"""Cross-board projections for an immutable diagnostic, never a trainer."""

from math import fsum, isfinite
import json

from src.game.observation import ActionTaken


def public_line(history, button):
    """Exact observed amounts and relative seats; board identities are omitted."""
    return json.dumps([
        [event.street.value, (event.seat - button) % 2,
         event.action.kind.value, event.action.raise_to]
        for event in history if isinstance(event, ActionTaken)
    ], separators=(",", ":"))


def pool_statistics(records):
    """Pool sufficient statistics by actual key and action name, per lineage.

    Each record's own-range/own-action-reach statistics are multiplied by its
    declared board weight exactly once. Street chance constants cancel because
    every root starts at the same turn line and has the same deck size.
    """
    totals = {}
    identities = set()
    for record in records:
        identity = (record["lineage"], record["spot"])
        if identity in identities:
            raise ValueError("Duplicate root-lineage projection statistics")
        identities.add(identity)
        weight = record["board_weight"]
        if not isfinite(weight) or weight <= 0:
            raise ValueError("Invalid board weight")
        seen = set()
        for row in record["groups"]:
            group = (record["lineage"], row["metric"], row["key"])
            if group in seen:
                raise ValueError("Duplicate sufficient-statistic group")
            seen.add(group)
            names = tuple(row["names"])
            mass = row["mass"]
            values = row["action_mass"]
            if (not names or len(set(names)) != len(names)
                    or len(values) != len(names) or mass < 0
                    or not all(isfinite(v) and v >= 0 for v in (mass, *values))
                    or abs(fsum(values) - mass) > 2e-5 * max(1, mass)):
                raise ValueError("Invalid projection sufficient statistics")
            if group not in totals:
                totals[group] = {"names": names, "mass": 0.,
                                 "action_mass": dict.fromkeys(names, 0.), "roots": 0}
            entry = totals[group]
            if set(entry["names"]) != set(names):
                raise ValueError("A pooled information key has incompatible menus")
            entry["mass"] += weight * mass
            entry["roots"] += 1
            for name, value in zip(names, values, strict=True):
                entry["action_mass"][name] += weight * value
    output = []
    for (lineage, metric, key), entry in sorted(totals.items()):
        names = entry["names"]
        mass = entry["mass"]
        p = ([entry["action_mass"][name] / mass for name in names]
             if mass else [1 / len(names)] * len(names))
        if not all(isfinite(v) for v in p) or abs(fsum(p) - 1) > 2e-5:
            raise ValueError("Non-finite or unnormalized pooled policy")
        output.append({"lineage": lineage, "metric": metric, "key": key,
                       "names": list(names), "probabilities": p,
                       "mass": mass, "roots": entry["roots"]})
    return {"format": "hu20-board-pooling-policy-v1", "groups": output,
            "root_lineage_records": len(identities),
            "zero_mass_rule": "uniform within the actual menu"}


def readout(bp, local, pooled):
    """Signed placement within the gap; projections are feasible witnesses."""
    if not all(isfinite(v) for v in (bp, local, pooled)):
        raise ValueError("Non-finite pooling readout")
    gap = bp - local
    position = (pooled - local) / gap if gap >= .1 else None
    label = ("insufficient blueprint-minus-per-root gap" if position is None
             else "board-pooling consistent" if position >= .7
             else "trainer/coverage consistent" if position <= .3 else "mixed")
    return {"gap_bb": gap, "board_pooling_difference_bb": pooled - local,
            "placement": position, "classification": label,
            "limitation": "feasible projection losses, not an abstraction lower bound"}


def crossfit_policies(records, folds):
    """Fit evaluation-fold policies using only the opposite frozen half."""
    if set(folds.values()) != {0, 1}:
        raise ValueError("Exactly two frozen folds are required")
    if any(r["spot"] not in folds for r in records):
        raise ValueError("Root missing from frozen split")
    result = {}
    for evaluation_fold in (0, 1):
        training = [r for r in records if folds[r["spot"]] != evaluation_fold]
        if not training:
            raise ValueError("No eligible training roots in a frozen half")
        policy = pool_statistics(training)
        result[str(evaluation_fold)] = dict(policy, evaluation_fold=evaluation_fold,
                training_fold=1-evaluation_fold, training_spots=sorted({r["spot"] for r in training}))
    return result
