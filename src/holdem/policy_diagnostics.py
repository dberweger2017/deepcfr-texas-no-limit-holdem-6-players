"""Descriptive policy changes on retained decisions, independent of target labels."""

from collections import defaultdict
from math import fsum, log
from time import perf_counter

import torch

from src.holdem.betting import ActionScores


def policy_change(reference: ActionScores, candidate: ActionScores) -> dict:
    if (
        reference.candidates != candidate.candidates
        or reference.candidates.decision.source != candidate.candidates.decision.source
    ):
        raise ValueError("Compare policies on the same public decision")
    before = reference.probabilities().tolist()
    after = candidate.probabilities().tolist()
    kinds = defaultdict(float)
    for action, probability in zip(candidate.candidates.actions, after):
        kinds[action.kind.value] += probability
    return {
        "tv": 0.5 * fsum(abs(p - q) for p, q in zip(before, after)),
        "entropy_nats": -fsum(p * log(p) for p in after if p > 0),
        "fallback": bool(candidate.regrets.max() <= 0),
        "action_mass": dict(kinds),
    }


def summarize_changes(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot summarize empty replay coverage")
    count = len(records)
    kinds = ("fold", "check", "call", "raise")
    return {
        "records": count,
        "mean_tv": fsum(r["tv"] for r in records) / count,
        "tv_above_005_fraction": sum(r["tv"] > 0.05 for r in records) / count,
        "tv_above_010_fraction": sum(r["tv"] > 0.10 for r in records) / count,
        "fallback_fraction": sum(r["fallback"] for r in records) / count,
        "mean_entropy_nats": fsum(r["entropy_nats"] for r in records) / count,
        "mean_action_mass": {
            kind: fsum(r["action_mass"].get(kind, 0) for r in records) / count
            for kind in kinds
        },
    }


def replay_policy_changes(memory, models, *, control, deadline=float("inf")):
    if control not in models or not memory:
        raise ValueError("Provide replay and a named control model")
    grouped = {name: defaultdict(list) for name in models}
    items = memory.items
    with torch.inference_mode():
        for start in range(0, len(items), 32):
            if perf_counter() >= deadline:
                raise TimeoutError("Replay diagnostics exceeded the deadline")
            samples = items[start : start + 32]
            candidates = [s.target.candidates for s in samples]
            predictions = {name: model(candidates) for name, model in models.items()}
            for name, scores in predictions.items():
                for ref, score in zip(predictions[control], scores):
                    street = score.candidates.decision.source.street.value
                    grouped[name][street].append(policy_change(ref, score))
    return {
        name: {
            "all": summarize_changes([r for rows in streets.values() for r in rows]),
            "streets": {
                street: summarize_changes(rows) for street, rows in streets.items()
            },
        }
        for name, streets in grouped.items()
    }
