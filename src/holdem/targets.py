"""Per-candidate supervision from evaluated branches, never predicted bet sizes."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isclose, isfinite

from src.game.types import Action
from src.holdem.actions import BetCandidates


@dataclass(frozen=True, slots=True)
class CandidateTargets:
    candidates: BetCandidates
    policy: tuple[float, ...]
    values_bb: tuple[float, ...]
    regrets_bb: tuple[float, ...]


def action_targets(
    candidates: BetCandidates,
    policy: Sequence[float],
    branch_values: Mapping[Action, float],
) -> CandidateTargets:
    """Branch values are the acting player's estimated net chip payoffs."""
    probabilities = tuple(float(p) for p in policy)
    if (
        len(probabilities) != len(candidates.actions)
        or any(not isfinite(p) or p < 0 for p in probabilities)
        or not isclose(fsum(probabilities), 1, rel_tol=0, abs_tol=1e-9)
    ):
        raise ValueError("Provide a probability for every candidate, summing to one")
    if set(branch_values) != set(candidates.actions):
        raise ValueError(
            "Evaluate every candidate exactly once, including zero-probability actions"
        )
    bb = candidates.decision.source.big_blind
    values = tuple(float(branch_values[action]) / bb for action in candidates.actions)
    if not all(isfinite(v) for v in values):
        raise ValueError("Branch values must be finite chip payoffs")
    baseline = fsum(p * value for p, value in zip(probabilities, values))
    regrets = tuple(value - baseline for value in values)
    if not all(isfinite(v) for v in regrets):
        raise ValueError("Non-finite regret target")
    return CandidateTargets(candidates, probabilities, values, regrets)
