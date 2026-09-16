"""One-phase regression with explicit root and uniform replay normalization."""

from collections.abc import Sequence
from math import fsum, isclose, isfinite

import torch

from src.holdem.betting import ActionScores, betting_loss
from src.holdem.outcome_sampling import SampledDecision
from src.holdem.targets import CandidateTargets


def sampled_betting_loss(
    scores: Sequence[ActionScores],
    decisions: Sequence[SampledDecision],
    *,
    roots: int,
    population: int | None = None,
    iteration_weight: float = 1.0,
) -> torch.Tensor:
    """Sum conditional errors / own reach; normalize by roots, including empty ones.

    Set population to the full stream size for uniform replay minibatches. This
    function covers one phase; mixed iterations need their own root counts.
    """
    if not scores or len(scores) != len(decisions):
        raise ValueError("Match a nonempty set of predictions and sampled decisions")
    if type(roots) is not int or roots < 1:
        raise ValueError("Root count must include every scheduled root")
    population = len(decisions) if population is None else population
    if type(population) is not int or population < 1:
        raise ValueError("Provide a positive replay stream size")
    if (
        type(iteration_weight) not in (int, float)
        or not isfinite(iteration_weight)
        or iteration_weight <= 0
    ):
        raise ValueError("Iteration weight must be positive and finite")
    losses = []
    for score, decision in zip(scores, decisions):
        if not isinstance(decision, SampledDecision):
            raise TypeError("Expected sampled decisions, not complete branch targets")
        count = len(decision.candidates.actions)
        if (
            len(decision.policy) != count
            or len(decision.values_bb) != count
            or any(not isfinite(x) or x < 0 for x in decision.policy)
            or not isclose(fsum(decision.policy), 1, rel_tol=0, abs_tol=1e-9)
        ):
            raise ValueError("Invalid sampled policy or value dimensions")
        reach = decision.own_sample_reach
        if not 0 < reach <= 1 or not isfinite(1 / reach):
            raise ValueError("Invalid own sampling reach")
        target = CandidateTargets(
            decision.candidates,
            decision.policy,
            decision.values_bb,
            decision.regrets_bb,
        )
        losses.append(betting_loss([score], [target]) / reach)
    # N/m corrects uniform replay sampling; a random observed weight sum would bias it.
    loss = torch.stack(losses).sum() * (
        iteration_weight * population / len(decisions) / roots
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite sampled regression loss")
    return loss


def sampled_replay_loss(scores, samples, *, iteration, population):
    """Estimate the linearly weighted mean of scheduled per-iteration root sums."""
    from src.holdem.replay import SampledReplaySample

    if (
        type(iteration) is not int
        or iteration < 1
        or not samples
        or len(scores) != len(samples)
        or type(population) is not int
        or population < 1
    ):
        raise ValueError("Invalid sampled replay batch or population")
    if any(
        type(s) is not SampledReplaySample or not 1 <= s.iteration <= iteration
        for s in samples
    ):
        raise ValueError("Expected sampled replay from completed iterations")
    total_weight = iteration * (iteration + 1) / 2
    terms = [
        sampled_betting_loss(
            [score],
            [s.target],
            roots=s.roots,
            iteration_weight=s.iteration / total_weight,
        )
        for score, s in zip(scores, samples)
    ]
    result = torch.stack(terms).sum() * (population / len(samples))
    if not torch.isfinite(result):
        raise FloatingPointError("Non-finite sampled replay loss")
    return result
