"""Fresh role fits with uniform minibatches and linear iteration weights."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, isnan
from random import Random
from time import perf_counter

import torch

from src.holdem.betting import ActionScores, BettingNetwork, betting_loss
from src.holdem.replay import ReplaySample, RoleReservoir
from src.solver.neural.network import deterministic_cpu, stream_seed


@dataclass(frozen=True, slots=True)
class FitConfig:
    width: int = 128
    steps: int = 128
    batch_size: int = 64
    learning_rate: float = 0.001
    diagnostic_samples: int = 128

    def __post_init__(self):
        if any(
            type(v) is not int or v < 1
            for v in (self.width, self.steps, self.batch_size, self.diagnostic_samples)
        ):
            raise ValueError("Fit dimensions and counts must be positive integers")
        if (
            type(self.learning_rate) not in (int, float)
            or not isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("Learning rate must be positive and finite")


@dataclass(frozen=True, slots=True)
class FitMetrics:
    steps: int
    diagnostic_samples: int
    loss_before: float
    loss_after: float


def weighted_betting_loss(
    scores: Sequence[ActionScores], samples: Sequence[ReplaySample], iteration: int
) -> torch.Tensor:
    if (
        type(iteration) is not int
        or iteration < 1
        or not samples
        or len(scores) != len(samples)
    ):
        raise ValueError("Match predictions to replay records and a positive iteration")
    if any(
        type(s.iteration) is not int or not 1 <= s.iteration <= iteration
        for s in samples
    ):
        raise ValueError("Replay cannot contain future or invalid iterations")
    losses = torch.stack(
        [
            betting_loss([score], [sample.target])
            for score, sample in zip(scores, samples)
        ]
    )
    weights = losses.new_tensor(
        [2 * sample.iteration / iteration for sample in samples]
    )
    result = (losses * weights).mean()
    if not torch.isfinite(result):
        raise FloatingPointError("Non-finite weighted betting loss")
    return result


def fit_role(
    memory: RoleReservoir,
    config: FitConfig,
    *,
    iteration: int,
    seed: int,
    deadline: float = float("inf"),
) -> tuple[BettingNetwork, FitMetrics]:
    if (
        not memory
        or type(seed) is not int
        or seed < 0
        or type(iteration) is not int
        or iteration < 1
        or isnan(deadline)
    ):
        raise ValueError(
            "Fitting needs replay, a nonnegative seed and a positive iteration"
        )
    if any(item.iteration > iteration for item in memory.items):
        raise ValueError("Replay contains future iterations")
    batches = Random(stream_seed(seed, "holdem-fit-batches", iteration, memory.role))
    diagnostics = Random(
        stream_seed(seed, "holdem-fit-diagnostics", iteration, memory.role)
    )
    fixed = tuple(
        diagnostics.sample(memory.items, min(config.diagnostic_samples, len(memory)))
    )
    with deterministic_cpu(), torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(
            stream_seed(seed, "holdem-fit-initial", iteration, memory.role)
        )
        model = BettingNetwork(config.width).cpu().float()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        def loss(samples):
            return weighted_betting_loss(
                model([s.target.candidates for s in samples]), samples, iteration
            )

        with torch.no_grad():
            before = float(loss(fixed))
        for _ in range(config.steps):
            if perf_counter() >= deadline:
                raise TimeoutError("Role fitting exceeded the iteration deadline")
            samples = memory.sample(config.batch_size, batches)
            error = loss(samples)
            optimizer.zero_grad(set_to_none=True)
            error.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
        model.eval().requires_grad_(False)
        with torch.no_grad():
            after = float(loss(fixed))
        if perf_counter() >= deadline:
            raise TimeoutError("Role fitting exceeded the iteration deadline")
    return model, FitMetrics(config.steps, len(fixed), before, after)
