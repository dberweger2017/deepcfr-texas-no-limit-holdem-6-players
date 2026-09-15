"""Small CPU networks, paper-style losses, and exact empirical fitting diagnostics."""

from contextlib import contextmanager
from hashlib import sha256
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.solver.neural.encoding import FEATURES
from src.solver.neural.memory import Reservoir


def stream_seed(seed: int, name: str, iteration: int = 0, player: int = 0) -> int:
    value = f"deep-cfr-v1/{seed}/{name}/{iteration}/{player}".encode()
    return int.from_bytes(sha256(value).digest()[:8], "big") % (2**63)


@contextmanager
def deterministic_cpu():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


class Network(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, features):
        return self.layers(features)


def new_network(hidden: int, seed: int, *, zero_output: bool = False) -> Network:
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = Network(hidden).float()
        if zero_output:
            nn.init.zeros_(model.layers[-1].weight)
            nn.init.zeros_(model.layers[-1].bias)
    return model


def strategy_probabilities(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)


def regret_matching(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.shape != mask.shape or not np.isfinite(values).all():
        raise ValueError("Expected finite advantage predictions")
    positive = np.maximum(values, 0) * mask
    totals = positive.sum(axis=1, keepdims=True)
    result = np.divide(positive, totals, out=np.zeros_like(positive), where=totals > 0)
    empty = totals[:, 0] <= 0
    # The paper selects the largest legal prediction when none is positive.
    best = np.where(mask, values, -np.inf).argmax(axis=1)
    result[empty, best[empty]] = 1
    return result


def weighted_loss(predictions, targets, mask, iterations, current_iteration):
    errors = ((predictions - targets) ** 2 * mask).sum(dim=1)
    return (errors * (2 * iterations / current_iteration)).mean()


def predict(
    model: Network, features: torch.Tensor, mask: torch.Tensor, *, strategy: bool
) -> np.ndarray:
    with torch.inference_mode():
        values = model(features)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite network output")
        if strategy:
            values = strategy_probabilities(values, mask)
        result = values.double().numpy()
        return result / result.sum(axis=1, keepdims=True) if strategy else result


def fitting_metrics(model, features, mask, memory, *, strategy):
    means, weights = memory.means(len(features))
    prediction = predict(model, features, mask, strategy=strategy)
    legal = mask.numpy()
    total = weights.sum()
    excess = float((weights[:, None] * (prediction - means) ** 2 * legal).sum() / total)
    ids = memory.infos[: memory.size]
    targets = memory.targets[: memory.size]
    noise = float(
        (
            memory.iterations[: memory.size, None]
            * (targets - means[ids]) ** 2
            * legal[ids]
        ).sum()
        / total
    )
    return {
        "excess_mse": excess,
        "sample_noise_mse": noise,
        "empirical_mse": excess + noise,
        "observed_information_sets": int((weights > 0).sum()),
        "stored_samples": memory.size,
        "seen_samples": memory.seen,
    }


def fit(
    memory: Reservoir,
    features: torch.Tensor,
    mask: torch.Tensor,
    *,
    hidden: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    iteration: int,
    seed: int,
    strategy: bool,
    deadline: float = float("inf"),
) -> tuple[Network, dict]:
    if not memory.size:
        raise ValueError("Cannot fit an empty memory")
    model = new_network(hidden, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(
        stream_seed(seed, "minibatches")
    )
    ids = torch.from_numpy(memory.infos[: memory.size].copy())
    targets = torch.from_numpy(memory.targets[: memory.size].copy())
    iterations = torch.from_numpy(memory.iterations[: memory.size].copy()).float()
    model.train()
    for _ in range(steps):
        if perf_counter() >= deadline:
            raise TimeoutError("Neural fitting exceeded the declared deadline")
        rows = torch.randint(memory.size, (batch_size,), generator=generator)
        selected = ids[rows]
        logits = model(features[selected])
        predictions = (
            strategy_probabilities(logits, mask[selected]) if strategy else logits
        )
        loss = weighted_loss(
            predictions, targets[rows], mask[selected], iterations[rows], iteration
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
    model.eval().requires_grad_(False)
    metrics = fitting_metrics(model, features, mask, memory, strategy=strategy)
    metrics["steps"] = steps
    return model, metrics
