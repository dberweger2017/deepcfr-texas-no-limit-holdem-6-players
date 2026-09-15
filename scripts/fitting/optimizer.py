"""Paired optimization experiments with the original replay-weighted objective."""

from hashlib import sha256
from math import cos, pi
from time import perf_counter

import torch

from src.solver.neural.network import (
    new_network,
    strategy_probabilities,
    stream_seed,
    weighted_loss,
)


def model_hash(model):
    value = sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def fitting_seed(seed, replicate, iteration):
    return stream_seed(
        seed,
        "strategy-fit" if replicate == 0 else "strategy-fitting-v1",
        iteration,
        replicate,
    )


def rate_at(recipe, step, steps):
    rate = recipe["learning_rate"]
    if rate["kind"] == "constant":
        return rate["value"]
    return rate["end"] + 0.5 * (rate["start"] - rate["end"]) * (
        1 + cos(pi * step / max(1, steps - 1))
    )


def grouped_objective(predictions, means, weights, mask, sample_count, iteration):
    return (weights[:, None] * (predictions - means).square() * mask).sum() * (
        2 / (sample_count * iteration)
    )


def optimize(solver, memory, recipe, fixed, seed, iteration, deadline, observe):
    model = new_network(fixed["strategy_hidden"], seed)
    initial_hash = model_hash(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=rate_at(recipe, 0, fixed["steps"])
    )
    generator = torch.Generator(device="cpu").manual_seed(
        stream_seed(seed, "minibatches")
    )
    ids = torch.from_numpy(memory.infos[: memory.size].copy())
    targets = torch.from_numpy(memory.targets[: memory.size].copy())
    times = torch.from_numpy(memory.iterations[: memory.size].copy()).float()
    means, weights = memory.means(len(solver.features))
    means, weights = torch.from_numpy(means).float(), torch.from_numpy(weights).float()
    batches = sha256()
    clipped, maximum_norm, norm_sum = 0, 0.0, 0.0
    model.train()
    for step in range(fixed["steps"]):
        if perf_counter() >= deadline:
            raise TimeoutError("Strategy fit exceeded its deadline")
        rate = rate_at(recipe, step, fixed["steps"])
        optimizer.param_groups[0]["lr"] = rate
        if recipe["objective"] == "sampled-original":
            rows = torch.randint(
                memory.size, (fixed["batch_size"],), generator=generator
            )
            batches.update(rows.numpy().tobytes())
            selected = ids[rows]
            logits = model(solver.features[selected])
            probabilities = strategy_probabilities(logits, solver.mask[selected])
            loss = weighted_loss(
                probabilities,
                targets[rows],
                solver.mask[selected],
                times[rows],
                iteration,
            )
        else:
            probabilities = strategy_probabilities(model(solver.features), solver.mask)
            loss = grouped_objective(
                probabilities, means, weights, solver.mask, memory.size, iteration
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite fitting loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
        )
        optimizer.step()
        clipped += norm > 1
        maximum_norm, norm_sum = max(maximum_norm, norm), norm_sum + norm
        if step + 1 in fixed["evaluation_steps"]:
            model.eval()
            observe(
                step + 1,
                model,
                {
                    "learning_rate": rate,
                    "clipped_updates": clipped,
                    "maximum_gradient_norm": maximum_norm,
                    "mean_gradient_norm": norm_sum / (step + 1),
                    "initial_weights_sha256": initial_hash,
                    "minibatch_indices_sha256": batches.hexdigest()
                    if recipe["objective"] == "sampled-original"
                    else None,
                },
            )
            model.train()
    return model.eval().requires_grad_(False)
