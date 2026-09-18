"""Collect independent reference contexts in separate CPU processes."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from time import perf_counter

from src.holdem.actions import bet_candidates
from src.holdem.multistreet_campaign import PROFILES
from src.holdem.multistreet_reference import split_specs
from src.holdem.targets import CandidateTargets


def single_thread_worker():
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def collect_context(plan, cache, index, deadline):
    if perf_counter() >= deadline:
        raise TimeoutError("Reference collection deadline reached")
    ((context, group),) = split_specs(plan, context_filter=lambda row, i: i == index)
    cached = [
        cache.get_or_compute(
            context, profile, max_nodes=plan["max_tree_nodes"], deadline=deadline
        )
        for profile in PROFILES
    ]
    first, second = cached
    candidates = bet_candidates(context.worlds[0].observe(context.hero_seat))
    target = CandidateTargets(
        candidates,
        tuple(first["target"]["policy"]),
        tuple(
            (a + 2 * b) / 3
            for a, b in zip(
                first["target"]["values_bb"], second["target"]["values_bb"], strict=True
            )
        ),
        tuple(
            (a + 2 * b) / 3
            for a, b in zip(
                first["target"]["regrets_bb"],
                second["target"]["regrets_bb"],
                strict=True,
            )
        ),
    )
    # Native engine objects stay in the worker. Training only needs the legal
    # observation, targets, and paired values, all plain Python data.
    return {
        "name": context.name,
        "split": context.split,
        "street": context.street,
        "situation": "facing" if context.facing else "open",
        "group": repr(group),
        "target": target,
        "world_count": len(context.worlds),
        "uncertainty": [entry["uncertainty_status"] for entry in cached],
        "action_se": [entry["action_standard_error_bb"] for entry in cached],
        "world_action_values_bb": [
            [(a + 2 * b) / 3 for a, b in zip(x, y, strict=True)]
            for x, y in zip(
                first["world_action_values_bb"],
                second["world_action_values_bb"],
                strict=True,
            )
        ],
        "cache_paths": [str(cache.path(context, profile)) for profile in PROFILES],
        "reference_nodes": sum(entry["nodes"] for entry in cached),
        "reference_worker_seconds": sum(entry["seconds"] for entry in cached),
    }


def collect_rows(
    plan, cache, *, deadline, context_filter=None, reference_workers=1, progress=None
):
    if reference_workers < 1:
        raise ValueError("reference_workers must be positive")
    indices = [
        index
        for index, row in enumerate(plan["contexts"])
        if context_filter is None
        or context_filter(
            {
                "name": f"{row['street']}-{index}",
                "split": row["split"],
                "street": row["street"],
                "facing": bool(row.get("facing", False)),
            }
        )
    ]
    completed = {}

    def record(index, row):
        completed[index] = row
        if progress:
            progress(len(completed), len(indices), row)

    if reference_workers == 1:
        for index in indices:
            record(index, collect_context(plan, cache, index, deadline))
    else:
        with ProcessPoolExecutor(
            max_workers=reference_workers,
            mp_context=get_context("spawn"),
            initializer=single_thread_worker,
        ) as executor:
            remaining = iter(indices)
            pending = {}
            for _ in range(min(len(indices), 2 * reference_workers)):
                index = next(remaining)
                pending[
                    executor.submit(collect_context, plan, cache, index, deadline)
                ] = index
            try:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        record(index, future.result())
                        following = next(remaining, None)
                        if following is not None:
                            pending[
                                executor.submit(
                                    collect_context, plan, cache, following, deadline
                                )
                            ] = following
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
    return [completed[index] for index in indices]
