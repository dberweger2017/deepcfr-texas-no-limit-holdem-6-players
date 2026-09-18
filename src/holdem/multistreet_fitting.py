"""Independent fit processes with durable completed-arm checkpoints."""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path

import torch

from src.holdem.multistreet_campaign import _atomic_json
from src.holdem.multistreet_collection import single_thread_worker
from src.solver.neural.network import deterministic_cpu


def _fit(job, targets, plan, deadline, cache_dir, identity):
    from scripts.check_multistreet_representation import fit_arm

    seed, variant = job
    path = Path(cache_dir) / f"{variant}-{seed}.pt"
    manifest_path = path.with_suffix(".json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["identity"] != identity
            or manifest["sha256"] != sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("Completed fit cache changed")
        result = torch.load(path, weights_only=False)
    else:
        with deterministic_cpu():
            result = fit_arm(variant, seed, targets, plan, deadline)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.partial")
        torch.save(result, temporary)
        os.replace(temporary, path)
        _atomic_json(
            manifest_path,
            {"identity": identity, "sha256": sha256(path.read_bytes()).hexdigest()},
        )
    return seed, variant, *result


def fit_all(targets, plan, deadline, *, cache_dir, identity, workers=1, progress=None):
    if workers < 1:
        raise ValueError("Fit workers must be positive")
    if set(targets) != {"train", "tuning"}:
        raise ValueError("Fit workers may receive only train and tuning data")
    jobs = [(seed, variant) for seed in plan["seeds"] for variant in plan["variants"]]
    results = {}
    if workers == 1:
        for job in jobs:
            results[job] = _fit(job, targets, plan, deadline, cache_dir, identity)
            if progress:
                progress(len(results), len(jobs))
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            mp_context=get_context("spawn"),
            initializer=single_thread_worker,
        ) as executor:
            futures = {
                executor.submit(
                    _fit, job, targets, plan, deadline, cache_dir, identity
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                if progress:
                    progress(len(results), len(jobs))
    return [results[job] for job in jobs]
