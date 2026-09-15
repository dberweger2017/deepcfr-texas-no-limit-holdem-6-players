"""Reproduce one bounded all-role collection phase; no model fitting or promotion."""

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from time import perf_counter

import torch

from src.arena.artifacts import environment, git, source_fingerprint
from src.game.hand import Table
from src.holdem.betting import BettingNetwork
from src.holdem.collection import collect_phase
from src.holdem.policy import FrozenProfile
from src.solver.neural.network import deterministic_cpu, stream_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, choices=(4, 5, 6), default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", choices=("uniform", "neural"), default="uniform")
    parser.add_argument("--max-nodes", type=int, default=10_000)
    parser.add_argument("--max-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.seed < 0 or args.max_nodes <= 0 or not 0 < args.max_seconds <= 900:
        parser.error(
            "Use a nonnegative seed, positive node budget and at most 900 seconds per phase"
        )
    models = []
    for seat in range(args.players):
        model = None
        if args.policy == "neural":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(
                    stream_seed(args.seed, "holdem-collection-smoke", player=seat)
                )
                model = BettingNetwork(width=16)
        models.append(model)
    profile = FrozenProfile(models)
    table = Table(
        tuple(f"player-{i}" for i in range(args.players)),
        (200,) * args.players,
        small_blind=1,
        big_blind=2,
        chip_unit="1",
    )
    started = perf_counter()
    with deterministic_cpu():
        first = collect_phase(
            table,
            profile,
            iteration=1,
            seed=args.seed,
            max_nodes=args.max_nodes,
            max_seconds=args.max_seconds,
        )
        second = collect_phase(
            table,
            profile,
            iteration=1,
            seed=args.seed,
            max_nodes=args.max_nodes,
            max_seconds=args.max_seconds,
        )
    if first != second:
        raise RuntimeError("Repeated collection differs from the original")
    data = json.dumps(
        asdict(first), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    print(
        json.dumps(
            {
                "check": "holdem-collection-smoke-v1",
                "revision": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain")),
                "source_sha256": source_fingerprint(),
                "environment": environment(),
                "config": vars(args),
                "stack_bb": 100,
                "collection_schema": first.schema,
                "profile_sha256": profile.fingerprint,
                "collection_sha256": sha256(data).hexdigest(),
                "reproduced": True,
                "elapsed_seconds": perf_counter() - started,
                "traversals": [
                    {
                        "seat": r.root.seat,
                        "nodes": r.nodes,
                        "terminals": r.terminals,
                        "targets": len(r.targets),
                        "value_bb": r.value_bb,
                    }
                    for r in first.traversals
                ],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
