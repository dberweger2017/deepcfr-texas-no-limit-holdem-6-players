"""Check two reproducible collect-and-fit iterations, without model promotion."""

import argparse
import json
from dataclasses import asdict
from time import perf_counter

from src.arena.artifacts import environment, git, source_fingerprint
from src.game.hand import Table
from src.holdem.fitting import FitConfig
from src.holdem.training import HoldemTrainer, TrainConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, choices=(4, 5, 6), default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.seed < 0 or not 0 < args.max_seconds <= 900:
        parser.error("Use a nonnegative seed and at most 900 seconds per iteration")
    table = Table(
        tuple(f"player-{i}" for i in range(args.players)),
        (200,) * args.players,
        small_blind=1,
        big_blind=2,
        chip_unit="1",
    )
    config = TrainConfig(
        seed=args.seed,
        capacity=128,
        max_nodes=10_000,
        max_seconds=args.max_seconds,
        fit=FitConfig(width=16, steps=8, batch_size=8, diagnostic_samples=16),
    )
    started = perf_counter()
    runs = []
    for _ in range(2):
        trainer = HoldemTrainer(table, config)
        reports = [asdict(trainer.step()) for _ in range(2)]
        runs.append(
            {
                "reports": reports,
                "profile_sha256": trainer.current_profile().fingerprint,
                "replay_sha256": [memory.fingerprint() for memory in trainer.memories],
            }
        )
    if runs[0] != runs[1]:
        raise RuntimeError(
            "Repeated training differs in models, replay, RNG state or reports"
        )
    print(
        json.dumps(
            {
                "check": "holdem-training-smoke-v1",
                "revision": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain")),
                "source_sha256": source_fingerprint(),
                "environment": environment(),
                "table": asdict(table),
                "config": asdict(config),
                "reproduced": True,
                "elapsed_seconds": perf_counter() - started,
                **runs[0],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
