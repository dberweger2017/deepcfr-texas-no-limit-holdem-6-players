"""Exploratory six-seat cash table: two blueprint, two random, two scripted."""

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from random import Random

from src.arena.artifacts import write_json
from src.arena.heuristics import STYLES, StylePolicy
from src.blueprint.artifact import load_training
from src.blueprint.evaluation import _TablePolicy
from src.game.hand import Hand, Table
from src.game.play import RandomPolicy, play_hand

PLAYER_IDS = (
    "trained-1",
    "trained-2",
    "random-1",
    "random-2",
    "tight-passive",
    "loose-aggressive",
)
GROUPS = {
    "trained": PLAYER_IDS[:2],
    "random": PLAYER_IDS[2:4],
    "scripted": PLAYER_IDS[4:],
}


def checkpoint_hash(path: Path) -> str:
    fingerprint = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    return fingerprint.hexdigest()


def run(trainer, *, hands: int, seed: int, sample_every: int = 100) -> dict:
    if (
        type(hands) is not int
        or hands < 1
        or type(sample_every) is not int
        or sample_every < 1
    ):
        raise ValueError("Hands and sample interval must be positive integers")
    if trainer.table.capacity != 6 or trainer.table.stacks != (10_000,) * 6:
        raise ValueError("The showcase needs a six-seat, 100 BB blueprint")
    table = Table(PLAYER_IDS, (10_000,) * 6)
    seeds = Random(seed)
    counts = Counter()
    policies = {
        "trained-1": _TablePolicy(
            trainer, seeds.getrandbits(64), counts, uniform=False
        ),
        "trained-2": _TablePolicy(
            trainer, seeds.getrandbits(64), counts, uniform=False
        ),
        "random-1": RandomPolicy(seeds.getrandbits(64)),
        "random-2": RandomPolicy(seeds.getrandbits(64)),
        "tight-passive": StylePolicy(STYLES["tight_passive"], seeds.getrandbits(64)),
        "loose-aggressive": StylePolicy(
            STYLES["loose_aggressive"], seeds.getrandbits(64)
        ),
    }
    deals = Random(seeds.getrandbits(64))
    cumulative = {identity: 0 for identity in PLAYER_IDS}
    samples = []
    milestones = {}
    for number in range(1, hands + 1):
        hand = Hand.start(
            replace(table, button=(number - 1) % 6),
            hand_id=f"mixed-table-{number}",
            seed=deals.getrandbits(64),
        )
        finished = play_hand(hand, policies)
        final = finished.observe(0)
        for player in final.players:
            cumulative[player.player_id] += player.stack - player.starting_stack
        if sum(cumulative.values()) != 0:
            raise RuntimeError(f"Chip accounting failed after hand {number}")
        if (
            number % sample_every == 0
            or number == hands
            or number in (1000, 2000, 3000)
        ):
            snapshot = {
                "hand": number,
                "net_bb": {
                    identity: cumulative[identity] / table.big_blind
                    for identity in PLAYER_IDS
                },
                "group_net_bb": {
                    group: sum(cumulative[identity] for identity in ids)
                    / table.big_blind
                    for group, ids in GROUPS.items()
                },
            }
            if number % sample_every == 0 or number == hands:
                samples.append(snapshot)
            if number in (1000, 2000, 3000):
                milestones[str(number)] = snapshot
    return {
        "hands": hands,
        "deal_seed": seed,
        "players": list(PLAYER_IDS),
        "groups": {key: list(ids) for key, ids in GROUPS.items()},
        "button_rotation": "one clockwise seat per hand",
        "stacks": "reset to 100 BB for every hand; balances are cumulative net winnings",
        "samples": samples,
        "milestones": milestones,
        "trained_decisions": {
            street: {
                "trained": counts[(street, "trained")],
                "fallback": counts[(street, "fallback")],
            }
            for street in ("preflop", "flop", "turn", "river")
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026092411)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output already exists")
    if args.hands < 1:
        parser.error("--hands must be positive")
    digest = checkpoint_hash(args.checkpoint)
    if digest != args.expected_sha256:
        parser.error("Chosen checkpoint hash differs from the declared checkpoint")
    trainer = load_training(args.checkpoint)
    result = run(trainer, hands=args.hands, seed=args.seed)
    args.out.mkdir(parents=True)
    write_json(
        args.out / "result.json",
        {
            "checkpoint_sha256": digest,
            "iteration": trainer.iteration,
            "entries": len(trainer.nodes),
            **result,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
