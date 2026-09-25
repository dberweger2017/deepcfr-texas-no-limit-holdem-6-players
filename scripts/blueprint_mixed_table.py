"""Exploratory six-seat cash tables with fixed blueprint and public opponents."""

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
SHOWCASE_LINEUPS = {
    "two-each": PLAYER_IDS,
    "three-trained-three-random": (
        "trained-1", "trained-2", "trained-3",
        "random-1", "random-2", "random-3",
    ),
    "one-trained-five-random": (
        "trained-1", "random-1", "random-2",
        "random-3", "random-4", "random-5",
    ),
    "trained-tight-passive-four-random": (
        "trained-1", "tight-passive", "random-1",
        "random-2", "random-3", "random-4",
    ),
    "trained-loose-aggressive-four-random": (
        "trained-1", "loose-aggressive", "random-1",
        "random-2", "random-3", "random-4",
    ),
}


def checkpoint_hash(path: Path) -> str:
    fingerprint = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    return fingerprint.hexdigest()


def run(
    trainer, *, hands: int, seed: int, sample_every: int = 100,
    player_ids: tuple[str, ...] = PLAYER_IDS,
) -> dict:
    if (
        type(hands) is not int
        or hands < 1
        or type(sample_every) is not int
        or sample_every < 1
    ):
        raise ValueError("Hands and sample interval must be positive integers")
    if trainer.table.capacity != 6 or trainer.table.stacks != (10_000,) * 6:
        raise ValueError("The showcase needs a six-seat, 100 BB blueprint")
    if len(player_ids) != 6 or len(set(player_ids)) != 6:
        raise ValueError("The showcase needs six distinct player identities")
    groups = {
        "trained": tuple(name for name in player_ids if name.startswith("trained-")),
        "random": tuple(name for name in player_ids if name.startswith("random-")),
        "scripted": tuple(
            name for name in player_ids
            if name in ("tight-passive", "loose-aggressive")
        ),
    }
    if sum(map(len, groups.values())) != 6:
        raise ValueError("Unknown showcase player identity")
    table = Table(player_ids, (10_000,) * 6)
    seeds = Random(seed)
    counts = Counter()
    policies = {}
    for identity in player_ids:
        policy_seed = seeds.getrandbits(64)
        if identity.startswith("trained-"):
            policies[identity] = _TablePolicy(trainer, policy_seed, counts, uniform=False)
        elif identity.startswith("random-"):
            policies[identity] = RandomPolicy(policy_seed)
        else:
            policies[identity] = StylePolicy(
                STYLES[identity.replace("-", "_")], policy_seed
            )
    deals = Random(seeds.getrandbits(64))
    cumulative = {identity: 0 for identity in player_ids}
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
                    for identity in player_ids
                },
                "group_net_bb": {
                    group: sum(cumulative[identity] for identity in ids)
                    / table.big_blind
                    for group, ids in groups.items()
                },
            }
            if number % sample_every == 0 or number == hands:
                samples.append(snapshot)
            if number in (1000, 2000, 3000):
                milestones[str(number)] = snapshot
    return {
        "hands": hands,
        "deal_seed": seed,
        "players": list(player_ids),
        "groups": {key: list(ids) for key, ids in groups.items()},
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
    parser.add_argument(
        "--suite", action="store_true",
        help="Five independent two-each tables and four alternate six-seat lineups",
    )
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output already exists")
    if args.hands < 1:
        parser.error("--hands must be positive")
    digest = checkpoint_hash(args.checkpoint)
    if digest != args.expected_sha256:
        parser.error("Chosen checkpoint hash differs from the declared checkpoint")
    trainer = load_training(args.checkpoint)
    args.out.mkdir(parents=True)
    header = {
        "checkpoint_sha256": digest,
        "iteration": trainer.iteration,
        "entries": len(trainer.nodes),
    }
    if args.suite:
        alternate_lineups = [
            (name, player_ids)
            for name, player_ids in SHOWCASE_LINEUPS.items()
            if name != "two-each"
        ]
        schedule = [
            (f"two-each-{index + 1}", SHOWCASE_LINEUPS["two-each"], args.seed + index)
            for index in range(5)
        ] + [
            (name, player_ids, args.seed + index + 5)
            for index, (name, player_ids) in enumerate(alternate_lineups)
        ]
        for name, player_ids, seed in schedule:
            result = run(
                trainer, hands=args.hands, seed=seed,
                player_ids=player_ids,
            )
            write_json(args.out / f"{name}.json", {**header, **result})
            print(f"Completed {name}: {result['milestones'].get(str(args.hands), result['samples'][-1])['group_net_bb']}", flush=True)
        write_json(args.out / "suite.json", {
            **header,
            "hands_per_table": args.hands,
            "sample_every": 100,
            "tables": [
                {"name": name, "file": f"{name}.json", "deal_seed": seed}
                for name, _, seed in schedule
            ],
        })
    else:
        result = run(trainer, hands=args.hands, seed=args.seed)
        write_json(args.out / "result.json", {**header, **result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
