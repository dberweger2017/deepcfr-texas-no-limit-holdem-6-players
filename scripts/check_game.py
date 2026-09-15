"""Run a bounded headless check of the observation-based game interface."""

import argparse
import json
from dataclasses import replace
from random import Random

from src.game.hand import Hand, Table
from src.game.play import PlayerHistory, RandomPolicy, play_hand


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, choices=(4, 5, 6), default=6)
    parser.add_argument("--hands", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.hands <= 0:
        parser.error("--hands must be positive")
    seeds = Random(args.seed)
    identities = tuple(f"player-{i}" for i in range(args.players))
    table = Table(identities, (10_000,) * args.players)
    policies = {
        identity: RandomPolicy(seeds.getrandbits(64)) for identity in identities
    }
    deals = Random(seeds.getrandbits(64))
    histories = {identity: PlayerHistory(identity) for identity in identities}
    profits = [0] * args.players
    for index in range(args.hands):
        hand = Hand.start(
            replace(table, button=index % args.players),
            hand_id=f"hand-{index}",
            seed=deals.getrandbits(64),
        )
        hand = play_hand(hand, policies, histories)
        for seat, identity in enumerate(identities):
            view = hand.observe(seat)
            histories[identity] = histories[identity].append(view)
            profits[seat] += view.players[seat].stack - table.stacks[seat]
    if sum(profits) != 0:
        raise RuntimeError("Completed hands did not conserve chips")
    print(
        json.dumps(
            {
                "completed_hands": args.hands,
                "players": args.players,
                "net_chips": profits,
                "chip_unit": table.chip_unit,
            }
        )
    )


if __name__ == "__main__":
    main()
