"""Exercise a bankroll session with departures, arrivals, sit-outs, and reloads."""

import argparse
import json
from random import Random

from src.game.play import RandomPolicy, play_session_hand
from src.game.session import Session, replay_session


def run(seed: int = 0, hands: int = 30) -> dict:
    if type(hands) is not int or hands < 12:
        raise ValueError("Run at least 12 hands to cover the lineup changes")
    deals = Random(seed)
    action_seeds = Random(seed + 1)
    session = Session("session-smoke")
    policies = {}
    for seat in range(6):
        identity = f"player-{seat}"
        session.join(identity, seat, 10000)
        policies[identity] = RandomPolicy(action_seeds.getrandbits(64))
    counts = []
    for index in range(hands):
        if index in (3, 5):
            session.leave(f"player-{5 if index == 3 else 4}")
        if index == 7:
            session.sit_out("player-0")
        if index == 8:
            # Replenish a short stack before returning to post a full blind.
            player = next(p for p in session.seats if p.player_id == "player-0")
            if player.stack < session.big_blind:
                session.top_up("player-0", 10000 - player.stack)
            session.return_to_play("player-0")
        if index == 9:
            session.move("player-1", 5)
            session.join("new-player", 1, 10000)
            policies["new-player"] = RandomPolicy(action_seeds.getrandbits(64))
        for player in session.seats:
            if player.status == "busted":
                session.top_up(player.player_id, 10000)
        session.start_hand(
            seed=deals.getrandbits(64), opening_button=0 if index == 0 else None
        )
        counts.append(len(session.participants))
        play_session_hand(session, policies)
    ledger = replay_session(session.events)
    return {
        "completed_hands": hands,
        "participant_counts": counts,
        "chips_in": ledger.chips_in,
        "chips_out": ledger.chips_out,
        "table_chips": sum(p.stack for p in ledger.seats),
        "chip_unit": ledger.chip_unit,
        "session_profile": ledger.profile,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hands", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.hands)))


if __name__ == "__main__":
    main()
