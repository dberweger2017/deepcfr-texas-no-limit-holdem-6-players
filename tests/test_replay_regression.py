"""Public observations retained from the pre-optimization replay implementation."""

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from random import Random

from src.game.hand import Hand, Table
from src.holdem.actions import bet_candidates


def observation_digest(players, seed):
    table = Table(
        tuple(f"player-{i}" for i in range(players)),
        (3, 7, 13, 21, 35, 51)[:players],
        button=seed % players,
        small_blind=1,
        big_blind=2,
        chip_unit="1",
    )
    hand = Hand.start(table, hand_id="replay-regression", seed=seed)
    random = Random(seed)
    digest = sha256()
    while True:
        for seat in range(players):
            view = hand.observe(seat)
            digest.update(json.dumps(asdict(view), sort_keys=True).encode())
        if hand.finished:
            break
        candidates = bet_candidates(hand.observe(hand.actor))
        hand = hand.apply(random.choice(candidates.actions))
    return digest.hexdigest()


def test_public_replays_match_recorded_observations():
    path = Path(__file__).parent / "fixtures/replay-regression.json"
    for case in json.loads(path.read_text())["cases"]:
        assert observation_digest(case["players"], case["seed"]) == case["sha256"]
