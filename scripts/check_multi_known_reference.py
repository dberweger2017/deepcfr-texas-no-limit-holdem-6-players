"""Independent OpenSpiel 2.0.2 values and exact per-player BR for a native profile.

Run in a separate reference environment; OpenSpiel is not a runtime dependency.
The adapter uses public information-state strings, never a hidden deal.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re


def payload(game: str, players: int, info: str) -> bytes:
    history: list[int] = []
    if game == "kuhn":
        match = re.fullmatch(r"(\d+)([pb]*)", info)
        assert match, info
        card, actions = match.groups()
        bet = False
        for i, action in enumerate(actions):
            native = (0 if bet else 1) if action == "p" else (1 if bet else 2)
            history.append((i % players) * 4 + native)
            bet |= action == "b"
        return bytes([ord("K"), players, len(actions) % players, int(card), 255, 0, *history])
    def field(name: str) -> str:
        match = re.search(r"\[" + name + r": ([^\]]*)\]", info)
        assert match, (name, info)
        return match[1]
    owner, card = int(field("Observer")), int(field("Private"))
    match = re.search(r"\[Public: (\d+)\]", info)
    board = int(match[1]) if match else 255
    folded: set[int] = set()
    actor = 0
    for street in range(1 + int(board != 255)):
        actor = next(p for p in range(players) if p not in folded)
        if street:
            history.append(255)
        for action in map(int, field(f"Round{street + 1}").split()):
            history.append(actor * 4 + action)
            if action == 0:
                folded.add(actor)
            actor = next((actor + i) % players for i in range(1, players + 1) if (actor + i) % players not in folded)
    assert actor == owner, info
    return bytes([ord("L"), players, owner, card, board, int(board != 255), *history])


def check(path: Path) -> dict:
    import pyspiel

    version = importlib.metadata.version("open_spiel")
    assert version == "2.0.2", version
    data = json.loads(path.read_text())
    name, players = data["identity"]["game"], data["identity"]["players"]
    params = {"players": players}
    if name == "leduc":
        params["suit_isomorphism"] = True
    game = pyspiel.load_game(name + "_poker", params)
    reference = pyspiel.UniformRandomPolicy(game).policy_table()
    used = set()
    for info, actions in reference.items():
        key = hashlib.blake2b(payload(name, players, info), digest_size=16).hexdigest()
        probabilities = data["policies"][key]
        assert len(actions) == len(probabilities)
        reference[info] = [(action, probability) for (action, _), probability in zip(actions, probabilities)]
        used.add(key)
    assert used == data["policies"].keys(), (len(used), len(data["policies"]))
    policy = pyspiel.TabularPolicy(reference)
    values = pyspiel.expected_returns(game.new_initial_state(), policy, -1, True)
    br = [pyspiel.TabularBestResponse(game, p, policy).value("") for p in range(players)]
    for label, observed in (("values", values), ("best_response", br)):
        assert max(abs(a - b) for a, b in zip(observed, data[label])) < 1e-10, (label, observed, data[label])
    return {"status": "passed", "open_spiel_version": version, "input": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "information_sets": len(used),
            "values": values, "best_response": br, "gains": [b-v for b, v in zip(br, values)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.profiles:
        print(json.dumps(check(path)), flush=True)
