"""Play complete local heads-up 20BB hands against a verified trained policy."""

import argparse
import json
from pathlib import Path
from random import Random
from time import time

from src.arena.runner import public_events
from src.arena.schedule import digest
from src.blueprint.abstraction import HU20_SCHEMA, choices
from src.blueprint.solver import HU20_GAME
from src.blueprint.windowed import WindowedDistribution
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


def _table(button):
    return Table(("human", "trained"), (2000, 2000), button=button)


def _label(action):
    return (f"raise to {action.raise_to / 100:g} BB"
            if action.kind == ActionKind.RAISE else action.kind.value)


def _display(view, output):
    position = "button / small blind" if view.button == 0 else "big blind"
    output(f"\n{view.street.value}: board {' '.join(view.board) or '(none)'}")
    output(f"Your cards: {' '.join(view.hole_cards)} | position: {position}")
    output(f"Pot: {view.pot / 100:g} BB | your stack: {view.players[0].stack / 100:g} BB "
           f"| bot stack: {view.players[1].stack / 100:g} BB")


def replay_history(path: Path) -> int:
    completed = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        hand = Hand.start(_table(row["button"]), hand_id=row["hand_id"], seed=row["deal_seed"])
        for record in row["actions"]:
            if hand.actor != record["seat"]:
                raise ValueError("History actor differs from native replay")
            hand = hand.apply(Action(ActionKind(record["kind"]), record["raise_to"]))
        if not hand.finished or digest(public_events(hand.events)) != row["public_events_sha256"]:
            raise ValueError("History does not replay to the saved public events")
        final = hand.observe(0)
        if final.players[0].stack-2000 != row["human_chips"]:
            raise ValueError("History chip result differs from native replay")
        completed += 1
    return completed


def play(index: Path, manifest_path: Path, *, arm: str, session_seed: int,
         history: Path, max_hands: int | None = None, input_fn=input, output=print):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("game") != HU20_GAME or manifest.get("abstraction") != HU20_SCHEMA:
        raise ValueError("Only a versioned trained HU20 artifact can play this game")
    source = WindowedDistribution(index, manifest, arm)
    random = Random(session_seed)
    bot_random = Random(session_seed ^ 0xB071)
    cumulative = 0
    hands = 0
    history.parent.mkdir(parents=True, exist_ok=True)
    if history.exists():
        raise FileExistsError("Choose a new history path to preserve prior sessions")
    output("Heads-up 20BB trained blueprint. Actions are restricted to its saved menu.")
    output(f"Model: {manifest['artifact_sha256']} | extraction: {arm} | game: {HU20_GAME}")
    try:
        with history.open("w", encoding="utf-8") as saved:
            while max_hands is None or hands < max_hands:
                button = hands % 2
                deal_seed = random.randrange(2**63)
                hand_id = f"human-hu20-{session_seed}-{hands}"
                hand = Hand.start(_table(button), hand_id=hand_id, seed=deal_seed)
                actions = []
                output(f"\nHand {hands+1}: you are {'button / small blind' if button == 0 else 'big blind'}")
                while not hand.finished:
                    view = hand.observe(hand.actor)
                    if hand.actor == 0:
                        _display(view, output)
                        menu = choices(view, free_fold=False)
                        for index_, item in enumerate(menu, 1):
                            output(f"  {index_}. {_label(item.action)}")
                        while True:
                            answer = input_fn("Choose action number: ").strip()
                            if answer.isdigit() and 1 <= int(answer) <= len(menu):
                                action = menu[int(answer)-1].action
                                break
                            output("Enter one of the listed action numbers.")
                    else:
                        menu, probabilities, _ = source.distribution(view)
                        action = bot_random.choices(menu, weights=probabilities, k=1)[0].action
                        output(f"Bot: {_label(action)}")
                    view.legal_actions.validate(action)
                    actions.append({"seat": hand.actor, "kind": action.kind.value,
                                    "raise_to": action.raise_to})
                    hand = hand.apply(action)
                final = hand.observe(0)
                human_chips = final.players[0].stack-2000
                if final.players[1].stack-2000 != -human_chips:
                    raise ValueError("Finished hand did not conserve chips")
                cumulative += human_chips
                output(f"Board: {' '.join(final.board) or '(none)'}")
                for seat, player in enumerate(final.players):
                    if player.shown_cards:
                        output(f"{('You', 'Bot')[seat]} showed: {' '.join(player.shown_cards)}")
                output(f"Hand: {human_chips / 100:+g} BB | session: {cumulative / 100:+g} BB")
                row = {"schema": "human-hu20-history-v1", "hand_id": hand_id,
                       "button": button, "deal_seed": deal_seed, "actions": actions,
                       "human_chips": human_chips,
                       "public_events_sha256": digest(public_events(hand.events)),
                       "model_sha256": manifest["artifact_sha256"]}
                saved.write(json.dumps(row, sort_keys=True) + "\n")
                saved.flush()
                hands += 1
                if max_hands is None and input_fn("Another hand? [Y/n] ").strip().lower() in ("n", "no"):
                    break
    finally:
        source.close()
    output(f"Saved {hands} replayable hands to {history}")
    return {"hands": hands, "human_chips": cumulative,
            "fallback_coverage": dict(source.coverage)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--arm", choices=("C", "A"), default="A")
    parser.add_argument("--seed", type=int, default=int(time()))
    parser.add_argument("--history", type=Path)
    parser.add_argument("--hands", type=int)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.replay:
        print(f"Verified {replay_history(args.replay)} completed hands")
        return 0
    if args.index is None or args.manifest is None:
        parser.error("--index and --manifest are required for live play")
    history = args.history or Path("results") / f"hu20-human-{args.seed}.jsonl"
    play(args.index, args.manifest, arm=args.arm, session_seed=args.seed,
         history=history, max_hands=args.hands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
