"""Experimental human versus two verified trained TP20 bots, with replay."""

import argparse
import json
from pathlib import Path
from random import Random
from time import time

from scripts.play_hu20 import _label
from src.arena.catalog import Checkpoint
from src.arena.runner import public_events
from src.arena.schedule import digest
from src.blueprint.abstraction import TP20_SCHEMA, choices
from src.blueprint.artifact import FrozenBlueprint, TP20_FORMAT
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


def table(button):
    return Table(("human","trained-1","trained-2"),(2000,)*3,button=button)


def replay_history(path):
    count = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["schema"] != "human-tp20-history-v1":
            raise ValueError("Wrong human-play schema")
        hand = Hand.start(table(row["button"]),hand_id=row["hand_id"],seed=row["deal_seed"])
        for record in row["actions"]:
            if hand.actor != record["seat"]:
                raise ValueError("Replay actor differs")
            hand = hand.apply(Action(ActionKind(record["kind"]),record["raise_to"]))
        if not hand.finished or digest(public_events(hand.events)) != row["public_events_sha256"]:
            raise ValueError("Replay public events differ")
        net = [p.stack-2000 for p in hand.observe(0).players]
        if net != row["net_chips"] or sum(net) != 0:
            raise ValueError("Replay native chip ledger differs")
        count += 1
    return count


def play(policies, hashes, *, session_seed, history, max_hands=None, input_fn=input, output=print):
    if len(policies) != 2 or len(hashes) != 2 or history.exists():
        raise ValueError("Two pinned trained policies and a new history path are required")
    sources = []
    for i,(path,sha) in enumerate(zip(policies,hashes)):
        source = FrozenBlueprint(Checkpoint(f"bot-{i}",str(path),sha,TP20_FORMAT),path)
        if (source.abstraction != TP20_SCHEMA or source.description["strategy"] != "current"
                or source.description["iteration"] < 1 or not source.entries):
            raise ValueError("Bots must be trained current TP20 profiles")
        sources.append(source)
    random = Random(session_seed)
    bot_random = [Random(session_seed ^ 0xB071),Random(session_seed ^ 0xB072)]
    telemetry = [{"trained":0,"fallback":0},{"trained":0,"fallback":0}]
    cumulative = [0,0,0]
    count = 0
    history.parent.mkdir(parents=True,exist_ok=True)
    output("Experimental three-player 20BB learner: restricted training menu, current C extraction.")
    output(f"Pinned bot models: {hashes}")
    with history.open("w") as saved:
        while max_hands is None or count < max_hands:
            button, seed = count%3, random.randrange(2**63)
            hand_id = f"human-tp20-{session_seed}-{count}"
            hand = Hand.start(table(button),hand_id=hand_id,seed=seed)
            actions = []
            output(f"\nHand {count+1}: button is seat {button}; you are seat 0.")
            while not hand.finished:
                view = hand.observe(hand.actor)
                if hand.actor == 0:
                    output(f"{view.street.value}: board {' '.join(view.board) or '(none)'}")
                    output(f"Your cards: {' '.join(view.hole_cards)} | pot {view.pot/100:g} BB")
                    output("Stacks: "+", ".join(f"seat {i}: {p.stack/100:g} BB"
                                               for i,p in enumerate(view.players)))
                    menu = choices(view,free_fold=False)
                    for i,item in enumerate(menu,1):
                        output(f"  {i}. {_label(item.action)}")
                    while True:
                        answer = input_fn("Choose action number: ").strip()
                        if answer.isdigit() and 1 <= int(answer) <= len(menu):
                            action = menu[int(answer)-1].action
                            break
                        output("Choose one of the listed action numbers.")
                else:
                    i = hand.actor-1
                    menu, probabilities, trained = sources[i].distribution(view)
                    telemetry[i]["trained" if trained else "fallback"] += 1
                    action = bot_random[i].choices(menu,weights=probabilities,k=1)[0].action
                    output(f"Bot seat {hand.actor}: {_label(action)}")
                view.legal_actions.validate(action)
                actions.append({"seat":hand.actor,"kind":action.kind.value,"raise_to":action.raise_to})
                hand = hand.apply(action)
            final = hand.observe(0)
            net = [p.stack-2000 for p in final.players]
            if sum(net) != 0:
                raise ValueError("Native settlement did not conserve chips")
            cumulative = [a+b for a,b in zip(cumulative,net)]
            output(f"Board: {' '.join(final.board) or '(none)'}")
            for i,p in enumerate(final.players):
                if p.shown_cards:
                    output(f"Seat {i} showed: {' '.join(p.shown_cards)}")
            output(f"Your hand {net[0]/100:+g} BB; session {cumulative[0]/100:+g} BB")
            saved.write(json.dumps({"schema":"human-tp20-history-v1","hand_id":hand_id,
                "deal_seed":seed,"button":button,"actions":actions,"net_chips":net,
                "model_sha256":hashes,"public_events_sha256":digest(public_events(hand.events))},
                sort_keys=True)+"\n")
            saved.flush()
            count += 1
            if max_hands is None and input_fn("Another hand? [Y/n] ").strip().lower() in ("n","no"):
                break
    output(f"Saved {count} replayable hands to {history}; lookup counts: {telemetry}")
    return {"hands":count,"net_chips":cumulative,"lookup_counts":telemetry}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policies",nargs=2,type=Path)
    p.add_argument("--hashes",nargs=2)
    p.add_argument("--history",type=Path)
    p.add_argument("--seed",type=int,default=int(time()))
    p.add_argument("--hands",type=int)
    p.add_argument("--replay",type=Path)
    a = p.parse_args()
    if a.replay:
        print(f"Verified {replay_history(a.replay)} completed hands")
        return 0
    if a.policies is None or a.hashes is None:
        p.error("Supply two inference --policies and their pinned --hashes")
    play(a.policies,a.hashes,session_seed=a.seed,
         history=a.history or Path("results")/f"human-tp20-{a.seed}.jsonl",max_hands=a.hands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
