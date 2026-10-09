"""Record HU20 decisions from the Python engine as parity fixtures for the native trainer.

Each JSON line is one hand: the 52-card deck, the chosen actions, and at every
decision the legal bounds, pot, abstract menu and v1 information key (or, with
`--card-buckets`, the equity-bucket key), then the final stacks. Menu actions are chosen uniformly; a fraction of raises use a
random legal size so off-menu history labels are exercised too.
"""

import argparse
import json
from pathlib import Path
from random import Random

from src.blueprint import equity_buckets
from src.blueprint.abstraction import (HU20_EQUITY_SCHEMA, HU20_UNCAPPED_SCHEMA, HU100_EQUITY_SCHEMA, HU100_SCHEMA, HU200_SCHEMA, HU200_EQUITY_SCHEMA,
                                       choices, information_key)
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind

DECK = tuple(rank + suit for rank in "23456789TJQKA" for suit in "cdhs")


def record(seed, off_menu, passive, stack_bb=20, equity=False):
    if stack_bb not in (20, 100, 200):
        raise ValueError("stack must be 20, 100 or 200 BB")
    rng = Random(seed)
    deck = list(DECK)
    rng.shuffle(deck)
    button = rng.randrange(2)
    hand = Hand.from_deck(Table(("a", "b"), (stack_bb * 100,) * 2, button=button), hand_id="parity", deck=tuple(deck))
    decisions, actions = [], []
    while not hand.finished:
        view = hand.observe(hand.actor)
        legal = view.legal_actions
        menu = choices(view, raise_cap=None, free_fold=False)
        decisions.append({
            "actor": hand.actor, "street": view.street.value, "pot": view.pot,
            "kinds": [k.value for k in legal.kinds], "call": legal.call_amount,
            "min_raise_to": legal.min_raise_to, "max_raise_to": legal.max_raise_to,
            "menu": [[c.name, c.action.raise_to] for c in menu],
            "key": information_key(view, menu, schema=(HU200_EQUITY_SCHEMA if equity else HU200_SCHEMA) if stack_bb == 200 else
                                   (HU100_EQUITY_SCHEMA if equity else HU100_SCHEMA) if stack_bb == 100
                                   else HU20_EQUITY_SCHEMA if equity else HU20_UNCAPPED_SCHEMA),
        })
        quiet = [c for c in menu if c.name in ("check", "call")]
        action = (rng.choice(quiet) if quiet and rng.random() < passive else rng.choice(menu)).action
        if (ActionKind.RAISE in legal.kinds and rng.random() < off_menu
                and legal.max_raise_to > legal.min_raise_to):
            action = Action(ActionKind.RAISE, rng.randint(legal.min_raise_to, legal.max_raise_to))
        actions.append([action.kind.value, action.raise_to])
        hand = hand.apply(action)
    final = hand.events[-1].stacks
    return {"stack_bb": stack_bb, "seed": seed, "button": button, "deck": deck, "actions": actions,
            "decisions": decisions, "final_stacks": list(final)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stack-bb", type=int, choices=(20, 100, 200), default=20)
    p.add_argument("--hands", type=int, default=20000)
    p.add_argument("--seed", type=int, default=202610050100)
    p.add_argument("--off-menu", type=float, default=0.15)
    p.add_argument("--passive", type=float, default=0.0, help="probability of checking or calling, to reach later streets")
    p.add_argument("--card-buckets", type=Path, help="key cards by the K=50 equity tables in this directory")
    p.add_argument("--card-tables-unpinned", action="store_true", help="admit tables other than #163's, for test fixtures")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.card_buckets:
        equity_buckets.register(equity_buckets.EquityCards(a.card_buckets, pinned=not a.card_tables_unpinned))
    with a.out.open("w") as stream:
        for index in range(a.hands):
            hand = record(a.seed + index, a.off_menu, a.passive, a.stack_bb, equity=a.card_buckets is not None)
            stream.write(json.dumps(hand, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
