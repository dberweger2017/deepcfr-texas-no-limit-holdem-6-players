"""Native HU20 exports for a separately installed postflop solver.

No solver source or production-policy changes belong in this module. The
external tree's street raise-to amounts and physical seat mapping are explicit.
"""

from dataclasses import asdict, replace
from hashlib import blake2b
from itertools import combinations, permutations
import json
from pathlib import Path
from random import Random
from time import monotonic

import numpy as np

from src.arena.endgame_quality import _world
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, _history, choices, information_key
from src.blueprint.hu20_river import public_identity
from src.blueprint.search import DECK
from src.diagnostics.exact_ranker import exact_seven_card
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken, BoardDealt, HandFinished, replay
from src.game.types import ActionKind, Street

VERSION = "hu20-exact-flop-check-v1"
SOLVER_COMMIT = "9d1509fe5077d019825f833eed04b16d342dfda1"
STREETS = (Street.FLOP, Street.TURN, Street.RIVER)


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def line_key(line):
    """Action-object key order differs between Python and the Rust JSON writer."""
    return json.dumps(line, sort_keys=True, separators=(",", ":"))


def fixture_root(kind, *, seed=202610010901, street=Street.FLOP, button=0):
    """Outcome-blind public roots; no private holding is part of their identity."""
    lines = {"limped": ("call", "check"), "min-raised": ("min", "call"),
             "pot-raised": ("pot", "call"), "3-bet": ("min", "min", "call"),
             "short-spr": ("pot", "pot", "min", "call"),
             "tiny-spr": ("min",) * 18 + ("call",)}
    hand = Hand.start(Table(("a", "b"), (2000, 2000), button=button),
                      hand_id="exact-flop-fixture", seed=seed)
    for label in lines[kind]:
        menu = choices(hand.observe(hand.actor), raise_cap=None, free_fold=False)
        hand = hand.apply(next(c.action for c in menu if c.name == label))
    while hand.observe(hand.actor).street != street:
        menu = choices(hand.observe(hand.actor), raise_cap=None, free_fold=False)
        hand = hand.apply(next(c.action for c in menu if c.name == "check"))
    return hand.events


def descriptor(cards, board):
    """The exact v1 descriptor, with #126's ranker on seven-card inputs."""
    value = exact_seven_card(tuple(cards) + tuple(board))
    top = value[1]
    suits = [sum(c[1] == s for c in (*cards, *board)) for s in "cdhs"]
    ranks = {"23456789TJQKA".index(c[0]) for c in (*cards, *board)}
    windows = ({12, 0, 1, 2, 3},) + tuple(set(range(i, i + 5)) for i in range(9))
    return (value[0], 0 if top < 8 else 1 if top < 12 else 2,
            int(len(board) < 5 and max(suits) >= 4),
            int(len(board) < 5 and any(len(w - ranks) == 1 for w in windows)),
            int(len({c[0] for c in board}) != len(board)))


def descriptor_code(value):
    rank, top, flush, straight, paired = value
    return ((((rank * 3 + top) * 2 + flush) * 2 + straight) * 2 + paired)


def decode_descriptor(code):
    paired = code % 2; code //= 2
    straight = code % 2; code //= 2
    flush = code % 2; code //= 2
    return (code // 3, code % 3, flush, straight, paired)


def key_template(view, menu):
    if view.street not in STREETS or len(view.players) != 2:
        raise ValueError("Factorization is only defined for HU20 postflop v1")
    return [HU20_UNCAPPED_SCHEMA, 2, (view.seat - view.button) % 2,
            view.street.value, None,
            [(view.players[(view.button + i) % 2].folded,
              view.players[(view.button + i) % 2].all_in) for i in (0, 1)],
            _history(view), [c.name for c in menu]]


def factored_key(template, value):
    payload = list(template)
    payload[4] = value
    return blake2b(json.dumps(payload, separators=(",", ":")).encode(),
                   digest_size=16).hexdigest()


def solver_action(view, choice):
    action = choice.action
    if action.kind != ActionKind.RAISE:
        return {"kind": action.kind.value.capitalize()}
    own = view.players[view.seat]
    jam = action.raise_to - own.street_bet == own.stack
    kind = "AllIn" if jam else "Raise" if view.legal_actions.call_amount else "Bet"
    return {"kind": kind, "amount": action.raise_to}


def _menu(view, cap):
    menu = choices(view, raise_cap=None, free_fold=False)
    if cap is None:
        return menu
    if type(cap) is not int or cap < 3:
        raise ValueError("Diagnostic fallback cap must be at least three")
    raises = sum(isinstance(e, ActionTaken) and e.street == view.street
                 and e.action.kind == ActionKind.RAISE for e in view.history)
    if raises < cap:
        return menu
    own = view.players[view.seat]
    return tuple(c for c in menu if c.action.kind != ActionKind.RAISE
                 or c.action.raise_to - own.street_bet == own.stack)


def compile_tree(root, *, raise_cap=None, max_nodes=300_000, seconds=600):
    """Compile one representative runout; betting legality is card independent.

    Chance deals are omitted only from the action-line identifiers. Histories
    retained privately supply an independent production replay for gate K/V2.
    A compilation guard never turns a partial tree into a valid solver request.
    """
    public = replay(root, 0, ())
    if (public.street not in STREETS or public.finished or public.actor is None
            or public.players[0].stack != public.players[1].stack
            or tuple(p.starting_stack for p in public.players) != (2000, 2000)
            or public.big_blind != 100 or public.small_blind != 50):
        raise ValueError("Need an equal-stack reset HU20 public street root")
    representative = _world(root, public.board, {})
    oop = (public.button + 1) % 2
    seat_map = (oop, public.button)
    nodes = []; histories = []; deadline = monotonic() + seconds

    def visit(hand, line):
        if len(nodes) >= max_nodes:
            raise MemoryError(f"Public-tree compilation exceeds {max_nodes} nodes")
        if monotonic() >= deadline:
            raise TimeoutError("Public-tree compilation deadline")
        index = len(nodes); histories.append(hand.events)
        node = {"line": line, "terminal": hand.finished}
        nodes.append(node)
        if hand.finished:
            finish = next(e for e in hand.events if isinstance(e, HandFinished))
            node["showdown"] = finish.showdown
            return
        view = hand.observe(hand.actor); menu = _menu(view, raise_cap)
        full_menu = choices(view, raise_cap=None, free_fold=False)
        node.update(street=view.street.value, player=seat_map.index(view.seat),
                    actions=[solver_action(view, c) for c in menu],
                    names=[c.name for c in menu], native_names=[c.name for c in full_menu],
                    template=key_template(view, full_menu),
                    native_actions=[asdict(c.action) for c in menu])
        for item, action in zip(menu, node["actions"], strict=True):
            visit(hand.apply(item.action), line + [action])

    visit(representative, [])
    request = {"format": VERSION, "solver_commit": SOLVER_COMMIT,
               "spot": public_identity(root), "board": list(public.board),
               "initial_street": public.street.value, "pot": public.pot,
               "effective_stack": public.players[0].stack,
               "seat_map": list(seat_map), "raise_cap": raise_cap, "nodes": nodes}
    return request, histories


def gate_k(compilations, *, samples=100_000, seed=202610010902):
    if samples < 100_000:
        raise ValueError("Gate K requires at least 100,000 samples")
    rng = Random(seed); counts = {}; mismatches = []
    pools = [[i for i, n in enumerate(req["nodes"]) if not n["terminal"]]
             for req, _ in compilations]
    for sample in range(samples):
        family = sample % len(compilations); request, histories = compilations[family]
        index = rng.choice(pools[family]); node = request["nodes"][index]
        length = {"flop": 3, "turn": 4, "river": 5}[node["street"]]
        available = [c for c in DECK if c not in request["board"][:3]]
        drawn = rng.sample(available, length - 3 + 2)
        board = tuple(request["board"][:3]) + tuple(drawn[:length - 3])
        holding = tuple(drawn[length - 3:]); history = []
        for event in histories[index]:
            if isinstance(event, BoardDealt):
                cards = board[:3] if event.street == Street.FLOP else (
                    (board[3],) if event.street == Street.TURN else (board[4],))
                event = replace(event, cards=cards)
            history.append(event)
        view = replay(tuple(history), request["seat_map"][node["player"]], holding)
        menu = choices(view, raise_cap=None, free_fold=False)
        full = information_key(view, menu, schema=HU20_UNCAPPED_SCHEMA)
        factored = factored_key(node["template"], descriptor(holding, board))
        label = f'{family}/{node["street"]}'
        counts[label] = counts.get(label, 0) + 1
        if full != factored:
            mismatches.append({"sample": sample, "node": index, "family": family})
            break
    return {"gate": "K", "passed": not mismatches and sum(counts.values()) == samples,
            "samples": sum(counts.values()), "seed": seed, "strata": counts,
            "mismatches": mismatches}


def export_descriptors(flop, path, check=lambda: None):
    """All 49 turns and 2,352 ordered rivers; blocked holdings have code 255."""
    remaining = tuple(c for c in DECK if c not in flop)
    holdings = tuple(combinations(remaining, 2))
    boards = [tuple(flop)] + [tuple(flop) + (c,) for c in remaining]
    boards += [tuple(flop) + pair for pair in permutations(remaining, 2)]
    codes = np.full((len(boards), len(holdings)), 255, dtype=np.uint8)
    for index, board in enumerate(boards):
        check(); blocked = set(board)
        for j, holding in enumerate(holdings):
            if not blocked.intersection(holding):
                codes[index, j] = descriptor_code(descriptor(holding, board))
    # Arrays are private tool inputs, never committed model artifacts.
    with Path(path).open("wb") as target:
        np.savez_compressed(target, codes=codes,
                            boards=np.asarray(["".join(b) for b in boards]),
                            holdings=np.asarray(["".join(h) for h in holdings]))
    return {"boards": len(boards), "turns": 49, "ordered_rivers": 2352,
            "holdings": len(holdings), "blocked_code": 255}


def export_policy_tables(request, blueprint, codes_by_street):
    """Deduplicate identical history templates, preserving actual menu labels."""
    if blueprint.abstraction != HU20_UNCAPPED_SCHEMA or blueprint.raise_cap is not None:
        raise ValueError("Expected a native-reopening v1 inference policy")
    tables = {}; node_tables = {}
    for node in request["nodes"]:
        if node["terminal"]:
            continue
        identity = blake2b(json.dumps(node["template"], separators=(",", ":")).encode(),
                           digest_size=16).hexdigest()
        node_tables[line_key(node["line"])] = identity
        if identity in tables:
            continue
        rows = {}
        for code in sorted(codes_by_street[node["street"]]):
            key = factored_key(node["template"], decode_descriptor(code))
            entry = blueprint.entries.get(key)
            if entry is not None and entry[0] != tuple(node["native_names"]):
                raise ValueError("Factored blueprint action labels differ")
            if node["names"] != node["native_names"]:
                raise ValueError("Capped policy needs removed-line reach audit before export")
            rows[str(code)] = list(entry[1]) if entry else [1 / len(node["names"])] * len(node["names"])
        tables[identity] = {"names": node["names"], "rows": rows}
    return {"tables": tables, "node_tables": node_tables,
            "source": blueprint.description, "fallback": "uniform in retained native menu"}
