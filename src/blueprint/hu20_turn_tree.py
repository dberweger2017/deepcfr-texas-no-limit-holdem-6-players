"""Compile exact HU20 betting from public events and a synthetic deal."""

from dataclasses import asdict
from time import monotonic

from src.arena.endgame_quality import _world
from src.blueprint.abstraction import Choice, choices
from src.blueprint.hu20_river import public_identity
from src.game.observation import ActionTaken, BoardDealt, HandFinished, replay
from src.game.types import ActionKind, Street

SOLVER_COMMIT = "9d1509fe5077d019825f833eed04b16d342dfda1"
VERSION = "hu20-public-turn-search-v1"


def round_root(history, street=None):
    street = replay(history, 0, ()).street if street is None else street
    if street not in (Street.TURN, Street.RIVER):
        raise ValueError("Search requires a turn or river betting round")
    for i, event in enumerate(history):
        if isinstance(event, BoardDealt) and event.street == street:
            root = history[:i + 2]
            view = replay(root, 0, ())
            if view.finished or view.actor is None:
                raise ValueError("Round root has no live decision")
            return root
    raise ValueError("Missing public round root")


def solver_action(view, action):
    if action.kind != ActionKind.RAISE:
        return {"kind": action.kind.value.capitalize()}
    own = view.players[view.seat]
    jam = action.raise_to - own.street_bet == own.stack
    kind = "AllIn" if jam else "Raise" if view.legal_actions.call_amount else "Bet"
    return {"kind": kind, "amount": action.raise_to}


def line_key(line):
    return tuple((a["kind"], a.get("amount")) for a in line)


def betting_line(root, history):
    result = []
    for i in range(len(root), len(history)):
        event = history[i]
        if isinstance(event, ActionTaken):
            result.append(solver_action(replay(history[:i], event.seat, ()), event.action))
    return result


def compile_tree(root, observed_history, *, menu="native", max_nodes=300_000,
                 deadline=None, fixed_menus=None):
    public = replay(root, 0, ())
    start = root[0]
    if (public.street not in (Street.TURN, Street.RIVER) or len(public.players) != 2
            or start.stacks != (2000, 2000) or start.small_blind != 50
            or start.big_blind != 100 or start.chip_unit != "0.01"
            or public.players[0].stack != public.players[1].stack):
        raise ValueError("Search needs the equal-stack reset HU20 round root")
    if menu not in ("native", "cap2"):
        raise ValueError("Unknown search menu")
    inserted = {}
    for i in range(len(root), len(observed_history)):
        event = observed_history[i]
        if isinstance(event, ActionTaken):
            prefix = line_key(betting_line(root, observed_history[:i]))
            inserted[prefix] = event.action
    fixed_menus = fixed_menus or {}
    seat_map = ((public.button + 1) % 2, public.button)
    nodes = []

    def visit(hand, line):
        if deadline is not None and monotonic() >= deadline:
            raise TimeoutError("Public-tree compilation deadline")
        if len(nodes) >= max_nodes:
            raise MemoryError("Public-tree node limit")
        node = {"line": line, "terminal": hand.finished}
        nodes.append(node)
        if hand.finished:
            node["showdown"] = next(e.showdown for e in hand.events if isinstance(e, HandFinished))
            return
        view = hand.observe(hand.actor)
        options = list(choices(view, raise_cap=None, free_fold=False))
        raises = sum(isinstance(e, ActionTaken) and e.street == view.street
                     and e.action.kind == ActionKind.RAISE for e in view.history)
        if menu == "cap2" and raises >= 2:
            own = view.players[view.seat]
            options = [c for c in options if c.action.kind != ActionKind.RAISE
                       or c.action.raise_to - own.street_bet == own.stack]
        prefix = line_key(line)
        additions = list(fixed_menus.get(prefix, ()))
        if prefix in inserted:
            additions.append(inserted[prefix])
        for action in additions:
            view.legal_actions.validate(action)
            if action not in [c.action for c in options]:
                options.append(Choice("inserted", action))
        node.update(street=view.street.value, player=seat_map.index(view.seat),
                    actions=[solver_action(view, c.action) for c in options],
                    names=[c.name for c in options],
                    native_actions=[asdict(c.action) for c in options])
        for item, action in zip(options, node["actions"], strict=True):
            visit(hand.apply(item.action), line + [action])

    # This synthetic world supplies legality only; no live private deal enters.
    visit(_world(root, public.board, {}), [])
    return {"format": VERSION, "solver_commit": SOLVER_COMMIT,
            "spot": public_identity(root), "board": list(public.board),
            "initial_street": public.street.value, "pot": public.pot,
            "effective_stack": public.players[0].stack, "seat_map": list(seat_map),
            "search_menu": menu, "nodes": nodes}
