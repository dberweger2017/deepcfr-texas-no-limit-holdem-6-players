"""Native HU200 river EV/BR versus the independent Python range-game evaluator."""

import argparse
import json
from math import fsum
from pathlib import Path
from time import monotonic

import numpy as np

from src.blueprint.hu20_turn_solver import ExternalTurnSolver, file_hash, parse_profiles
from src.blueprint.hu20_turn_tree import betting_line, compile_tree, line_key
from src.blueprint.hu200_river_search import HU200RiverSearchPolicy, HU200RiverSearchPlayer
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA, choices
from src.arena.endgame_quality import _world
from src.blueprint.river_cfr import profile_quality
from src.blueprint.river_game import RiverGame
from src.blueprint.search import DECK
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street


class ReferenceRiver(RiverGame):
    table_players = 2
    free_fold = False


def fixture():
    hand = Hand.start(Table(("reference-a", "reference-b"), (20000, 20000), button=0),
                      hand_id="hu200-river-reference", seed=31)
    hand = hand.apply(Action(ActionKind.RAISE, 4000))
    while hand.observe(hand.actor).street != Street.RIVER:
        view = hand.observe(hand.actor)
        hand = hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                                 else ActionKind.CALL))
    return hand


def check_hand_replay(binary, out):
    """Replay a complete searched river continuation with fresh policy/player state."""
    out.mkdir(parents=True, exist_ok=False)
    root = fixture()
    board = root.observe(root.actor).board
    cards = [c for c in DECK if c not in board]
    holdings = tuple(tuple(sorted(cards[i:i + 2])) for i in (0, 2, 4, 6))
    ranges = {s: tuple((h, .25) for h in holdings) for s in (0, 1)}
    hero = root.actor
    root = _world(root.events, board, {hero: holdings[0], 1 - hero: holdings[1]})

    class Blueprint:
        abstraction = HU200_EQUITY_SCHEMA
        description = {"fixture": "uniform; four stipulated holdings; no strength evidence"}

        def distribution(self, view):
            menu = choices(view, raise_cap=None, free_fold=False)
            return menu, (1 / len(menu),) * len(menu), False

    class SmallRange(HU200RiverSearchPolicy):
        def _ranges(self, root, bot_seat, deadline):
            return ranges, {"fixture": True}

    class ReplayOnly:
        expected_sha256 = file_hash(binary)

        def solve(self, *args):
            raise AssertionError("Fresh replay requested native compute")

    solver = ExternalTurnSolver(binary, out / "native")
    results = []
    for replayed in (False, True):
        policy = SmallRange(Blueprint(), ReplayOnly() if replayed else solver,
                            ledger=out / "ledger", replay_ledger=replayed)
        player = HU200RiverSearchPlayer(policy, seed=3)
        hand, actions = root, []
        while not hand.finished:
            view = hand.observe(hand.actor)
            if hand.actor == hero:
                action = player.choose_action(view)
            elif not view.legal_actions.call_amount and view.players[1 - hero].street_bet == 0:
                target = min(view.legal_actions.max_raise_to, view.pot * 3 // 2)
                action = Action(ActionKind.RAISE, target)
            else:
                action = Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                                else ActionKind.CALL)
            actions.append((action.kind.value, action.raise_to))
            hand = hand.apply(action)
        results.append(dict(actions=actions, events=hand.events, stats=dict(policy.stats)))
    if results[0]["actions"] != results[1]["actions"] or results[0]["events"] != results[1]["events"]:
        raise ValueError("Fresh searched-hand replay differs")
    result = dict(status="hand-replay-verified", native_solves=len(solver.records),
        actions=results[0]["actions"], stats=results[0]["stats"],
        replay_stats=results[1]["stats"], fresh_replay_did_not_solve=True,
        binary_sha256=file_hash(binary))
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


def run(binary, out, resume=None, retained_only=False, action_values=False):
    out.mkdir(parents=True, exist_ok=False)
    solver = ExternalTurnSolver(binary, out / "processes")
    retained = []
    if resume is not None:
        for path in sorted((resume / "processes").iterdir(), key=lambda p: (p / "request.json").stat().st_mtime_ns):
            manifest = json.loads((path / "manifest.json").read_text())
            for name, pin in manifest.items():
                member = path / name
                if member.stat().st_size != pin["bytes"] or file_hash(member) != pin["sha256"]:
                    raise ValueError("Retained correctness evidence changed")
            retained.append(path)

    def solve(request):
        if not retained:
            return solver.solve(request, monotonic() + 120, mode="quality")
        path = retained.pop(0)
        previous = json.loads((path / "request.json").read_text())
        science = {k: v for k, v in previous.items() if k not in ("seconds", "dump_path", "mode")}
        if science != json.loads(json.dumps(request)) or previous["mode"] != "quality":
            raise ValueError("Retained correctness request differs")
        receipt = json.loads((path / "receipt.json").read_text())
        events = [json.loads(line) for line in (path / "response.jsonl").read_text().splitlines()]
        completion = [e for e in events if e.get("event") == "completion"]
        if (receipt["executable_sha256"] != file_hash(binary) or not completion
                or completion[-1].get("status") != "play_complete"
                or completion[-1].get("iterations") != request["max_iterations"]):
            raise ValueError("Retained native solve was incomplete or used another binary")
        profiles = parse_profiles(previous, path / "profile.jsonl")
        solver.records.append(dict(receipt, retained_adapter_revalidated=True,
            quality=[e for e in events if e.get("event") == "quality"]))
        return profiles
    hand = fixture()
    root = hand.events
    board = hand.observe(hand.actor).board
    available = [c for c in DECK if c not in board]
    holdings = [tuple(sorted(available[i:i + 2])) for i in (0, 2, 4, 6)]
    weights = ((.1, .2, .3, .4), (.4, .3, .2, .1))
    ranges = {seat: tuple(zip(holdings, weights[seat], strict=True)) for seat in (0, 1)}
    rows = []
    first = None
    for amount in (None, 7333, 12000):
        if retained_only and not retained:
            break
        observed = hand if amount is None else hand.apply(Action(ActionKind.CHECK)).apply(
            Action(ActionKind.RAISE, amount))
        request = compile_tree(root, observed.events, stack_chips=20000)
        request.update(threads=6, compress=False, max_iterations=50,
                       memory_budget_bytes=2 * 1024**3,
                       ranges=[[{"hand": list(h), "weight": w} for h, w in ranges[s]]
                               for s in request["seat_map"]], locks=[])
        if action_values:
            request["action_values"] = True
        if first is not None:
            matrix = first[()]
            request["locks"] = [{"line": [], "board": list(board), "player": 0,
                "actions": request["nodes"][0]["actions"], "holdings": matrix.holdings,
                "strategy": matrix.probabilities.T.ravel().tolist()}]
        profiles = solve(request)
        repeat = solve(request)
        error = max(float(np.max(np.abs(profiles[k].probabilities - repeat[k].probabilities)))
                    for k in profiles)
        if error != 0:
            raise ValueError("Repeated native matrices differ")
        game = ReferenceRiver(root, ranges, observed_history=observed.events,
                              raise_cap=None, max_public_nodes=300000)
        profile = {}
        for node in game.nodes:
            if node.actor is None:
                continue
            matrix = profiles[line_key(betting_line(root, node.history))]
            columns = {choice.action: i for i, choice in enumerate(matrix.menu)}
            if set(columns) != {choice.action for choice in node.menu}:
                raise ValueError("Independent reference menus differ")
            profile[node.id] = np.asarray([[matrix.row(h)[columns[choice.action]]
                for choice in node.menu] for h in game.holdings[game.seats.index(node.actor)]])
        unrestricted = profile_quality(game, profile)
        locked_lines = {line_key(lock["line"]) for lock in request["locks"]}
        locked_nodes = [n.id for n in game.nodes if n.actor is not None
                        and line_key(betting_line(root, n.history)) in locked_lines]
        oracle = profile_quality(game, profile, locked_nodes=locked_nodes)
        quality = next(q for q in solver.records[-1]["quality"] if q["law"] == "search")
        action_ev_error = None
        if action_values:
            matrix = profiles[()]
            if matrix.action_values_chips is None:
                raise ValueError("Native conditional action values are missing")
            actor = game.nodes[0].actor
            seat_index = game.seats.index(actor)
            marginal = game.joint.sum(axis=1 if seat_index == 0 else 0)
            weighted = np.asarray([fsum(p * v for p, v in zip(matrix.row(h),
                matrix.action_values_chips[matrix.holding_indices[h]], strict=True))
                for h in game.holdings[seat_index]])
            root_ev = float(marginal @ weighted) - game.root_pot / 2
            action_ev_error = abs(root_ev - oracle["profile_values_bb"][seat_index] * 100)
            if action_ev_error > .01:
                raise ValueError("Conditional action values disagree with independent profile EV")
        errors = []
        for i, seat in enumerate(request["seat_map"]):
            index = game.seats.index(seat)
            ev = oracle["profile_values_bb"][index] * 100
            br = ev + oracle["best_response_gains_bb"][index] * 100
            errors.extend((abs(ev - quality["current_ev_chips"][i]),
                           abs(br - quality["mes_ev_chips"][i])))
        if amount is not None and not any(a.get("amount") == amount for n in request["nodes"]
                if not n["terminal"] for a in n["actions"]):
            raise ValueError("Exact wager absent")
        if first is not None and not np.allclose(profiles[()].probabilities,
                first[()].probabilities, rtol=0, atol=2e-5):
            raise ValueError("Prior hero matrix changed")
        rows.append(dict(inserted_raise_to=amount, nodes=len(request["nodes"]),
                         repeat_max_absolute_error=error, max_value_error_chips=max(errors),
                         constrained_reference=oracle, unrestricted_reference=unrestricted,
                         locked_reference_nodes=locked_nodes, native=quality))
        rows[-1]["action_ev_error_chips"] = action_ev_error
        (out / "progress.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        if max(errors) > .01:
            raise ValueError("Native/reference EV or best response differs by over .01 chips")
        if first is None:
            first = profiles
    result = dict(status="retained-analysis" if retained_only else "passed", binary_sha256=file_hash(binary), rows=rows,
                  note="Fixed small ranges establish integration, not full-range convergence or cost")
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Reuse verified completed solves after reference plumbing repair")
    parser.add_argument("--retained-only", action="store_true", help="Analyze existing profiles without invoking native solver")
    parser.add_argument("--action-values", action="store_true")
    parser.add_argument("--hand-replay-only", action="store_true")
    args = parser.parse_args()
    if args.hand_replay_only:
        check_hand_replay(args.binary, args.out)
    else:
        run(args.binary, args.out, args.resume, args.retained_only, args.action_values)


if __name__ == "__main__":
    main()
