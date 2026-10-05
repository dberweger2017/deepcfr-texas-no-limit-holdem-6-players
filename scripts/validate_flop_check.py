"""Validate the external solver on native river fixtures and locked policies."""

import argparse
from dataclasses import replace
from itertools import combinations, permutations
import json
from math import sqrt
from pathlib import Path
from random import Random

import numpy as np

from src.arena.endgame_quality import _world
from src.blueprint.abstraction import choices
from src.blueprint.hu20_river import HU20RiverGame
from src.blueprint.river_cfr import profile_quality
from src.blueprint.search import DECK
from src.diagnostics.flop_check import (atomic_json, compile_tree, fixture_root,
                                        solver_action)
from src.diagnostics.flop_check_runtime import run_tool
from src.game.observation import BoardDealt, HandFinished, replay
from src.game.types import Street
from scripts.prepare_flop_check import load_policy


def small_ranges(board):
    cards = [c for c in DECK if c not in board]
    pairs = tuple(tuple(cards[i:i + 2]) for i in range(0, 8, 2))
    return {seat: tuple((pair, (i + 1) / 10) for i, pair in enumerate(pairs))
            for seat in (0, 1)}


def native_line(root, board, hands, line):
    hand = _world(root, tuple(board), hands)
    for expected in line:
        view = hand.observe(hand.actor)
        menu = choices(view, raise_cap=None, free_fold=False)
        item = next(c for c in menu if solver_action(view, c) == expected)
        hand = hand.apply(item.action)
    return hand


def locks_for_river(request, histories, ranges, source):
    rows = []; native_profile = {}
    for index, (node, history) in enumerate(zip(request["nodes"], histories, strict=True)):
        if node["terminal"]:
            continue
        seat = request["seat_map"][node["player"]]
        holdings = [list(h) for h, _ in ranges[seat]]
        probabilities = []
        for pair in holdings:
            view = replay(history, seat, tuple(pair))
            menu, p, _ = source.distribution(view)
            if [c.name for c in menu] != node["names"]:
                raise ValueError("Fixture blueprint lock menu differs")
            probabilities.append(p)
        matrix = np.asarray(probabilities)
        native_profile[index] = matrix
        rows.append({"line": node["line"], "board": request["board"],
                     "player": node["player"], "actions": node["actions"],
                     "holdings": holdings, "strategy": matrix.T.ravel().tolist()})
    return rows, native_profile


def monte_carlo(root, ranges, source, *, deals=20_000, seed=202610010904):
    if deals < 20_000:
        raise ValueError("V4 requires at least 20,000 independent deals")
    rng = Random(seed); view = replay(root, 0, ())
    # Rejection of independent weighted holdings gives the exact compatible
    # product law, without rebuilding a quadratic cumulative vector per deal.
    from bisect import bisect
    from itertools import accumulate
    pairs = {s:[h for h,w in ranges[s] if w>0] for s in (0,1)}
    cumulative = {s:list(accumulate(w for h,w in ranges[s] if w>0)) for s in (0,1)}
    if any(not cumulative[s] for s in (0,1)):
        raise ValueError("Empty Monte Carlo range")
    values = []
    for _ in range(deals):
        for attempt in range(100_000):
            a,b = (pairs[s][bisect(cumulative[s],rng.random()*cumulative[s][-1])] for s in (0,1))
            if not set(a) & set(b):break
        else:raise ValueError("Compatible range rejection failed")
        available = [c for c in DECK if c not in (*view.board, *a, *b)]
        board = view.board + tuple(rng.sample(available, 5 - len(view.board)))
        hand = _world(root, board, {0: a, 1: b})
        while not hand.finished:
            menu, p, _ = source.distribution(hand.observe(hand.actor))
            hand = hand.apply(rng.choices(menu, weights=p, k=1)[0].action)
        finish = next(e for e in hand.events if isinstance(e, HandFinished))
        values.append((finish.stacks[0] - view.players[0].stack - view.pot / 2) / 100)
    mean = float(np.mean(values)); half = 1.96 * float(np.std(values, ddof=1)) / sqrt(deals)
    return {"mean_bb": mean, "ci95": [mean - half, mean + half],
            "deals": deals, "seed": seed, "independent_deals": True,
            "range_sampler":"independent weighted holdings, reject blockers"}


def response_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def locks_for_runouts(request, histories, ranges, source):
    """Replay the real Python policy on every chance context of a small fixture."""
    flop = tuple(request["board"]); available = [c for c in DECK if c not in flop]
    boards = ({"flop": [flop], "turn": [flop + (c,) for c in available],
               "river": [flop + p for p in permutations(available, 2)]} if len(flop)==3
              else {"turn":[flop],"river":[flop+(c,) for c in available]})
    rows = []
    for node, history in zip(request["nodes"], histories, strict=True):
        if node["terminal"]:
            continue
        seat = request["seat_map"][node["player"]]
        holdings = [list(h) for h, _ in ranges[seat]]
        for board in boards[node["street"]]:
            events = []
            for event in history:
                if isinstance(event, BoardDealt):
                    cards = board[:3] if event.street == Street.FLOP else (
                        (board[3],) if event.street == Street.TURN else (board[4],))
                    event = replace(event, cards=cards)
                events.append(event)
            probabilities = []
            for holding in holdings:
                if set(holding).intersection(board):
                    probabilities.append([1 / len(node["actions"])] * len(node["actions"]))
                    continue
                menu, p, _ = source.distribution(replay(tuple(events), seat, tuple(holding)))
                if [c.name for c in menu] != node["names"]:
                    raise ValueError("Flop fixture lock menu differs")
                probabilities.append(p)
            rows.append({"line": node["line"], "board": list(board),
                         "player": node["player"], "actions": node["actions"],
                         "holdings": holdings,
                         "strategy": np.asarray(probabilities).T.ravel().tolist()})
    return rows


def validate_flop(binary, source, out, *, memory_bytes=512 * 1024**2):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    root = fixture_root("tiny-spr")
    request, histories = compile_tree(root)
    ranges = small_ranges(request["board"])
    request.update(mode="solve", memory_budget_bytes=memory_bytes,
                   ranges=[[{"hand": list(h), "weight": w} for h, w in ranges[s]]
                           for s in request["seat_map"]],
                   max_iterations=1000, progress_every=25, seconds=180)
    # Folding has a fixed terminal payoff on any runout. All-in terminals on
    # early streets instead represent an average over undealt cards.
    folds = [n for n in request["nodes"] if n["terminal"]
             and n["line"][-1]["kind"] == "Fold"]
    rng = Random(202610010906); queries = []; expected = []
    public = replay(root, 0, ())
    pairs = [(a, b) for a, _ in ranges[request["seat_map"][0]]
             for b, _ in ranges[request["seat_map"][1]] if not set(a) & set(b)]
    for _ in range(300):
        a, b = rng.choice(pairs); node = rng.choice(folds)
        available = [c for c in DECK if c not in (*public.board, *a, *b)]
        board = public.board + tuple(rng.sample(available, 2))
        hands = dict(zip(request["seat_map"], (a, b), strict=True))
        hand = native_line(root, board, hands, node["line"])
        finish = next(e for e in hand.events if isinstance(e, HandFinished))
        expected.append([finish.stacks[s] - public.players[s].stack - public.pot / 2
                         for s in request["seat_map"]])
        queries.append({"line": node["line"], "board": board, "hands": [a, b]})
    locks = locks_for_runouts(request, histories, ranges, source)
    path = out / "both-locked.json"
    atomic_json(path, dict(request, locks=locks, max_iterations=1, terminal_queries=queries))
    runtime = run_tool(binary, path, out / "both-locked", memory_bytes=memory_bytes,
                       threads=2, seconds=240, job_memory_bytes=6 * 1024**3)
    if runtime["status"] != "completed":
        raise RuntimeError(runtime["failure"])
    rows = response_rows(out / "both-locked/response.jsonl"); final = rows[-1]
    actual = [r["payoff_chips"] for r in rows if r["event"] == "payoff_query"]
    if len(actual) != len(expected):
        raise RuntimeError(f"Incomplete payoff queries: {len(actual)}/{len(expected)}")
    error = np.abs(np.asarray(actual) - expected)
    payoff_gate = {"gate": "V2", "kind": "tiny-spr-flop", "samples": len(actual),
                   "integer_chip_mismatches": int(np.count_nonzero(np.rint(actual) != expected)),
                   "maximum_float_chip_error": float(error.max()),
                   "passed": bool(np.all(np.rint(actual) == expected) and error.max() < 0.01)}
    if not payoff_gate["passed"]:
        atomic_json(out / "failure.json", {"gates": [payoff_gate]})
        return [payoff_gate]
    mc = monte_carlo(root, ranges, source, seed=202610010905)
    value = final["current_ev_chips"][request["seat_map"].index(0)] / 100
    gate = {"gate": "V4", "kind": "tiny-spr-flop", "solver_ev_bb": value,
            "native_mc": mc, "locks": len(locks), "public_nodes": len(request["nodes"]),
            "passed": mc["ci95"][0] <= value <= mc["ci95"][1]}
    gates = [payoff_gate, gate]
    if gate["passed"]:
        path = out / "equilibrium.json"; atomic_json(path, request)
        runtime = run_tool(binary, path, out / "equilibrium", memory_bytes=memory_bytes,
                           threads=2, seconds=240, job_memory_bytes=6 * 1024**3)
        if runtime["status"] != "completed":
            raise RuntimeError(runtime["failure"])
        final = response_rows(out / "equilibrium/response.jsonl")[-1]
        gates.append({"gate": "V5", "kind": "tiny-spr-flop", "passed":
                      final["exploitability_pct_pot"] <= 0.2,
                      "exploitability_pct_pot": final["exploitability_pct_pot"]})
    atomic_json(out / ("result.json" if all(g["passed"] for g in gates) else "failure.json"),
                {"gates": gates, "source": source.description})
    return gates


def validate_full_flop_payoffs(binary, out, *, memory_bytes=1024**3):
    """Cover ordinary multi-street bets and early all-in fixed-runout payoffs."""
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    root = fixture_root("short-spr"); request, _ = compile_tree(root)
    ranges = small_ranges(request["board"]); rng = Random(202610020010)
    view = replay(root, 0, ())
    pairs = [(a, b) for a, _ in ranges[request["seat_map"][0]]
             for b, _ in ranges[request["seat_map"][1]] if not set(a) & set(b)]
    terminals = [n for n in request["nodes"] if n["terminal"]]
    queries = []; expected = []; kinds = {}
    for _ in range(1000):
        node = rng.choice(terminals); a, b = rng.choice(pairs)
        available = [c for c in DECK if c not in (*view.board, *a, *b)]
        board = view.board + tuple(rng.sample(available, 2))
        hands = dict(zip(request["seat_map"], (a, b), strict=True))
        hand = native_line(root, board, hands, node["line"])
        finish = next(e for e in hand.events if isinstance(e, HandFinished))
        expected.append([finish.stacks[s] - view.players[s].stack - view.pot / 2
                         for s in request["seat_map"]])
        queries.append({"line": node["line"], "board": board, "hands": [a, b]})
        label = "showdown" if finish.showdown else "fold"
        kinds[label] = kinds.get(label, 0) + 1
    request.update(mode="solve", memory_budget_bytes=memory_bytes,
                   ranges=[[{"hand": list(h), "weight": w} for h, w in ranges[s]]
                           for s in request["seat_map"]], terminal_queries=queries,
                   max_iterations=1, progress_every=1, seconds=120)
    path = out / "request.json"; atomic_json(path, request)
    runtime = run_tool(binary, path, out / "solver", memory_bytes=memory_bytes,
                       threads=2, seconds=180, job_memory_bytes=5 * 1024**3)
    if runtime["status"] != "completed":
        raise RuntimeError(runtime["failure"])
    actual = [r["payoff_chips"] for r in response_rows(out / "solver/response.jsonl")
              if r["event"] == "payoff_query"]
    if len(actual) != len(expected):
        raise RuntimeError(f"Incomplete payoff queries: {len(actual)}/{len(expected)}")
    error = np.abs(np.asarray(actual) - expected)
    gate = {"gate": "V2", "kind": "short-spr-full-flop", "samples": len(actual),
            "terminal_types": kinds, "public_nodes": len(request["nodes"]),
            "integer_chip_mismatches": int(np.count_nonzero(np.rint(actual) != expected)),
            "maximum_float_chip_error": float(error.max()),
            "passed": bool(np.all(np.rint(actual) == expected) and error.max() < 0.01),
            "coverage": "multi-street non-all-in betting and early all-in fixed runouts"}
    atomic_json(out / ("result.json" if gate["passed"] else "failure.json"), {"gates": [gate]})
    return [gate]


def validate(binary, source, out, *, memory_bytes=512 * 1024**2):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    gates = []; rng = Random(202610010903)
    for kind in ("limped", "3-bet"):
        root = fixture_root(kind, street=Street.RIVER)
        request, histories = compile_tree(root)
        ranges = small_ranges(request["board"])
        request.update(mode="solve", memory_budget_bytes=memory_bytes,
                       ranges=[[{"hand": list(h), "weight": w} for h, w in ranges[s]]
                               for s in request["seat_map"]],
                       max_iterations=2000, progress_every=25, seconds=180)
        locks, profile = locks_for_river(request, histories, ranges, source)
        game = HU20RiverGame(root, ranges, raise_cap=None)
        if len(game.nodes) != len(request["nodes"]):
            raise ValueError("Native reference compilation differs")
        quality = profile_quality(game, profile)
        terminal = [n for n in request["nodes"] if n["terminal"]]
        pairs = [(a, b) for a, _ in ranges[request["seat_map"][0]]
                 for b, _ in ranges[request["seat_map"][1]] if not set(a) & set(b)]
        queries = []; expected = []
        public = replay(root, 0, ())
        for _ in range(300):
            node = rng.choice(terminal); a, b = rng.choice(pairs)
            holes = dict(zip(request["seat_map"], (a, b), strict=True))
            finished = native_line(root, request["board"], holes, node["line"])
            finish = next(e for e in finished.events if isinstance(e, HandFinished))
            expected.append([finish.stacks[s] - public.players[s].stack - public.pot / 2
                             for s in request["seat_map"]])
            queries.append({"line": node["line"], "board": request["board"], "hands": [a, b]})
        for target in (0, 1):
            document = dict(request, locks=[r for r in locks if r["player"] == target])
            if target == 0:
                document["terminal_queries"] = queries
            request_path = out / f"{kind}-target-{target}.json"
            atomic_json(request_path, document)
            run = out / f"{kind}-target-{target}"
            runtime = run_tool(binary, request_path, run, memory_bytes=memory_bytes,
                               threads=2, seconds=240, job_memory_bytes=6 * 1024**3)
            if runtime["status"] != "completed":
                raise RuntimeError(runtime["failure"])
            rows = response_rows(run / "response.jsonl")
            final = rows[-1]
            responder = 1 - target; physical = request["seat_map"][responder]
            expected_br = quality["profile_values_bb"][physical] + quality["best_response_gains_bb"][physical]
            actual_br = final["mes_ev_chips"][responder] / 100
            passed = abs(actual_br - expected_br) < 3e-5
            gates.append({"gate": "V3", "kind": kind, "target": target,
                          "passed": passed, "expected_br_bb": expected_br, "actual_br_bb": actual_br})
            if target == 0:
                actual = [r["payoff_chips"] for r in rows if r["event"] == "payoff_query"]
                if len(actual) != len(expected):
                    raise RuntimeError(f"Incomplete payoff queries: {len(actual)}/{len(expected)}")
                errors = np.abs(np.asarray(actual) - expected)
                mismatches = int(np.count_nonzero(np.rint(actual) != expected))
                gates.append({"gate": "V2", "kind": kind, "passed": not mismatches
                              and float(errors.max()) < 0.01,
                              "samples": len(actual), "integer_chip_mismatches": mismatches,
                              "maximum_float_chip_error": float(errors.max())})
            if not all(g["passed"] for g in gates):
                atomic_json(out / "failure.json", {"gates": gates})
                return gates
        locked = dict(request, locks=locks, max_iterations=1)
        path = out / f"{kind}-both-locked.json"; atomic_json(path, locked)
        run = out / f"{kind}-both-locked"
        runtime = run_tool(binary, path, run, memory_bytes=memory_bytes, threads=2, seconds=120,
                           job_memory_bytes=6 * 1024**3)
        if runtime["status"] != "completed":
            raise RuntimeError(runtime["failure"])
        final = response_rows(run / "response.jsonl")[-1]
        mc = monte_carlo(root, ranges, source)
        solver_ev = final["current_ev_chips"][request["seat_map"].index(0)] / 100
        gates.append({"gate": "V4", "kind": kind, "solver_ev_bb": solver_ev, "native_mc": mc,
                      "passed": mc["ci95"][0] <= solver_ev <= mc["ci95"][1]})
        path = out / f"{kind}-equilibrium.json"; atomic_json(path, request)
        run = out / f"{kind}-equilibrium"
        runtime = run_tool(binary, path, run, memory_bytes=memory_bytes, threads=2, seconds=240,
                           job_memory_bytes=6 * 1024**3)
        if runtime["status"] != "completed":
            raise RuntimeError(runtime["failure"])
        final = response_rows(run / "response.jsonl")[-1]
        gates.append({"gate": "V5", "kind": kind, "exploitability_pct_pot": final["exploitability_pct_pot"],
                      "passed": final["exploitability_pct_pot"] <= 0.2})
        atomic_json(out / "gates.json", {"gates": gates, "source": source.description})
        if not all(g["passed"] for g in gates):
            atomic_json(out / "failure.json", {"gates": gates})
            return gates
    atomic_json(out / "result.json", {"gates": gates, "source": source.description})
    return gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--flop", action="store_true")
    parser.add_argument("--payoffs-flop", action="store_true")
    args = parser.parse_args()
    if args.payoffs_flop:
        gates = validate_full_flop_payoffs(args.binary, args.out)
    else:
        source = load_policy(json.loads(args.plan.read_text())["policies"][0], args.inputs)
        gates = (validate_flop if args.flop else validate)(args.binary, source, args.out)
    if not all(g["passed"] for g in gates):
        raise SystemExit("Fixture validation gate failed")


if __name__ == "__main__":
    main()
