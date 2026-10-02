"""Short exact-solver reference and macOS/Linux comparison; no model loading."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
from time import monotonic

import numpy as np

from src.arena.schedule import digest
from src.blueprint.hu20_river import HU20RiverGame
from src.blueprint.hu20_turn_solver import ExternalTurnSolver, file_hash
from src.blueprint.hu20_turn_tree import compile_tree, betting_line, line_key
from src.blueprint.river_cfr import profile_quality
from src.blueprint.search import DECK
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street

TOLERANCES = {"strategy_absolute": 1e-5, "value_chips_absolute": .001}


def fixture(street):
    hand = Hand.start(Table(("a", "b"), (2000, 2000), button=0), hand_id="solver-parity", seed=31)
    hand = hand.apply(Action(ActionKind.RAISE, 1000))
    while hand.observe(hand.actor).street != street:
        view = hand.observe(hand.actor)
        hand = hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL))
    return hand


def check(binary, out, reference=None):
    out.mkdir(parents=True, exist_ok=False)
    solver = ExternalTurnSolver(binary, out / "processes")
    rows = []
    for street in (Street.TURN, Street.RIVER):
        hand = fixture(street); view = hand.observe(hand.actor)
        root = hand.events
        cards = [c for c in DECK if c not in view.board]
        holdings = [tuple(sorted(cards[i:i+2])) for i in (0, 2, 4, 6)]
        ranges = {s: tuple((h, .25) for h in holdings) for s in (0, 1)}
        request = compile_tree(root, root)
        request.update(threads=2, compress=True, max_iterations=50, memory_budget_bytes=512*1024**2,
            ranges=[[{"hand": list(h), "weight": w} for h, w in ranges[s]] for s in request["seat_map"]], locks=[])
        started = monotonic()
        first = solver.solve(request, started+30)
        cold = monotonic()-started
        repeated = solver.solve(request, monotonic()+30, mode="quality")
        differences = [float(np.max(np.abs(first[k].probabilities-repeated[k].probabilities))) for k in first]
        if max(differences) > TOLERANCES["strategy_absolute"]:
            raise ValueError("Repeat request strategy exceeds frozen tolerance")
        record = solver.records[-1]
        oracle = None
        if street == Street.RIVER:
            game = HU20RiverGame(root, ranges, raise_cap=None)
            profile = {}
            for node in game.nodes:
                if node.actor is None: continue
                matrix = first[line_key(betting_line(root, node.history))]
                if matrix.menu != node.menu: raise ValueError("Native/oracle menu differs")
                profile[node.id] = np.asarray([matrix.row(h) for h in game.holdings[game.seats.index(node.actor)]])
            oracle = profile_quality(game, profile)
            quality = next(q for q in record["quality"] if q["law"] == "search")
            seat_map = request["seat_map"]
            for i, seat in enumerate(seat_map):
                index = game.seats.index(seat)
                if abs(quality["current_ev_chips"][i]/100-oracle["profile_values_bb"][index]) > .001/100:
                    raise ValueError("External/native river oracle value differs")
        rows.append({"street": street.value, "request_sha256": digest(request),
            "cold_seconds": cold, "repeat_max_absolute_difference": max(differences),
            "profiles": [{"line": list(k), "menu": [asdict(c) for c in m.menu],
                          "holdings": m.holdings, "probabilities": m.probabilities.tolist()}
                         for k, m in sorted(first.items(), key=lambda pair: str(pair[0]))],
            "quality": record["quality"], "river_oracle": oracle})
    result = {"status": "passed", "platform": platform.platform(), "architecture": platform.machine(),
        "python": platform.python_version(), "binary_sha256": file_hash(binary),
        "tolerances": TOLERANCES, "rows": rows}
    if reference:
        old = json.loads(reference.read_text())
        for left, right in zip(old["rows"], rows, strict=True):
            if left["street"] != right["street"] or left["request_sha256"] != right["request_sha256"]:
                raise ValueError("Parity request identity differs")
            for a, b in zip(left["profiles"], right["profiles"], strict=True):
                if any(a[k] != json.loads(json.dumps(b[k])) for k in ("line", "menu", "holdings")):
                    raise ValueError("Parity holding/action indexing differs")
                if not np.allclose(a["probabilities"], b["probabilities"], atol=1e-5, rtol=0):
                    raise ValueError("Cross-platform strategy parity failed")
            for a, b in zip(left["quality"], right["quality"], strict=True):
                for field in ("current_ev_chips", "mes_ev_chips"):
                    if not np.allclose(a[field], b[field], atol=.001, rtol=0):
                        raise ValueError("Cross-platform value parity failed")
        result["cross_platform_parity"] = "passed"
        result["reference_sha256"] = file_hash(reference)
    (out / "reference.json").write_text(json.dumps(result, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    args = p.parse_args()
    check(args.binary, args.out, args.reference)


if __name__ == "__main__": main()
