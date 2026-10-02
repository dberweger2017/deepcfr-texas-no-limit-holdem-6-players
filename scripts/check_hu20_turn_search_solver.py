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


def check(binary, out, reference=None, config=None, compare_threads=False):
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
        if config:
            request.update(threads=config.threads,compress=config.compress,max_iterations=config.iterations)
        started = monotonic()
        first = solver.solve(request, started+30)
        cold = monotonic()-started
        # Asymmetric reference weights exercise both-factor reweighting.
        evaluation_ranges=[[dict(row,weight=weight) for row,weight in zip(rows,weights,strict=True)]
                           for rows,weights in zip(request["ranges"],((.1,.2,.3,.4),(.4,.3,.2,.1)),strict=True)]
        repeated = solver.solve(dict(request,evaluation_ranges=evaluation_ranges), monotonic()+30, mode="quality")
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
                if abs(quality["mes_ev_chips"][i]/100-(oracle["profile_values_bb"][index]
                        +oracle["best_response_gains_bb"][index])) > .001/100:
                    raise ValueError("External/native river best response differs")
            weighted_ranges={s:tuple((tuple(r["hand"]),r["weight"]) for r in evaluation_ranges[i])
                             for i,s in enumerate(seat_map)}
            weighted_game=HU20RiverGame(root,weighted_ranges,raise_cap=None)
            weighted_profile={n.id:np.asarray([first[line_key(betting_line(root,n.history))].row(h)
                for h in weighted_game.holdings[weighted_game.seats.index(n.actor)]])
                for n in weighted_game.nodes if n.actor is not None}
            weighted_oracle=profile_quality(weighted_game,weighted_profile)
            quality=next(q for q in record["quality"] if q["law"]=="reference")
            for i,seat in enumerate(seat_map):
                index=weighted_game.seats.index(seat)
                if abs(quality["current_ev_chips"][i]/100-weighted_oracle["profile_values_bb"][index])>.001/100:
                    raise ValueError("Original-law reweighted river EV differs")
                if abs(quality["mes_ev_chips"][i]/100-(weighted_oracle["profile_values_bb"][index]
                    +weighted_oracle["best_response_gains_bb"][index]))>.001/100:
                    raise ValueError("Original-law reweighted river BR differs")
            oracle["reference_law"]=weighted_oracle
        lock_check = None
        if street == Street.TURN:
            later=hand.apply(Action(ActionKind.CHECK)).apply(Action(ActionKind.RAISE,333))
            inserted=compile_tree(root,later.events)
            matrix=first[()]
            inserted.update({k:request[k] for k in ("threads","compress","max_iterations","memory_budget_bytes","ranges")})
            inserted["locks"]=[{"line":[],"board":request["board"],"player":0,
                "actions":request["nodes"][0]["actions"],"holdings":matrix.holdings,
                "strategy":matrix.probabilities.T.ravel().tolist()}]
            locked=solver.solve(inserted,monotonic()+30)
            error=float(np.max(np.abs(locked[()].probabilities-matrix.probabilities)))
            if error>1e-5:raise ValueError("Frozen hero matrix changed after exact wager insertion")
            if not any(a.get("amount")==333 for n in inserted["nodes"] if not n["terminal"] for a in n["actions"]):
                raise ValueError("Exact opponent wager missing")
            lock_check={"request_sha256":digest(inserted),"root_matrix_max_difference":error,
                "request_without_threads_sha256":digest({k:v for k,v in inserted.items() if k!="threads"}),
                "profiles":[{"line":list(k),"menu":[asdict(c) for c in m.menu],"holdings":m.holdings,
                             "probabilities":m.probabilities.tolist()}
                            for k,m in sorted(locked.items(),key=lambda pair:str(pair[0]))]}
        rows.append({"street": street.value, "request_sha256": digest(request),
            "request_without_threads_sha256":digest({k:v for k,v in request.items() if k!="threads"}),
            "cold_seconds": cold, "repeat_max_absolute_difference": max(differences),
            "profiles": [{"line": list(k), "menu": [asdict(c) for c in m.menu],
                          "holdings": m.holdings, "probabilities": m.probabilities.tolist()}
                         for k, m in sorted(first.items(), key=lambda pair: str(pair[0]))],
            "quality": record["quality"], "river_oracle": oracle,"lock_check":lock_check})
    result = {"status": "passed", "platform": platform.platform(), "architecture": platform.machine(),
        "python": platform.python_version(), "binary_sha256": file_hash(binary),
        "tolerances": TOLERANCES, "rows": rows,"config":asdict(config) if config else None}
    if reference:
        old = json.loads(reference.read_text())
        strategy_differences=[];value_differences=[]
        for left, right in zip(old["rows"], rows, strict=True):
            request_field="request_without_threads_sha256" if compare_threads else "request_sha256"
            if left["street"] != right["street"] or left[request_field] != right[request_field]:
                raise ValueError("Parity request identity differs")
            for a, b in zip(left["profiles"], right["profiles"], strict=True):
                if any(a[k] != json.loads(json.dumps(b[k])) for k in ("line", "menu", "holdings")):
                    raise ValueError("Parity holding/action indexing differs")
                if not np.allclose(a["probabilities"], b["probabilities"], atol=1e-5, rtol=0):
                    raise ValueError("Cross-platform strategy parity failed")
                strategy_differences.append(float(np.max(np.abs(np.asarray(a["probabilities"])-b["probabilities"]))))
            for a, b in zip(left["quality"], right["quality"], strict=True):
                for field in ("current_ev_chips", "mes_ev_chips"):
                    if not np.allclose(a[field], b[field], atol=.001, rtol=0):
                        raise ValueError("Cross-platform value parity failed")
                    value_differences.append(float(np.max(np.abs(np.asarray(a[field])-b[field]))))
            if left.get("lock_check") or right.get("lock_check"):
                a,b=left.get("lock_check"),right.get("lock_check")
                if not a or not b or a[request_field]!=b[request_field]:
                    raise ValueError("Lock parity fixture differs")
                for x,y in zip(a["profiles"],b["profiles"],strict=True):
                    if any(x[k]!=json.loads(json.dumps(y[k])) for k in ("line","menu","holdings")):
                        raise ValueError("Lock action/holding indexing differs")
                    if not np.allclose(x["probabilities"],y["probabilities"],atol=1e-5,rtol=0):
                        raise ValueError("Cross-platform locked strategy differs")
                    strategy_differences.append(float(np.max(np.abs(np.asarray(x["probabilities"])-y["probabilities"]))))
        comparison = ("thread_strategy_parity" if compare_threads else "cross_platform_parity" if old["platform"] != result["platform"]
                      or old["architecture"] != result["architecture"] else "repeat_reference_parity")
        result[comparison] = "passed"
        result["reference_max_strategy_absolute_difference"]=max(strategy_differences)
        result["reference_max_value_chips_absolute_difference"]=max(value_differences)
        result["reference_sha256"] = file_hash(reference)
    (out / "reference.json").write_text(json.dumps(result, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--config",type=Path,help="Selected native settings; use identically on both hosts")
    p.add_argument("--compare-threads",action="store_true",help="Permit only request thread-count differences")
    args = p.parse_args()
    from src.blueprint.hu20_turn_search import TurnSearchConfig
    config=TurnSearchConfig(**json.loads(args.config.read_text())) if args.config else None
    if config and config.menu!="native":p.error("Production parity requires native menu")
    check(args.binary, args.out, args.reference,config,args.compare_threads)


if __name__ == "__main__": main()
