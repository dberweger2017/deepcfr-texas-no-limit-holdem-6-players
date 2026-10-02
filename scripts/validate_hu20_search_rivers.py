"""Validate 32 prospectively sampled river roots, never every turn runout."""

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
from pathlib import Path
from random import Random
from time import monotonic

import numpy as np

from scripts.calibrate_hu20_turn_search import replay_root
from scripts.evaluate_hu20_turn_search import load
from scripts.hu20_search_runtime import RunBudget, atomic_json, install_stop_handlers
from src.arena.endgame_quality import _world
from src.arena.schedule import digest
from src.blueprint.hu20_turn_search import HU20TurnSearchPolicy, TurnSearchConfig, observed_likelihood
from src.blueprint.hu20_turn_solver import ExternalTurnSolver, SolveFailure, file_hash
from src.blueprint.hu20_turn_tree import round_root
from src.blueprint.search import DECK
from src.game.observation import ActionTaken, replay
from src.game.types import Street


def slots(items):
    """Four slots per stratum; both seats, all lineages, no payoff reads."""
    strata=sorted({(i["root"]["kind"],i["root"]["button"]) for i in items})
    if len(strata)!=8:raise ValueError("Eight strata required")
    seeds=sorted({i["policy"]["seed"] for i in items})
    if len(seeds)!=3:raise ValueError("Three lineages required")
    output=[]
    for index,stratum in enumerate(strata):
        for sample in range(4):
            seed=seeds[(index*4+sample)%3]
            options=sorted((i for i in items if i["policy"]["seed"]==seed
                and (i["root"]["kind"],i["root"]["button"])==stratum),key=lambda i:i["root"]["spot"])
            if not options:raise ValueError("Missing sampled stratum/lineage")
            output.append({"slot":index*4+sample,"item":options[sample%len(options)],
                           "bot":sample%2,"sampling_seed":202610020900+index*4+sample})
    return output


def sampled_river(policy, root, bot, random, guard):
    solution=policy._resolve(root,bot,monotonic()+30)
    ranges=solution.ranges;board=replay(root,0,()).board;counts=Counter()
    for attempt in range(200):
        guard()
        for _ in range(10000):
            holes={s:random.choices([h for h,w in ranges[s]],weights=[w for h,w in ranges[s]],k=1)[0]
                   for s in (0,1)}
            if not set(holes[0]).intersection(holes[1]):break
        else:raise SolveFailure("zero_joint_support","River sampling compatible-deal refusal")
        river=random.choice([c for c in DECK if c not in board+holes[0]+holes[1]])
        hand=_world(root,board+(river,),holes)
        while not hand.finished and hand.observe(hand.actor).street==Street.TURN:
            view=hand.observe(hand.actor)
            matrix=policy._resolve(view.history,bot,monotonic()+30).matrix(view.history)
            action=random.choices(matrix.menu,weights=matrix.row(holes[view.seat]),k=1)[0].action
            view.legal_actions.validate(action);hand=hand.apply(action)
        if not hand.finished and hand.observe(hand.actor).street==Street.RIVER:
            counts["attempts"]=attempt+1
            return hand,dict(counts)
        counts["earlier_terminal"]+=1
    raise SolveFailure("river_sampling_no_live_root","200 frozen sampling attempts exhausted")


def check_conditioning(policy, history, bot, conditioned):
    """Recompute normalized turn matrix factors independently, including blockers."""
    root=round_root(history,Street.TURN);initial,_=policy._ranges(root,bot,monotonic()+30)
    river_board=replay(history,0,()).board
    expected={s:{h:w for h,w in initial[s] if not set(h).intersection(river_board)} for s in (0,1)}
    for index,event in enumerate(history):
        if not isinstance(event,ActionTaken) or event.street!=Street.TURN:continue
        matrix=policy._resolve(history[:index],bot,monotonic()+30).matrix(history[:index])
        for holding in expected[event.seat]:
            if expected[event.seat][holding]==0:continue
            factor=observed_likelihood(matrix.menu,matrix.row(holding),event.action)
            if event.seat!=bot:factor=max(factor,policy.config.opponent_likelihood_floor)
            expected[event.seat][holding]*=factor
    difference=0
    for seat in (0,1):
        total=sum(expected[seat].values())
        if total<=0:raise SolveFailure("zero_support_conditioning","Sampled matrix factors have no support")
        actual=dict(conditioned[seat])
        difference=max(difference,max(abs(w/total-actual[h]) for h,w in expected[seat].items()))
    if difference>1e-10:raise ValueError("River conditioning differs from used turn matrices")
    return difference


def run(references,config,inputs,binary,out,budget):
    out.mkdir(parents=True,exist_ok=False);sample=slots(references["items"])
    atomic_json(out/"sample-freeze.json",{"slots":sample,"settings":asdict(config),
        "selection_uses_payoffs":False,"maximum_attempts_per_slot":200})
    rows=[];started=monotonic()
    for slot in sample:
        budget.check();item=slot["item"];source=load(item["policy"],inputs)
        solver=ExternalTurnSolver(binary,out/"solves"/str(slot["slot"]),resource_check=budget.check)
        policy=HU20TurnSearchPolicy(source,solver,config)
        row={"slot":slot["slot"],"lineage":item["policy"]["seed"],"bot":slot["bot"],
            "stratum":[item["root"]["kind"],item["root"]["button"]],"status":"failed"}
        try:
            hand,counts=sampled_river(policy,replay_root(item["root"]),slot["bot"],Random(slot["sampling_seed"]),budget.check)
            root=round_root(hand.events);cold=monotonic()
            solution=policy._resolve(root,slot["bot"],cold+30)
            row["cold_seconds"]=monotonic()-cold
            row["conditioning_max_absolute_difference"]=check_conditioning(policy,root,slot["bot"],solution.ranges)
            for node in solution.request["nodes"]:
                if node["terminal"]:continue
                # Request's native actions were validated by the exact native compiler.
                if node["street"]!="river":raise ValueError("River validation crossed rounds")
            profiles=solver.solve(solution.request,monotonic()+300,mode="quality")
            difference=max(float(np.max(np.abs(m.probabilities-profiles[k].probabilities)))
                           for k,m in solution.profiles.items())
            if difference>1e-5:raise ValueError("River played/quality profile differs")
            row.update(status="passed",sampling=counts,public_request_sha256=digest(solution.request),
                quality=solver.records[-1]["quality"],range_coverage=solution.coverage,
                repeat_strategy_absolute_difference=difference,full_native=config.menu=="native")
        except (SolveFailure,ValueError,KeyError) as exc:
            row.update(cause=getattr(exc,"cause","invalid_output"),reason=str(exc))
        row["receipts"]=solver.records;rows.append(row)
        with (out/"rivers.jsonl").open("a") as stream:stream.write(json.dumps(row,sort_keys=True)+"\n")
        atomic_json(out/"status.json",{"status":"running","roots":len(rows),"seconds":monotonic()-started})
        del policy,solver,source;gc.collect()
    result={"status":"complete" if all(r["status"]=="passed" for r in rows) else "incomplete",
            "rows":rows,"roots":len(rows),"seconds":monotonic()-started,"settings":asdict(config)}
    atomic_json(out/"summary.json",result);atomic_json(out/"status.json",result)
    (out/"status.md").write_text(f"River validation: {result['status']}\n\nRoots: {len(rows)}/32\n")
    atomic_json(out/"manifest.json",{str(p.relative_to(out)):{"sha256":file_hash(p),"bytes":p.stat().st_size}
        for p in out.rglob("*") if p.is_file() and p.name!="manifest.json"})
    return result


def main():
    install_stop_handlers()
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("references","config","inputs","binary","out","admission","budget"):
        p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    budget=RunBudget(a.budget,a.out.parent,"river-validation",86400,json.loads(a.admission.read_text()))
    status="failed"
    reason=None
    try:
        result=run(json.loads(a.references.read_text()),TurnSearchConfig(**json.loads(a.config.read_text())),
                   a.inputs,a.binary,a.out,budget);status=result["status"]
    except BaseException as exc:
        reason=f"{type(exc).__name__}: {exc}"
        a.out.mkdir(parents=True,exist_ok=True)
        atomic_json(a.out/"failure.json",{"status":"failed","reason":reason})
        atomic_json(a.out/"status.json",{"status":"failed","reason":reason})
        (a.out/"status.md").write_text(f"River validation stopped: {reason}\n")
        atomic_json(a.out/"manifest.json",{str(p.relative_to(a.out)):{"sha256":file_hash(p),"bytes":p.stat().st_size}
            for p in a.out.rglob("*") if p.is_file() and p.name!="manifest.json"})
        raise
    finally:budget.close(status,reason)
    if status!="complete":raise SystemExit(1)


if __name__ == "__main__":main()
