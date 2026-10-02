"""Guarded public-root calibration; quality work is separate from play latency."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import gc
import itertools
import json
from pathlib import Path
from time import monotonic

import numpy as np

from scripts.evaluate_hu20_turn_search import load
from scripts.hu20_search_runtime import RunBudget, atomic_json, install_stop_handlers
from src.arena.endgame_quality import _world
from src.arena.schedule import digest
from src.blueprint.hu20_turn_search import HU20TurnSearchPolicy, TurnSearchConfig
from src.blueprint.hu20_turn_solver import ExternalTurnSolver, SolveFailure, file_hash
from src.diagnostics.hu20_search_protocol import qualify_curve
from src.game.hand import Hand, Table
from src.game.observation import BoardDealt, replay
from src.game.types import Action, ActionKind, Street


def replay_root(record):
    """Reconstruct public turn history; synthetic private cards supply legality only."""
    hand=Hand.start(Table(("a","b"),(2000,2000),button=record["button"]),
                    hand_id="search-calibration",seed=0)
    for event in record["events"]:
        if "action" in event:
            if hand.actor != event["seat"]:raise ValueError("Public root actor differs")
            hand=hand.apply(Action(ActionKind(event["action"]["kind"]),event["action"]["raise_to"]))
    board=tuple(record["board"])
    history=tuple(replace(e,cards=board[:3] if e.street==Street.FLOP else (board[3],))
                  if isinstance(e,BoardDealt) else e for e in hand.events)
    root=_world(history,board,{}).events
    if len(board)!=4 or root[-1].seat is None:raise ValueError("Need a live public turn root")
    events=[asdict(replace(e,hand_id="")) if index==0 else asdict(e) for index,e in enumerate(root)]
    identity=sha256(json.dumps(events,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if identity!=record["spot"]:raise ValueError("Frozen public root identity differs")
    return root


def configurations(protocol):
    # The frozen order tests native first, low work first, then declared axes.
    for menu, iterations, threads, compress, floor in itertools.product(
        protocol["menus"],protocol["iterations"],protocol["threads"],
        protocol["compress"],protocol["opponent_likelihood_floor"]):
        yield TurnSearchConfig(menu=menu,iterations=iterations,threads=threads,
                              compress=compress,opponent_likelihood_floor=floor)


def screen_items(items):
    strata=sorted({(i["root"]["kind"],i["root"]["button"]) for i in items})
    output=[]
    for index,stratum in enumerate(strata):
        candidates=[i for i in items if (i["root"]["kind"],i["root"]["button"])==stratum]
        spot=min(i["root"]["spot"] for i in candidates)
        lineages=sorted((i for i in candidates if i["root"]["spot"]==spot),key=lambda i:i["policy"]["seed"])
        output.append((lineages[index%len(lineages)],index%2))
    return output


def finalists(rows, protocol):
    settings={}
    for row in rows:
        if row["config"]["menu"]=="native":
            settings.setdefault(row["configuration_id"],[]).append(row)
    selected=[]
    for floor in protocol["opponent_likelihood_floor"]:
        ranked=sorted((v for v in settings.values() if v[0]["config"]["opponent_likelihood_floor"]==floor),
            key=lambda v:(float(np.percentile([r["cold_seconds"] for r in v],95)),
                          float(np.median([r["cold_seconds"] for r in v])),v[0]["configuration_id"]))
        for v in ranked[:protocol["staging"]["finalist_settings_per_floor"]]:
            for count in protocol["iterations"]:
                selected.append(TurnSearchConfig(**dict(v[0]["config"],iterations=count)))
    return selected


def quality_row(source, binary, config, item, bot, out, guard, *, quality_seconds=300):
    from dataclasses import asdict
    root=replay_root(item["root"]);identity=digest(asdict(config))
    row={"root":f"{item['root']['spot']}/{item['policy']['seed']}/{bot}",
        "configuration_id":identity,"config":asdict(config),"bot_seat":bot,
        "weight":item["root"].get("reach_weight",item["root"].get("multiplicity",1)),
        "blueprint_pct_pot":item.get("e_bp_pct_pot"),"reference_supported":False,
        "played_strategy_verified":False,"full_native_verified":False,"fallback":False,
        "stratum":[item["root"]["kind"],item["root"]["button"]],
        "reference_provenance":item.get("provenance"),"failures":[]}
    solver=ExternalTurnSolver(binary,out,resource_check=guard)
    policy=HU20TurnSearchPolicy(source,solver,config)
    started=monotonic()
    try:
        guard();solution=policy._resolve(root,bot,started+30)
        public=replay(root,replay(root,0,()).actor,())
        matrix=solution.matrix(root)
        completed=policy._complete_matrix(public,matrix)
        policy._remember_play(public,completed,False,supported=matrix.holdings)
        policy._check(started+30)
        row["cold_seconds"]=monotonic()-started
        row["range_coverage"]=solution.coverage
    except SolveFailure as exc:
        row.update(cold_seconds=monotonic()-started,fallback=True,
                   fallback_cause=exc.cause,residual_pct_pot=item.get("e_bp_pct_pot"))
        row["failures"].append({"phase":"play","cause":exc.cause,"reason":str(exc)})
        # A complete native #145 blueprint metric evaluates the policy actually used.
        verified=bool(item.get("reference_native_verified") and item.get("reference_ranges")
                      and item.get("e_bp_pct_pot") is not None)
        row.update(reference_supported=verified,full_native_verified=verified,
                   played_strategy_verified=verified)
        row["receipts"]=solver.records
        return row
    reference=item.get("reference_ranges")
    if not reference:
        row["failures"].append({"phase":"quality","cause":"missing_reference"})
    elif config.menu != "native":
        row["failures"].append({"phase":"quality","cause":"reduced_full_native_evaluation_unavailable"})
    else:
        try:
            guard()
            request=dict(solution.request,evaluation_ranges=reference)
            profiles=solver.solve(request,monotonic()+quality_seconds,mode="quality")
            difference=max(float(np.max(np.abs(profiles[k].probabilities
                                -solution.profiles[k].probabilities))) for k in profiles)
            row["repeat_strategy_absolute_difference"]=difference
            row["played_strategy_verified"]=difference<=1e-5
            metrics=solver.records[-1]["quality"]
            row["quality_laws"]=metrics
            original=next(m for m in metrics if m["law"]=="reference")
            row["residual_pct_pot"]=original["exploitability_pct_pot"]
            row["reference_supported"]=all(v>=1-1e-6 for v in original["retained_mass"])
            row["full_native_verified"]=item.get("reference_native_verified") is True
        except (SolveFailure,KeyError,StopIteration,ValueError) as exc:
            row["failures"].append({"phase":"quality","cause":getattr(exc,"cause","invalid_quality"),
                                    "reason":str(exc)})
    row["receipts"]=solver.records
    row["policy_records"]=policy.records
    return row


def run(protocol, references, inputs, binary, out, budget):
    if protocol["stage"] != "frozen-final":
        raise ValueError("Push staged corpus/settings freeze before calibration")
    if references.get("145_final_report_pushed") is not True or not references.get("final_report_sha256"):
        raise ValueError("Calibration requires published #145 final report")
    if protocol.get("reference_index_sha256")!=digest(references):
        raise ValueError("Frozen reference index differs")
    items=references["items"]
    coordinates=[f"{i['root']['spot']}/{i['policy']['seed']}/{s}" for i in items for s in (0,1)]
    if len(set(coordinates))!=len(coordinates):raise ValueError("Duplicate reference root/lineage")
    if {tuple((i['root']['kind'],i['root']['button'])) for i in items} != {
        (k,b) for k in ("limped","min-raised","pot-raised","3-bet") for b in (0,1)}:
        raise ValueError("Calibration needs all eight frozen strata")
    out.mkdir(parents=True,exist_ok=False);rows=[];screen=[];failure=None;started=monotonic()
    # Keep all attempted cells; only final-stage rows can qualify for play.
    try:
        screening=[c for c in configurations(protocol) if c.iterations==protocol["staging"]["screen_iterations"]]
        specs={i["policy"]["name"]:i["policy"] for i in items}
        for stage in ("screen","final"):
            candidates=screening if stage=="screen" else finalists(screen,protocol)
            for config in candidates:
                targets=screen_items(items) if stage=="screen" else [(i,b) for i in items for b in (0,1)]
                for name,spec in sorted(specs.items()):
                    budget.check();source=load(spec,inputs)
                    for item,bot in targets:
                        if item["policy"]["name"]!=name:continue
                        budget.check()
                        path=out/"solves"/stage/digest([name,config.__repr__(),item["root"]["spot"],bot])
                        row=quality_row(source,binary,config,item,bot,path,budget.check)
                        row["stage"]=stage
                        (screen if stage=="screen" else rows).append(row)
                        with (out/"curve.jsonl").open("a") as stream:
                            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
                        atomic_json(out/"status.json",{"status":"running","roots":len(rows),
                            "stage":stage,"seconds":monotonic()-started,
                            "fallbacks":sum(r['fallback'] for r in rows+screen)})
                        (out/"status.md").write_text(f"Calibration roots: {len(rows)}\n\nElapsed: {monotonic()-started:.1f}s\n")
                    del source;gc.collect()
    except Exception as exc:failure=f"{type(exc).__name__}: {exc}"
    result=qualify_curve(rows,coordinates)
    if failure:result.update(status="incomplete",failure=failure,selected=None,tier=None)
    result.update(protocol=protocol,reference_index_sha256=digest(references),
        seconds=monotonic()-started,roots=len(rows),screen_curve=screen,
        exclusions=references.get("exclusions",[]))
    atomic_json(out/"summary.json",result);atomic_json(out/"status.json",result)
    (out/"status.md").write_text(f"Calibration: {result['status']}\n\nRoots: {len(rows)}\n\nFailure: {failure}\n")
    atomic_json(out/"manifest.json",{str(p.relative_to(out)): {"sha256":file_hash(p),"bytes":p.stat().st_size}
        for p in out.rglob("*") if p.is_file() and p.name!="manifest.json"})
    return result


def main():
    install_stop_handlers()
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("protocol","references","inputs","binary","out","admission","budget"):
        p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    budget=RunBudget(a.budget,a.out.parent,"calibration",86400,json.loads(a.admission.read_text()))
    status="failed";reason=None
    try:
        result=run(json.loads(a.protocol.read_text()),json.loads(a.references.read_text()),
                   a.inputs,a.binary,a.out,budget)
        status=result["status"];reason=result.get("failure")
    finally:budget.close(status,reason)
    if status not in ("qualified","owner-decision-needed"):raise SystemExit(1)


if __name__ == "__main__":main()
