"""Guarded public-root calibration; quality work is separate from play latency."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import gc
import itertools
import json
import platform
from pathlib import Path
from time import monotonic

import numpy as np

from scripts.evaluate_hu20_turn_search import load
from scripts.hu20_search_resume import SharedRootSolver, retained_screen
from scripts.hu20_search_runtime import RunBudget, atomic_json, install_stop_handlers, start_resource_watchdog
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
                              compress=compress,opponent_likelihood_floor=floor,
                              decision_seconds=protocol["decision_seconds"])


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


def quality_row(source, binary, config, item, bot, out, guard, *, quality_seconds=300,
                allocation_budget=None, shared_solver=None):
    from dataclasses import asdict
    root=replay_root(item["root"]);identity=digest(asdict(config))
    row={"root":f"{item['root']['spot']}/{item['policy']['seed']}/{bot}",
        "configuration_id":identity,"config":asdict(config),"bot_seat":bot,
        "host":platform.node(),"architecture":platform.machine(),
        "weight":item["root"].get("reach_weight",item["root"].get("multiplicity",1)),
        "blueprint_pct_pot":item.get("e_bp_pct_pot"),"reference_supported":False,
        "played_strategy_verified":False,"full_native_verified":False,"fallback":False,
        "stratum":[item["root"]["kind"],item["root"]["button"]],
        "reference_provenance":item.get("provenance"),"failures":[]}
    solver=shared_solver or ExternalTurnSolver(binary,out,resource_check=guard,allocation_budget=allocation_budget)
    if shared_solver:shared_solver.start_coordinate()
    policy=HU20TurnSearchPolicy(source,solver,config)
    started=monotonic()
    try:
        guard();solution=policy._resolve(root,bot,started+config.decision_seconds)
        public=replay(root,replay(root,0,()).actor,())
        matrix=solution.matrix(root)
        completed=policy._complete_matrix(public,matrix)
        policy._remember_play(public,completed,False,supported=matrix.holdings)
        policy._check(started+config.decision_seconds)
        row["cold_seconds"]=monotonic()-started
        row["actual_cold_seconds"]=row["cold_seconds"]
        row["range_coverage"]=solution.coverage
        if shared_solver:
            if solver.reused_play_seconds:
                row["cold_seconds"]=max(row["cold_seconds"]+solver.reused_play_seconds,
                    solver.source_cold_seconds or 0)
                row["cold_latency_accounting"]="remeasured preparation + retained native receipt; at least source cold"
                if row["cold_seconds"]>=config.decision_seconds:
                    raise SolveFailure("timeout","Reconstructed shared cold decision exceeds deadline")
            else:
                solver.source_cold_seconds=row["cold_seconds"]
    except SolveFailure as exc:
        row.update(cold_seconds=max(row.get("cold_seconds",0),monotonic()-started),fallback=True,
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
    screen=[]
    if protocol.get("resume_screen"):
        if protocol.get("research_latency_fallback") or protocol["decision_seconds"]!=30:
            raise ValueError("Retained resume is approved for the 30-second final only")
        screening=[asdict(c) for c in configurations(protocol)
                   if c.iterations==100 and c.menu=="native"]
        screen=retained_screen(protocol["resume_screen"],screening,screen_items(items))
        selected=finalists(screen,protocol)
        if [digest(asdict(c)) for c in selected] != protocol["finalist_configuration_ids"]:
            raise ValueError("Prospectively frozen timing-only finalists differ")
    out.mkdir(parents=True,exist_ok=False);rows=[];failure=None;started=monotonic()
    if screen:
        atomic_json(out/"retained-screen-provenance.json",protocol["resume_screen"])
        with (out/"curve.jsonl").open("x") as stream:
            for row in screen:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    # Keep all attempted cells; only final-stage rows can qualify for play.
    try:
        deadlines=[protocol["decision_seconds"]]
        if protocol.get("research_latency_fallback"):
            deadlines.append(protocol["research_latency_fallback"]["decision_seconds"])
        for decision_seconds in deadlines:
            if decision_seconds != protocol["decision_seconds"] and qualify_curve(rows,coordinates)["selected"]:
                break
            active_protocol=dict(protocol,decision_seconds=decision_seconds)
            deadline_screen=list(screen) if protocol.get("resume_screen") else []
            screening=[c for c in configurations(active_protocol) if c.iterations==protocol["staging"]["screen_iterations"]]
            specs={i["policy"]["name"]:i["policy"] for i in items}
            for stage in (("final",) if protocol.get("resume_screen") else ("screen","final")):
                candidates=screening if stage=="screen" else finalists(deadline_screen,active_protocol)
                for config in candidates:
                    targets=screen_items(items) if stage=="screen" else [(i,b) for i in items for b in (0,1)]
                    for name,spec in sorted(specs.items()):
                        budget.check();source=load(spec,inputs)
                        shared_solver=None
                        for item,bot in targets:
                            if item["policy"]["name"]!=name:continue
                            budget.check()
                            path=out/"solves"/stage/digest([name,config.__repr__(),item["root"]["spot"],bot])
                            if stage=="final" and protocol.get("share_identical_unlocked_seats") and config.opponent_likelihood_floor==0:
                                if bot==0:
                                    shared_solver=SharedRootSolver(ExternalTurnSolver(binary,path,
                                        resource_check=budget.check,
                                        allocation_budget=getattr(budget,"native_allocation_budget",None)))
                            else:shared_solver=None
                            row=quality_row(source,binary,config,item,bot,path,budget.check,
                                shared_solver=shared_solver,
                                allocation_budget=getattr(budget,"native_allocation_budget",None))
                            row["stage"]=stage
                            row["decision_deadline_seconds"]=decision_seconds
                            (screen if stage=="screen" else rows).append(row)
                            if stage=="screen":deadline_screen.append(row)
                            with (out/"curve.jsonl").open("a") as stream:
                                stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
                            atomic_json(out/"status.json",{"status":"running","roots":len(rows),
                                "stage":stage,"decision_deadline_seconds":decision_seconds,"seconds":monotonic()-started,
                                "fallbacks":sum(r['fallback'] for r in rows+screen)})
                            (out/"status.md").write_text(f"Calibration roots: {len(rows)}\n\nElapsed: {monotonic()-started:.1f}s\n")
                        del source;shared_solver=None;gc.collect()
    except Exception as exc:failure=f"{type(exc).__name__}: {exc}"
    result=qualify_curve(rows,coordinates,research_deadline=
        protocol.get("research_latency_fallback",{}).get("decision_seconds"))
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
    protocol=json.loads(a.protocol.read_text())
    reserve=protocol.get("river_reserve_seconds",0)
    remaining=86400-json.loads(a.budget.read_text())["used_seconds"]-reserve
    if remaining<=0:raise TimeoutError("No calibration allowance after frozen river reserve")
    budget=RunBudget(a.budget,a.out.parent,"calibration",remaining,json.loads(a.admission.read_text()))
    watchdog=start_resource_watchdog(budget)
    status="failed";reason=None
    try:
        result=run(protocol,json.loads(a.references.read_text()),
                   a.inputs,a.binary,a.out,budget)
        status=result["status"];reason=result.get("failure")
    finally:
        watchdog();budget.close(status,reason)
    if status not in ("qualified","owner-decision-needed"):raise SystemExit(1)


if __name__ == "__main__":main()
