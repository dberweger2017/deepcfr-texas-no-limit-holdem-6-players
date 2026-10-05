"""Frozen paired HU20 extraction/search worker; never provisions paid hardware."""

import argparse
from collections import Counter
import gc
import gzip
import json
import platform
from pathlib import Path
from random import Random
import subprocess
from time import perf_counter

from scripts.evaluate_hu20_cfr_average import opponent, summarize
from scripts.hu20_search_runtime import RunBudget, PaidWorkerBudget, atomic_json, install_stop_handlers, start_resource_watchdog
from src.arena.catalog import Checkpoint
from src.arena.runner import public_events
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import choices, information_key, HU20_UNCAPPED_SCHEMA
from src.blueprint.artifact import FrozenBlueprint
from src.blueprint.hu20_turn_search import HU20TurnSearchPolicy, TurnSearchConfig
from src.blueprint.hu20_turn_solver import ExternalTurnSolver, file_hash
from src.diagnostics.cfr_average import DiagnosticAverage
from src.diagnostics.hu20_search_protocol import select_base
from src.diagnostics.stackoff_tails import snapshot, hand_tails
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


def load(spec, inputs):
    path = inputs / spec["path"]
    if path.stat().st_size != spec["bytes"] or file_hash(path) != spec["sha256"]:
        raise ValueError("Frozen policy length/hash differs")
    source = (DiagnosticAverage(path, spec["sha256"]) if spec["strategy"] == "average"
              else FrozenBlueprint(Checkpoint(spec["name"],str(path),spec["sha256"],spec["format"]),path))
    if (source.description["training_seed"] != spec["seed"]
            or source.description["iteration"] != spec["iteration"]
            or source.abstraction != HU20_UNCAPPED_SCHEMA or source.raise_cap is not None):
        raise ValueError("Frozen lineage/schema differs")
    if spec["strategy"] == "average" and source.description["source_checkpoint_sha256"] != spec["checkpoint_sha256"]:
        raise ValueError("Average checkpoint provenance differs")
    return source


def play(source, spec, panel, root, block, rotation, guard=lambda:None, *, search=None, arm=None, failure_dir=None):
    deal = stream_seed(root,"test","deal",2,block)
    random = Random(stream_seed(root,"test","action",2,block,0))
    rival = opponent(panel, search or source, stream_seed(root,"test","opponent",2,block,1))
    hand_id = f"turn-search/{panel['name']}/{block}/{rotation}"
    hand = Hand.start(Table(("seat0","seat1"),(2000,2000),button=block%2),hand_id=hand_id,seed=deal)
    records_begin = len(search.records) if search else 0
    stats_before = Counter(search.stats) if search else Counter()
    actions = []; coverage = Counter(); started = perf_counter()
    try:
        for index in range(1000):
            guard()
            if hand.finished: break
            view = hand.observe(hand.actor); logical = int(hand.actor != rotation)
            key = mass_status = None
            base_trained = None
            if not logical:
                base_menu, base_p, base_trained = source.distribution(view)
                key = information_key(view,base_menu,schema=source.abstraction)
                mass_status = ("missing" if not base_trained else "zero_mass" if key in getattr(source,"zero_mass",())
                               else "positive_mass" if isinstance(source,DiagnosticAverage) else "current")
                menu,p,trained = search.distribution(view,query_kind="play") if search else (base_menu,base_p,base_trained)
                action = random.choices(menu,weights=p,k=1)[0].action
                coverage[mass_status] += 1; coverage[view.street.value+":"+mass_status] += 1
            else:
                menu = choices(view,raise_cap=None,free_fold=False); p = trained = None
                action = rival.choose_action(view)
            observed = snapshot(view,menu,p,trained,None); observed["logical_player"] = logical
            entry = {"index":index,"seat":hand.actor,"logical_player":logical,"street":view.street.value,
                "kind":action.kind.value,"raise_to":action.raise_to,"observation":observed,
                "target_key":key,"average_mass_status":mass_status,"base_trained":base_trained}
            if logical and hasattr(rival,"telemetry"):
                entry["lbr"] = dict(rival.telemetry[-1])
            view.legal_actions.validate(action); actions.append(entry); hand = hand.apply(action)
    except BaseException as exc:
        if failure_dir:
            failure_dir.mkdir(parents=True,exist_ok=True)
            coordinate=[spec["name"],arm or spec["strategy"],panel["name"],block,rotation]
            atomic_json(failure_dir/(digest(coordinate)+".json"),{
                "status":"incomplete","coordinate":coordinate,"hand_id":hand_id,
                "deal_seed":deal,"root_seed":root,"button":block%2,"actions":actions,
                "failure":f"{type(exc).__name__}: {exc}"})
        raise
    if not hand.finished: raise RuntimeError("Native hand decision limit")
    chips = [p.stack-2000 for p in hand.observe(0).players]
    replay = Hand.start(hand.table,hand_id=hand_id,seed=deal)
    for entry in actions:
        if replay.actor != entry["seat"]: raise ValueError("Replay actor differs")
        replay = replay.apply(Action(ActionKind(entry["kind"]),entry["raise_to"]))
    events = digest(public_events(hand.events))
    if (not replay.finished or sum(chips)
            or [p.stack-2000 for p in replay.observe(0).players] != chips
            or digest(public_events(replay.events)) != events):
        raise ValueError("Native settlement/events differ")
    row = {"status":"complete","policy":spec["name"],"strategy":spec["strategy"],"arm":arm or spec["strategy"],
        "host":platform.node(),"architecture":platform.machine(),
        "seed":spec["seed"],"players":2,"panel":panel["name"],"contract":panel["contract"],
        "block":block,"rotation":rotation,"button":block%2,"root_seed":root,"deal_seed":deal,
        "hand_id":hand_id,"actions":actions,"target_chips":chips[rotation],"net_chips_by_seat":chips,
        "public_events_sha256":events,"native_replay_verified":True,"coverage":dict(coverage),
        "seconds":perf_counter()-started,"lbr_zero_likelihood":getattr(rival,"zero_likelihood",[]),
        "search_records":search.records[records_begin:] if search else [],
        "search_counts":dict(search.stats-stats_before) if search else {}}
    row["tails"] = hand_tails(row)
    return row


def summarize_phase(rows,phase):
    # The existing paired estimator uses current/average names. Keep those aliases
    # inside arithmetic only; retained hands identify their actual policy and arm.
    analysis=[dict(row,strategy={"base":"current","search":"average"}[row["arm"]])
              for row in rows] if phase=="arena" else rows
    result=summarize(analysis)
    if phase=="arena":
        bases={row["seed"]:row["strategy"] for row in rows}
        for panel in result["panels"]:
            panel["arm"]={"current":"base","average":"search"}[panel["strategy"]]
            panel["strategy"]=bases[panel["seed"]]
        for change in result["three_lineage_changes"]:
            change["search_minus_base"]=change.pop("average_minus_current")
    return result


SEARCH_PHASES = ("arena", "timing")


def timing_summary(rows, loaded):
    """Outcome-blind cost of each joint block (both arms, positions and lineages); no payoffs read."""
    blocks = {}
    for row in rows:
        blocks.setdefault((row["panel"], row["block"]), []).append(row["seconds"])
    panels = {}
    for (panel, _), seconds in blocks.items():
        panels.setdefault(panel, []).append(sum(seconds))
    def quantile(values, q):
        ordered = sorted(values)
        return ordered[min(len(ordered)-1, int(q*len(ordered)))]
    solves = [r for row in rows for r in row.get("search_records", []) if r["status"] in ("completed", "failure")]
    latencies = [r["seconds"] for r in solves if r["status"] == "completed"]
    return {"timing_panels": {name: {"paired_blocks": len(v), "seconds_per_joint_block_mean": sum(v)/len(v),
                "seconds_per_joint_block_p50": quantile(v, .5), "seconds_per_joint_block_p95": quantile(v, .95),
                "seconds_per_joint_block_max": max(v)} for name, v in sorted(panels.items())},
            "load_seconds": sum(item["seconds"] for item in loaded),
            "solves": len(solves), "solve_failures": sum(r["status"] == "failure" for r in solves),
            "solve_seconds_mean": sum(latencies)/len(latencies) if latencies else None,
            "solve_seconds_p95": quantile(latencies, .95) if latencies else None,
            "solve_seconds_max": max(latencies) if latencies else None}


def run(plan, inputs, out, budget, *, phase="part-a", binary=None, search_config=None,
        worker_index=0, worker_count=1):
    if not 0 <= worker_index < worker_count:
        raise ValueError("Invalid independent worker coordinate")
    if phase not in SEARCH_PHASES and worker_count != 1:
        raise ValueError("M4 phases use one guarded worker")
    if phase=="arena":
        from dataclasses import asdict
        if (plan.get("stage")!="frozen-final" or len(plan["models"])!=3
                or len({s["seed"] for s in plan["models"]})!=3
                or plan.get("selected_search_config")!=asdict(search_config)
                or not plan.get("part_a_summary_sha256") or not plan.get("calibration_sha256")):
            raise ValueError("Arena must bind all three selected-base lineages and published science")
        if any(p["blocks"]!=(2048 if p["name"] in ("lbr","native-pressure") else 256) for p in plan["panels"]):
            raise ValueError("Do not shrink the scientific arena to fit M4")
    if phase == "timing":
        from dataclasses import asdict
        # A separate outcome-blind timing schedule: its own deal root, small balanced block counts.
        if (plan.get("stage") != "timing-pilot" or len({s["seed"] for s in plan["models"]}) != 3
                or plan.get("selected_search_config") != asdict(search_config)):
            raise ValueError("Timing pilot must bind three lineages and the selected search configuration")
    expected = (2*len(plan["models"])*(2 if phase in SEARCH_PHASES else 1)
                *sum(len(range(worker_index,p["blocks"],worker_count)) for p in plan["panels"]))
    out.mkdir(parents=True,exist_ok=False)
    started = perf_counter(); rows = []; loaded = []; failure = None
    source_commit = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    try:
        if phase == "part-a" and plan["stage"] != "frozen-final":
            raise ValueError("Push frozen timing/counts before Part A outcomes")
        for spec in plan["models"]:
            budget.check(); begun = perf_counter(); source = load(spec,inputs)
            loaded.append({"model":spec,"seconds":perf_counter()-begun,"description":source.description})
            solver = ExternalTurnSolver(binary,out / "solver" / spec["name"],
                resource_check=budget.check,
                allocation_budget=getattr(budget,"native_allocation_budget",None),
                profile_retention=plan.get("evidence_retention",{}).get("profile_rule")
                    if phase=="arena" else None) if phase in SEARCH_PHASES else None
            policy = HU20TurnSearchPolicy(source,solver,search_config) if solver else None
            arms = ("base","search") if phase in SEARCH_PHASES else (spec["strategy"],)
            for arm in arms:
                with gzip.open(out / (spec["name"]+"."+arm+".hands.jsonl.gz"),"wt") as stream:
                    for panel in plan["panels"]:
                        for block in range(worker_index,panel["blocks"],worker_count):
                            for rotation in (0,1):
                                row = play(source,spec,panel,plan["root"],block,rotation,budget.check,
                                    search=policy if arm == "search" and phase in SEARCH_PHASES else None,arm=arm,failure_dir=out/"partials")
                                stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n");stream.flush()
                                compact = {k:v for k,v in row.items() if k not in
                                    ("actions","search_records","search_counts","lbr_zero_likelihood")}
                                compact["actions"] = [{"target_key":a["target_key"],"street":a["street"],
                                    **({"lbr":{"completed":a["lbr"]["completed"]}} if "lbr" in a else {})}
                                    for a in row["actions"]]
                                if phase == "timing":
                                    compact["search_records"] = row["search_records"]
                                rows.append(compact)
                                if policy:
                                    policy.records.clear(); solver.records.clear()
                                if len(rows)%12 == 0:
                                    atomic_json(out/"status.json",{"status":"running","phase":phase,
                                        "hands":len(rows),"expected_hands":expected,
                                        "seconds":perf_counter()-started})
                                    (out/"status.md").write_text(f"Phase: {phase}\n\nHands: {len(rows)}/{plan['expected_hands']}\n\nElapsed: {perf_counter()-started:.1f}s\n")
            del policy,solver,source; gc.collect()
        if len(rows) != expected: raise ValueError("Frozen schedule coverage differs")
        aggregate = timing_summary(rows,loaded) if phase == "timing" else summarize_phase(rows,phase)
        decision = select_base(aggregate) if phase == "part-a" else None
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"; aggregate = {}; decision = None
    result = {"status":"incomplete" if failure else "complete","failure":failure,"phase":phase,
        "source":source_commit,"plan_sha256":digest(plan),"plan":plan,"hands":len(rows),"loaded":loaded,
        "worker_index":worker_index,"worker_count":worker_count,"expected_worker_hands":expected,
        "seconds":perf_counter()-started,"base_decision":decision,
        "arm_labels":{"base":"base-only","search":"base-plus-search"} if phase in SEARCH_PHASES else None,
        **aggregate}
    atomic_json(out/"summary.json",result)
    atomic_json(out/"status.json",{k:result[k] for k in ("status","phase","hands","seconds","failure")})
    (out/"status.md").write_text(f"Phase: {phase}\n\nStatus: {result['status']}\n\nHands: {len(rows)}/{plan['expected_hands']}\n\nFailure: {failure}\n")
    manifest = {str(p.relative_to(out)):{"bytes":p.stat().st_size,"sha256":file_hash(p)}
                for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    atomic_json(out/"manifest.json",manifest)
    return result


def main():
    install_stop_handlers()
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("plan","inputs","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--admission",type=Path); p.add_argument("--budget",type=Path)
    p.add_argument("--phase",choices=("pilot","part-a","arena","timing"),default="part-a")
    p.add_argument("--binary",type=Path); p.add_argument("--search-config",type=Path)
    p.add_argument("--paid-approval",type=Path)
    p.add_argument("--worker-index",type=int,default=0);p.add_argument("--worker-count",type=int,default=1)
    args=p.parse_args(); plan=json.loads(args.plan.read_text())
    if args.phase in SEARCH_PHASES:
        # The arena needs its approved quote; the timing pilot needs the owner-approved pilot document.
        if not args.paid_approval or not args.binary or not args.search_config:
            p.error("Search phases need an owner approval document, binary and selected configuration")
        approval=json.loads(args.paid_approval.read_text())
        if (approval.get("owner_approved") is not True or approval.get("arena_plan_sha256") != digest(plan)
                or approval.get("selected_settings_parity") != "passed"):
            raise ValueError("Owner approval and selected-settings Linux/M4 parity required")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    config=TurnSearchConfig(**json.loads(args.search_config.read_text())) if args.search_config else None
    limit=21600 if args.phase=="part-a" else 300 if args.phase=="pilot" else plan["max_seconds"]
    if args.phase in SEARCH_PHASES:
        from dataclasses import asdict
        budget=PaidWorkerBudget(args.out.parent,approval,digest(plan),digest(asdict(config)))
    else:
        if not args.admission or not args.budget: p.error("M4 phases need admission and cumulative budget journal")
        admission=json.loads(args.admission.read_text())
        budget=RunBudget(args.budget,args.out.parent,args.phase,limit,admission)
    watchdog=start_resource_watchdog(budget)
    status="failed"; reason=None
    try:
        result=run(plan,args.inputs,args.out,budget,phase=args.phase,binary=args.binary,search_config=config,
                   worker_index=args.worker_index,worker_count=args.worker_count)
        status=result["status"];reason=result["failure"]
        print(json.dumps({k:result[k] for k in ("status","hands","failure","base_decision")}))
    finally:
        watchdog();budget.close(status,reason)
    if status != "complete":raise SystemExit(1)


if __name__ == "__main__":main()
