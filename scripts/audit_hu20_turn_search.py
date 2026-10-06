"""Independent streaming native replay and merged paired arithmetic, without models."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from time import perf_counter

from scripts.evaluate_hu20_turn_search import summarize_phase
from scripts.hu20_search_runtime import atomic_json, owned_rss
from src.arena.runner import public_events
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import Choice, choices, information_key, HU20_UNCAPPED_SCHEMA
from src.blueprint.hu20_turn_solver import file_hash
from src.diagnostics.stackoff_tails import snapshot, hand_tails
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


def audit(directories, *, allow_incomplete=False):
    started=perf_counter();seen=set();rows=[];search_counts=Counter();lbr=Counter();latencies=[];host_decisions={}
    plan=None;phase=None;worker_ids=set();manifest_ids=[]
    for directory in directories:
        summary=json.loads((directory/"summary.json").read_text())
        if (not allow_incomplete and summary["status"]!="complete") or digest(summary["plan"])!=summary["plan_sha256"]:
            raise ValueError("Incomplete or changed worker protocol")
        if plan is None:plan=summary["plan"];phase=summary["phase"]
        if plan!=summary["plan"] or phase!=summary["phase"]:raise ValueError("Worker plans differ")
        if summary["worker_index"] in worker_ids:raise ValueError("Duplicate worker")
        worker_ids.add(summary["worker_index"])
        manifest=json.loads((directory/"manifest.json").read_text())
        for name,spec in manifest.items():
            path=directory/name
            if path.stat().st_size!=spec["bytes"] or file_hash(path)!=spec["sha256"]:
                raise ValueError("Retained evidence bytes differ")
        manifest_ids.append(file_hash(directory/"manifest.json"))
        specs={s["name"]:s for s in plan["models"]}
        for path in sorted(directory.glob("*.hands.jsonl.gz")):
            with gzip.open(path,"rt") as stream:
                for line in stream:
                    row=json.loads(line);spec=specs[row["policy"]]
                    coord=(row["policy"],row["arm"],row["panel"],row["block"],row["rotation"])
                    if coord in seen:raise ValueError("Duplicate paired coordinate")
                    if row["strategy"]!=spec["strategy"]:raise ValueError("Actual base strategy differs")
                    if row["seed"]!=spec["seed"] or row["button"]!=row["block"]%2:
                        raise ValueError("Lineage/position differs")
                    if row["deal_seed"]!=stream_seed(plan["root"],"test","deal",2,row["block"]):
                        raise ValueError("Paired deal differs")
                    hand=Hand.start(Table(("seat0","seat1"),(2000,2000),button=row["button"]),
                                    hand_id=row["hand_id"],seed=row["deal_seed"])
                    coverage=Counter()
                    for index,a in enumerate(row["actions"]):
                        if hand.actor!=a["seat"] or a["index"]!=index or int(hand.actor!=row["rotation"])!=a["logical_player"]:
                            raise ValueError("Actor/action coordinate differs")
                        view=hand.observe(hand.actor);observed=a["observation"]
                        menu=tuple(Choice(c["name"],Action(ActionKind(c["kind"]),c["raise_to"])) for c in observed["menu"])
                        for c in menu:view.legal_actions.validate(c.action)
                        actual=snapshot(view,menu,observed["probabilities"],observed["trained"],None)
                        actual["logical_player"]=a["logical_player"]
                        if json.loads(json.dumps(actual))!=observed:raise ValueError("Observation differs")
                        if not a["logical_player"]:
                            base_menu=choices(view,raise_cap=None,free_fold=False)
                            if information_key(view,base_menu,schema=HU20_UNCAPPED_SCHEMA)!=a["target_key"]:
                                raise ValueError("Base key differs")
                            status=a["average_mass_status"]
                            valid={"missing","current"} if spec["strategy"]=="current" else {"missing","zero_mass","positive_mass"}
                            if status not in valid or bool(a["base_trained"])!=(status!="missing"):
                                raise ValueError("Average/base coverage differs")
                            coverage[status]+=1;coverage[view.street.value+":"+status]+=1
                        if "lbr" in a:
                            lbr["decisions"]+=1;lbr["incomplete_batches"]+=not a["lbr"]["completed"]
                            lbr["soft_overruns"]+=a["lbr"].get("seconds",0)>5
                        hand=hand.apply(Action(ActionKind(a["kind"]),a["raise_to"]))
                    chips=[p.stack-2000 for p in hand.observe(0).players]
                    if (not hand.finished or sum(chips) or chips!=row["net_chips_by_seat"]
                        or chips[row["rotation"]]!=row["target_chips"]
                        or digest(public_events(hand.events))!=row["public_events_sha256"]):
                        raise ValueError("Native settlement differs")
                    if dict(coverage)!=row["coverage"] or hand_tails(row)!=row["tails"]:
                        raise ValueError("Coverage/tails differ")
                    search_counts.update(row["search_counts"])
                    host=host_decisions.setdefault(row["host"],{"seconds":[],"fallback_causes":Counter()})
                    for record in row["search_records"]:
                        if record["status"]=="decision" or record["status"]=="fallback" and record.get("query_kind")=="play":
                            host["seconds"].append(record["seconds"])
                            if record["status"]=="fallback":host["fallback_causes"][record["cause"]]+=1
                    latencies.extend(r["seconds"] for r in row["search_records"]
                        if r["status"]=="decision" or r["status"]=="fallback" and r.get("query_kind")=="play")
                    compact={k:v for k,v in row.items() if k not in ("actions","search_records","search_counts","lbr_zero_likelihood")}
                    compact["actions"]=[{"target_key":a["target_key"],"street":a["street"],
                        **({"lbr":{"completed":a["lbr"]["completed"]}} if "lbr" in a else {})} for a in row["actions"]]
                    rows.append(compact);seen.add(coord)
    expected={(s["name"],arm,p["name"],b,r) for s in plan["models"]
        for arm in (("base","search") if phase=="arena" else (s["strategy"],))
        for p in plan["panels"] for b in range(p["blocks"]) for r in (0,1)}
    if not seen <= expected:raise ValueError("Unexpected frozen arena coordinate")
    if not allow_incomplete and (seen!=expected or len(rows)!=plan["expected_hands"]):raise ValueError("Frozen arena coverage incomplete")
    if len(directories)==1 and not allow_incomplete:
        for k,v in summarize_phase(rows,phase).items():
            if summary[k]!=v:raise ValueError("Paired report arithmetic differs")
    return {"status":"verified","hands":len(rows),"phase":phase,"plan_sha256":digest(plan),
        "manifest_sha256":manifest_ids,"seconds":perf_counter()-started,"owned_rss_bytes":owned_rss(),
        "no_models_loaded":True,"search_counts":dict(search_counts),"decision_seconds":latencies,
        "turn_conditioning_gap_count":sum(v for k,v in search_counts.items() if k.startswith("range:turn_conditioning_fallback:")),
        "turn_conditioning_gap_tolerance":0,
        "turn_conditioning_within_tolerance":not any(v for k,v in search_counts.items()
            if k.startswith("range:turn_conditioning_fallback:")),
        "lbr":dict(lbr),"decision_latency_by_host":latency_report(host_decisions),**({} if allow_incomplete else summarize_phase(rows,phase))}


def latency_report(hosts):
    import numpy as np
    return {host:{"decisions":len(data["seconds"]),
        "p95_seconds":float(np.percentile(data["seconds"],95)) if data["seconds"] else None,
        "p99_seconds":float(np.percentile(data["seconds"],99)) if data["seconds"] else None,
        "max_seconds":max(data["seconds"],default=None),
        "fallback_causes":dict(data["fallback_causes"]),
        "timeout_fallback_rate":data["fallback_causes"]["timeout"]/len(data["seconds"]) if data["seconds"] else None}
        for host,data in hosts.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,action="append",required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("Retain earlier verification")
    atomic_json(a.out,audit(a.directory))


if __name__ == "__main__":main()
