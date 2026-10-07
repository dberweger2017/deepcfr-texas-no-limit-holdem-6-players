"""Frozen paired-board readout for global equity witnesses."""

import argparse
from collections import defaultdict
import json
import gzip
from pathlib import Path
import numpy as np

from src.diagnostics.flop_check import atomic_json, line_key
from src.diagnostics.saved_hu20 import file_hash
from scripts.run_global_bucket_validation import prepared_compact

METRICS = ("e_cross_v1", "e_cross_eq50", "e_global50", "e_global200", "e_root_v1", "e_bp")
LABELS = ("v1 witness", "Fitted equity-50 witness", "Global K=50 witness", "Global K=200 witness", "Per-root v1", "Blueprint")


def summarize(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row["spot"]].append(row)
    spots=sorted(grouped)
    matrix=np.asarray([[np.mean([r[m] for r in grouped[s]]) for m in METRICS] for s in spots])
    weights=np.asarray([grouped[s][0]["board_weight"] for s in spots])
    if not np.isfinite(matrix).all() or not np.isfinite(weights).all() or (weights<=0).any():
        raise ValueError("Invalid weighted loss matrix")
    point=np.average(matrix,axis=0,weights=weights)
    draw=np.random.default_rng(202610030304).integers(0,len(spots),(2000,len(spots)))
    boot=(matrix[draw]*weights[draw,None]).sum(axis=1)/weights[draw].sum(axis=1)[:,None]
    result={m:{"mean":float(point[i]),"ci95":np.quantile(boot[:,i],[.025,.975]).tolist()} for i,m in enumerate(METRICS)}
    result["global200_minus_global50"]={"mean":float(point[3]-point[2]),"ci95":np.quantile(boot[:,3]-boot[:,2],[.025,.975]).tolist()}
    result["global50_minus_fitted50"]={"mean":float(point[2]-point[1]),"ci95":np.quantile(boot[:,2]-boot[:,1],[.025,.975]).tolist()}
    return result


def raw_rows(root):
    manifest=json.loads((root/"prepared/manifest.json").read_text())
    jobs=manifest["jobs"]
    if len(jobs)!=120 or len({j["spot"] for j in jobs})!=40:
        raise ValueError("Frozen 120-job/40-board schedule differs")
    result=[];inventory={}
    for job in jobs:
        paths=[root/"baseline/main-06"/phase/job["job"]/"result.json" for phase in ("collect","relock")]
        paths += [root/"run"/phase/job["job"]/"result.json" for phase in ("collect","relock")]
        records=[]
        for path in paths:
            inventory[str(path.relative_to(root))]=file_hash(path)
            records.append(json.loads(path.read_text()))
        old_collect,old_lock,new_collect,new_lock=records
        if any(not r["reference_gate"]["passed"] for r in (new_collect,new_lock)):
            raise ValueError("Unqualified global record")
        for phase,record in (("collect",new_collect),("relock",new_lock)):
            response=Path(record["response"]["zip_path"])
            if file_hash(response)!=record["response"]["zip_sha256"]:
                raise ValueError("Raw native response hash differs")
        if job["replay_sample"] and (len(new_lock.get("replay_gates", []))!=2 or not all(g["passed"] for g in new_lock["replay_gates"])):
            raise ValueError("Scheduled global statistics/BR replay gate absent")
        if new_lock["rows"][-1]["iterations"]!=0:
            raise ValueError("Locked evaluation performed CFR iterations")
        metrics=old_collect["metrics"]+old_lock["metrics"]+[r for r in new_lock["rows"] if r["event"]=="pooling_metric"]
        for seat in (0,1):
            selected={m["metric"]:m for m in metrics if m["target_solver_seat"]==seat}
            if not set(METRICS)<=selected.keys():raise ValueError("Missing required seat measurement")
            result.append({"spot":job["spot"],"lineage":job["lineage"],"seat":seat,"board_weight":job["board_weight"],
                "evaluation_fold":job["evaluation_fold"], **{m:selected[m]["gain_bb"] for m in METRICS},
                "coverage":{m:selected[m]["fallback_coverage"] for m in ("e_global50","e_global200")}})
    return result,inventory


def coverage(rows):
    totals=defaultdict(lambda:[0.,0.])
    for row in rows:
        for metric,streets in row["coverage"].items():
            for street,(total,missing,_) in streets.items():
                if not np.isfinite([total,missing]).all() or not 0<=missing<=total+1e-5:
                    raise ValueError("Invalid missing decision reach")
                cell=totals[metric,row["evaluation_fold"],row["lineage"],street]
                cell[0]+=row["board_weight"]*total;cell[1]+=row["board_weight"]*missing
    if len(totals)!=24:raise ValueError("Missing K/fold/lineage/street coverage cell")
    return [{"metric":m,"fold":f,"lineage":l,"street":s,"total_reach":t,"missing_reach":missing,
             "fraction":missing/t if t else None} for (m,f,l,s),(t,missing) in sorted(totals.items())]


def key_counts(root):
    jobs=json.loads((root/"prepared/manifest.json").read_text())["jobs"]
    lookup={};available=defaultdict(set);labels=defaultdict(set)
    seen=set()
    for job in jobs:
        if job["spot"] in seen:continue
        seen.add(job["spot"])
        request=json.loads(Path(job["request"]).read_text())
        compact=prepared_compact(job)
        by_table={}
        for node in request["nodes"]:
            if node["terminal"]:continue
            table=compact["node_tables"][line_key(node["line"])]
            if table in by_table and by_table[table]!=node["street"]:
                raise ValueError("Pool table aliases two streets")
            by_table[table]=node["street"]
        for k,alias in ((50,"eq50-fit0"),(200,"eq50-fit1")):
            for table,mapping in compact["pool_keys"][alias].items():
                street=by_table[table]
                for label,key in mapping.items():
                    identity=(alias,key)
                    if identity in lookup and lookup[identity]!=street:raise ValueError("Global key aliases streets")
                    lookup[identity]=street
                    available[k,job["evaluation_fold"],street].add(key)
                    labels[k,street].add(int(label))
    fitted=[]
    for path in sorted((root/"run").glob("crossfit-*.json")):
        policy=json.loads(path.read_text());counts=defaultdict(set);positive=defaultdict(set)
        for group in policy["groups"]:
            street=lookup[group["metric"],group["key"]]
            counts[group["metric"],street].add(group["key"])
            if group["mass"]>0:positive[group["metric"],street].add(group["key"])
        for (alias,street),keys in sorted(counts.items()):
            fitted.append({"policy":path.name,"alias":alias,"street":street,"keys":len(keys),
                           "positive_training_mass_keys":len(positive[alias,street])})
    return {"occupied_labels":{f"K{k}/{street}":len(v) for (k,street),v in sorted(labels.items())},
            "available_keys_by_evaluation_half":{f"K{k}/half{fold}/{street}":len(v) for (k,fold,street),v in sorted(available.items())},
            "fitted_policy_keys":fitted}


def report(root,out):
    rows,inventory=raw_rows(root)
    groups={"pooled":summarize(rows)}
    for lineage in sorted({r["lineage"] for r in rows}):groups[str(lineage)]=summarize([r for r in rows if r["lineage"]==lineage])
    for seat in (0,1):groups[f"seat-{seat}"]=summarize([r for r in rows if r["seat"]==seat])
    cells=coverage(rows);coverage_pass=all(c["fraction"] is not None and c["fraction"]<=.05 for c in cells)
    primary=groups["pooled"]["e_global50"]
    near=abs(primary["mean"]-.3874)<=.10
    separated=primary["ci95"][1]<.6273
    summary={"key_counts":key_counts(root),"groups":groups,"rows":rows,"input_inventory":inventory,"coverage":cells,
             "pass":near and separated and coverage_pass,"criteria":{"within_0_10_bb":near,"upper_below_v1_lower":separated,"coverage":coverage_pass},
             "seed":202610030304,"draws":2000,"boards":40,"seat_lineage_rows":240,
             "interval_scope":"conditional on fixed opposite-half fitted witnesses; no refitting uncertainty"}
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/"summary.json",summary)
    text=["# Global full-deck equity witness validation", "", "| Witness | E BB [95% interval] |", "|---|---:|"]
    for metric,label in zip(METRICS,LABELS,strict=True):
        value=groups["pooled"][metric];a,b=value["ci95"]
        text.append(f"| {label} | {value['mean']:.4f} [{a:.4f}, {b:.4f}] |")
    text += ["",f"Primary criterion: **{'PASS' if summary['pass'] else 'FAIL'}**.","",summary["interval_scope"]+"."]
    (out/"report.md").write_text("\n".join(text)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();report(a.root.resolve(),a.out.resolve())


if __name__=="__main__":main()
