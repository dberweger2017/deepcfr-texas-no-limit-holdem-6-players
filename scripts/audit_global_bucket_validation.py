"""Independent scalar recomputation from raw solver values and paired draws."""

import argparse
from collections import Counter
import gzip
import json
import zipfile
from math import fsum
from pathlib import Path
import numpy as np


def audit(root,summary_path,out):
    summary=json.loads(summary_path.read_text());manifest=json.loads((root/"prepared/manifest.json").read_text())
    rows=[];checked=0
    names=("e_cross_v1","e_cross_eq50","e_global50","e_global200","e_root_v1","e_bp")
    for job in manifest["jobs"]:
        # Re-open retained responses; do not trust the reporter's derived rows.
        raw=[]
        for phase in ("collect","relock"):
            raw+=json.loads((root/"baseline/main-06"/phase/job["job"]/"result.json").read_text())["metrics"]
        locator=json.loads((root/"run/relock"/job["job"]/"result.json").read_text())["response"]
        with zipfile.ZipFile(locator["zip_path"]) as archive,archive.open("response.jsonl") as source:
            native=[json.loads(line) for line in source]
        final=native[-1]
        if final["iterations"]!=0 or final["status"]!="locked-evaluated":raise ValueError("Not a zero-CFR lock")
        raw += [r for r in native if r.get("event")=="pooling_metric"]
        for seat in (0,1):
            by_name={r["metric"]:r for r in raw if r["target_solver_seat"]==seat}
            derived={}
            for name in names:
                metric=by_name[name]
                # Native computes the BR-reference difference in f32, then scales in f64.
                delta=float(np.float32(np.float32(metric["responder_br_chips"])-np.float32(metric["reference_responder_value_chips"])))
                value=delta/100
                if abs(value-metric["gain_bb"])>1e-12 or abs(delta/2-metric["gain_pct_pot"])>1e-10:
                    raise ValueError("Native metric arithmetic does not reconcile")
                if name.startswith("e_global") and metric["reference_responder_value_chips"]!=final["reference_equilibrium_ev_chips"][seat^1]:
                    raise ValueError("Wrong opposite-seat equilibrium reference")
                derived[name]=value;checked+=1
            rows.append(dict(derived,spot=job["spot"],lineage=job["lineage"],seat=seat,weight=job["board_weight"]))
    differences=[]
    for group,expected in summary["groups"].items():
        chosen=[r for r in rows if group=="pooled" or group==str(r["lineage"]) or group==f"seat-{r['seat']}"]
        spots=sorted({r["spot"] for r in chosen});weights={s:next(r["weight"] for r in chosen if r["spot"]==s) for s in spots}
        means={s:{m:fsum(r[m] for r in chosen if r["spot"]==s)/sum(r["spot"]==s for r in chosen) for m in names} for s in spots}
        draws=np.random.default_rng(202610030304).integers(0,len(spots),(2000,len(spots)))
        points={m:fsum(weights[s]*means[s][m] for s in spots)/fsum(weights.values()) for m in names}
        values={m:[] for m in names}
        for draw in draws:
            counts=Counter(int(i) for i in draw);denom=fsum(n*weights[spots[i]] for i,n in counts.items())
            for m in names:values[m].append(fsum(n*weights[spots[i]]*means[spots[i]][m] for i,n in counts.items())/denom)
        for m in names:
            ci=np.quantile(values[m],[.025,.975])
            differences.extend([abs(points[m]-expected[m]["mean"]),*(abs(float(ci[i])-expected[m]["ci95"][i]) for i in (0,1))])
        for label,a,b in (("global200_minus_global50","e_global200","e_global50"),("global50_minus_fitted50","e_global50","e_cross_eq50")):
            contrasts=[x-y for x,y in zip(values[a],values[b],strict=True)]
            ci=np.quantile(contrasts,[.025,.975]);differences.extend([abs(points[a]-points[b]-expected[label]["mean"]),*(abs(float(ci[i])-expected[label]["ci95"][i]) for i in (0,1))])
    maximum=max(differences)
    if maximum>1e-12:raise ValueError("Independent paired-board arithmetic differs")
    out.write_text(json.dumps({"passed":True,"native_seat_metrics_checked":checked,"boards":40,"draws":2000,
        "seed":202610030304,"maximum_absolute_summary_difference":maximum,
        "method":"native f32 chip subtraction; independent fsum scalar weighting with bootstrap multiplicities"},indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","summary","out"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();audit(a.root.resolve(),a.summary.resolve(),a.out.resolve())


if __name__=="__main__":main()
