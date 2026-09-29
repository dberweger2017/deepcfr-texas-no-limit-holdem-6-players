"""Post-worker inventory, fresh-deal audit and independent primary arithmetic."""

import argparse
from collections import defaultdict
import gzip
import json
from math import sqrt
from pathlib import Path
from statistics import fmean
import subprocess
from time import time

from scipy.stats import t

from scripts.hu20_scaling_common import inventory
from scripts.report_hu20_scaling import combine, verify_phase
from scripts.train_hu20 import write_json
from src.arena.schedule import stream_seed
from src.blueprint.solver import _seed
from src.blueprint.windowed import _hash


def independent(paths, seeds, final):
    """Rebuild both primary block differences from raw, native-replayed chips."""
    raw = defaultdict(dict); hashes = {}
    for path in paths:
        hashes[str(path)] = _hash(path)
        with gzip.open(path, "rt") as saved:
            for line in saved:
                r=json.loads(line)
                if r["attacker"] not in ("LBR-original-cap2","Pressure-native") or r["arm"] != "B" or r["milestone"] not in (20000000,final): continue
                key=(r["training_seed"],r["attacker"],r["milestone"],r["block"])
                if r["rotation"] in raw[key]: raise ValueError("Independent duplicate hand")
                if r["status"] != "complete": raise ValueError("Independent failed confirmation hand")
                raw[key][r["rotation"]]=r["target_chips"]
    result={}
    for attack in ("LBR-original-cap2","Pressure-native"):
        blocks=sorted({b for s,a,n,b in raw if a==attack});values=[]
        for b in blocks:
            contrasts=[]
            for s in seeds:
                early=raw[(s,attack,20000000,b)];late=raw[(s,attack,final,b)]
                if set(early)!={0,1} or set(late)!={0,1}:raise ValueError("Independent missing paired role")
                contrasts.append((late[0]+late[1]-early[0]-early[1])/2)
            values.append(sum(contrasts)/len(seeds))
        mu=sum(values)/len(values)
        variance=sum((v-mu)**2 for v in values)/(len(values)-1)
        half=float(t.ppf(.9875,len(values)-1))*sqrt(variance/len(values))
        result[attack]={"blocks":len(values),"bb100":mu,"interval":[mu-half,mu+half]}
    return {"primary":result,"raw_hand_hashes":hashes}


def finish(plan, root):
    if time() >= plan["deadline"]: raise TimeoutError("Reporting must retain original deadline")
    state=json.loads((root/"campaign.json").read_text())
    out=root/"final-audit"
    if out.exists():raise FileExistsError("Retain the existing final audit; do not duplicate")
    out.mkdir()
    if state["status"] != "ready_for_final_report":
        write_json(out/"results.json",{"status":"incomplete","campaign":state,
                   "reason":"Fixed final quality comparison incomplete; no promotion or checkpoint selection"})
        return {"status":"incomplete"}
    audit4=root/"audit-m4-retained.json";audit1=root/"audit-m1/results.json"
    report=combine(plan,[audit1,audit4],out/"results.json")
    if report["native_replayed_hands"] != plan["expected_confirmation_hands"]:
        raise ValueError("Missing frozen confirmation hand count")
    # Retain M4's native hand rows locally for a separate arithmetic path. No run is repeated.
    retained=root/"m4-retained"
    subprocess.run(["rsync","-a","m4:"+plan['hosts']['m4']['source']+"/"+plan['root']+"/",str(retained)+"/"],check=True,timeout=900)
    rawpaths=list((root/"evaluation-m1").glob("*.jsonl.gz"))+list((retained/"evaluation-m4").glob("*.jsonl.gz"))
    independent_result=independent(rawpaths,plan["training_seeds"],plan["training_total_nodes"])
    for attack,r in independent_result["primary"].items():
        ref=report["primary"][attack]["long_minus_20M"]
        if r["blocks"]!=ref["blocks"] or abs(r["bb100"]-ref["bb100"])>1e-8 or any(abs(a-b)>1e-8 for a,b in zip(r["interval"],ref["interval"])):
            raise ValueError("Independent paired arithmetic disagrees")
    write_json(out/"independent-summary.json",independent_result)
    prior=set();prior_files=0
    for dirname in plan["prior_roots"]:
        code=("import gzip,json;from pathlib import Path;values=set();count=0; "
              f"root=Path({dirname!r});\n"
              "for p in root.rglob('*hands.jsonl*'):\n"
              " count+=1;opener=gzip.open if p.suffix=='.gz' else open\n"
              " with opener(p,'rt') as f:\n"
              "  for line in f:\n"
              "   row=json.loads(line)\n"
              "   if 'deal_seed' in row:values.add(row['deal_seed'])\n"
              "print(json.dumps({'files':count,'deals':sorted(values)}))")
        from scripts.run_hu20_scaling_campaign import remote
        # The prior inventory scan is bounded but can exceed a status RPC's 30s timeout.
        host=plan["hosts"]["m4"]
        import shlex
        data=json.loads(subprocess.check_output(["ssh","-o","ConnectTimeout=10","m4",shlex.join([host["python"],"-c",code])],text=True,timeout=240))
        prior.update(data["deals"]);prior_files+=data["files"]
    fresh=set()
    for path in rawpaths:
        with gzip.open(path,"rt") as f:
            for line in f:fresh.add(json.loads(line)["deal_seed"])
    if fresh & prior:raise ValueError("New confirmation reused an opened prior deal")
    training=[];work={}
    independent_deals=set();observations=0
    from scripts.hu20_reopening_common import case_view
    with gzip.open(plan["independent_path"],"rt") as f:
        for line in f:
            row=json.loads(line);case_view(row);independent_deals.add(row["seed"]);observations+=1
    if observations!=4248 or fresh & independent_deals:
        raise ValueError("Changed reused observation fixture or confirmation overlap")
    excluded_deals=fresh|independent_deals
    phase_files=0
    for host,seeds in plan["training_assignment"].items():
        base=root if host=="m1" else retained
        for seed in seeds:
            folder=base/"training"/f"B-{seed}"
            r=json.loads((folder/"result.json").read_text());training.append(r)
            phase_files+=verify_phase(folder)
            if r["initial_nodes"] != plan["parents"][str(seed)]["completed_nodes"] or r["initial_iteration"] != plan["parents"][str(seed)]["iteration"]:
                raise ValueError("Continued work counter reset")
            if r["status"] != "complete" or r["completed_nodes"] < plan["training_total_nodes"]:
                raise ValueError("Lineage dropped or short work target")
            for i in range(1,r["completed_iterations"]+2):
                for seat in (0,1):
                    if _seed(seed,i,seat,0,"deal") in excluded_deals:raise ValueError("Training/confirmation/independent deal overlap")
            counters=defaultdict(int);segments=[];last=plan["parents"][str(seed)]["completed_nodes"];old_entries=r["initial_entries"]
            with (folder/"iterations.jsonl").open() as f:
                for line in f:
                    x=json.loads(line)
                    for field in ("traverser_visits_by_street","new_entries_by_street","revisited_keys_by_street"):
                        for street,n in x[field].items():counters[field+":"+street]+=n
                    for name,n in x["attempted_work"].items():
                        if isinstance(n,(int,float)):counters["attempted:"+name]+=n
            for m in r["milestones"]:
                segments.append({"requested_total_nodes":m["requested_total_nodes"],"completed_nodes":m["completed_nodes"],
                                 "new_keys":m["entries"]-old_entries,"additional_nodes":m["completed_nodes"]-last,
                                 "new_keys_per_additional_million":(m["entries"]-old_entries)*1e6/(m["completed_nodes"]-last)})
                last=m["completed_nodes"];old_entries=m["entries"]
            work[str(seed)]={"counters":dict(counters),"segments":segments}
    preflight_deals={stream_seed(plan["preflight_root"],"validation","deal",2,b) for b in range(8)}
    if fresh & preflight_deals:raise ValueError("Preflight/confirmation deal overlap")
    verification={"status":"verified","native_replayed_hands":report["native_replayed_hands"],
                  "prior_deal_overlap":0,"training_deal_overlap":0,"preflight_deal_overlap":0,
                  "prior_files":prior_files,"prior_unique_deals":len(prior),"fresh_unique_deals":len(fresh),
                  "independent_observations_replayed":observations,"verified_training_phase_files":phase_files,
                  "all_three_continued_lineages":training,"additional_work":work,
                  "finished":time(),"deadline":plan["deadline"],"within_deadline":time()<plan["deadline"]}
    write_json(out/"verification.json",verification)
    # Fixed first lineage, never selected by profit. Keep the old 20M demo intact.
    if time()<plan["deadline"]-600:
        from scripts.play_hu20_native import play
        from scripts.play_hu20 import replay_history
        model=next(s for s in json.loads(Path(plan["coordinator_models"]).read_text())
                   if s["arm"]=="B" and s["seed"]==plan["training_seeds"][0] and s["milestone"]==plan["training_total_nodes"])
        visible=[]
        choose=lambda prompt: next(x.split('.')[0].strip() for x in reversed(visible)
                                  if x.startswith('  ') and ('check' in x or 'call' in x))
        smoke=play(Path(model["path"]),model["sha256"],out/"candidate-human-smoke.jsonl",
                   seed=plan["demo_root"],max_hands=20,input_fn=choose,output=visible.append)
        smoke["replayed"]=replay_history(out/"candidate-human-smoke.jsonl")
        write_json(out/"candidate-human-smoke-result.json",smoke)
        (out/"candidate-human-transcript.txt").write_text('\n'.join(visible)+'\n')
        write_json(out/"candidate-model.json",model)
    else:write_json(out/"candidate-human-smoke-result.json",{"status":"not run","reason":"Original reporting reserve"})
    # Per-phase seals and output current probabilities were verified during native audit.
    # Check the global inventory only after all worker/coordinator logs have stopped.
    files=inventory(root)
    write_json(root.with_name(root.name+"-final-manifest.json"),{"deadline":plan["deadline"],"finished":time(),"files":files})
    return {"status":report["status"],"hands":report["native_replayed_hands"],"quality_gate":report["quality_gate"],"pressure_safeguard":report["pressure_safeguard"]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());print(json.dumps(finish(plan,Path(plan["root"]))));return 0


if __name__ == "__main__":raise SystemExit(main())
