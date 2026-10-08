"""M4-only action-translation pilot, frozen final run, replay and reporting."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from time import monotonic, sleep, time

import numpy as np
import psutil

from scripts.evaluate_native_hu100_baseline import OPPONENTS
from scripts.hu20_scaling_supervise import terminate_child
from scripts.native_hu_followup_limits import memory_snapshot
from scripts.report_native_hu100_learning_curves import frozen_schedule, interval
from scripts.run_tp20_campaign import swap_bytes
from src.blueprint.action_translation import TranslationOptions
from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/action-translation"
PILOT_ROOT = 2026100820511
FINAL_ROOT = 2026100820512
PRIOR_ROOTS = [(2026100819711,16),(2026100819712,2048),
               (2026100820311,16),(2026100820312,2048),
               (2026100820411,16),(2026100820412,2048),
               (2026100850411,16),(2026100850412,2048)]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False);f.write("\n")


def source():
    if subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip() != "Apple M4":
        raise ValueError("This campaign is authorized only on the M4")
    if Path.cwd()!=ROOT or subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True).strip():
        raise ValueError("An isolated committed clean source is required")
    return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()


def settings():
    result=read(ROOT/"configs/arena/hu100-playing-baseline-v1.json")
    result.update(pilot_root=PILOT_ROOT,final_root=FINAL_ROOT)
    result["model"]={"name":"HU100-average-39438279","path":str(OUT/"retrieved/research/terminal/average.gz"),
        "sha256":"ba62d13536120a9d549f2f3ff84bcb2a96fbd8143ac2fc8477addab368dee0c4",
        "bytes":193277097,"format":"holdem-hu100-stored-cfr-average-research-v1",
        "actual_nodes":39438279,"entries":7643261,"iteration":30080,
        "source_checkpoint_sha256":"792a675ce6d45d4de8d1b7f3fc6d976548610810f11da8f336c3926f37f8d416"}
    return result


def freshness(config):
    all_config={**config,"models":[config["model"]]}
    previous={}
    for root,blocks in PRIOR_ROOTS+[(PILOT_ROOT,16),(FINAL_ROOT,2048)]:
        document=frozen_schedule(all_config,blocks,root)
        seeds={b["deal_seeds"][0] for p in document["panels"].values() for b in p["blocks"]}
        for old,old_seeds in previous.items():
            if seeds & old_seeds:raise ValueError(f"Physical deal collision: {old}/{root}")
        previous[root]=seeds
    return {"status":"verified","prior_roots":PRIOR_ROOTS,
            "pilot_root":PILOT_ROOT,"final_root":FINAL_ROOT,"all_pairwise_disjoint":True}


def operation(name, module, args, directory, deadline=None):
    """One process family; guards stop it and preserve all partial outputs."""
    guard=directory/(name+"-guard");guard.mkdir()
    command=[sys.executable,"-m",module,*map(str,args)]
    swap0=subprocess.check_output(["sysctl","vm.swapusage"],text=True)
    admission={"memory":memory_snapshot(),"swap":swap0,
               "power":subprocess.check_output(["pmset","-g","batt"],text=True),
               "disk_free":shutil.disk_usage(OUT).free}
    if (admission["memory"]["pressure_level"]!=1 or admission["memory"]["free_percent"]<32
            or "AC Power" not in admission["power"] or admission["disk_free"]<20*1024**3
            or psutil.virtual_memory().used>=10*1024**3):
        write(guard/"admission.json",admission)
        raise RuntimeError("Resource admission refused")
    write(guard/"admission.json",admission)
    started=time();peak=0;failure=None;child=None;count=0
    original={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def interrupt(signum,frame):raise RuntimeError("Supervisor interrupted")
    for s in original:signal.signal(s,interrupt)
    try:
        with (guard/"worker.log").open("x") as log,(guard/"resources.jsonl").open("x") as stream:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            parent=psutil.Process(os.getpid());next_host=0;host={}
            while child.poll() is None:
                now=time()
                family=[parent,*parent.children(recursive=True)]
                rss=0
                for process in family:
                    try:rss+=process.memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss)
                if now>=next_host:
                    host={"swap_growth":swap_bytes(subprocess.check_output(["sysctl","vm.swapusage"],text=True))-swap_bytes(swap0),
                          "power":subprocess.check_output(["pmset","-g","batt"],text=True),
                          "free_disk":shutil.disk_usage(OUT).free,
                          "memory":memory_snapshot(),"used_memory":psutil.virtual_memory().used}
                    next_host=now+5
                stream.write(json.dumps({"at":now,"family_rss":rss,**host})+"\n");stream.flush();count+=1
                if rss>=3*1024**3:raise MemoryError("Whole-family RSS ceiling")
                if host["swap_growth"]>256*1024**2:raise MemoryError("Swap growth ceiling")
                if host["free_disk"]<=20*1024**3:raise OSError("Disk floor")
                if "AC Power" not in host["power"]:raise RuntimeError("AC power guard")
                if host["memory"]["pressure_level"]!=1 or host["used_memory"]>=10*1024**3:
                    raise MemoryError("System memory guard")
                if deadline is not None and now>=deadline:raise TimeoutError("Frozen time budget")
                sleep(.2)
            if child.returncode:raise RuntimeError(f"Worker failed: {name} exit {child.returncode}")
    except BaseException as exc:
        failure=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if child is not None and child.poll() is None:terminate_child(child)
        for s,h in original.items():signal.signal(s,h)
        write(guard/"receipt.json",{"command":command,"started":started,"finished":time(),
            "seconds":time()-started,"peak_family_rss":peak,"samples":count,
            "status":"failed" if failure else "complete","failure":failure,
            "deadline":deadline,"host_sample_seconds":5,"rss_sample_seconds":.2})
    return read(guard/"receipt.json")


def panels(prefix, blocks, root, revision, deadline=None):
    folder=OUT/prefix;folder.mkdir()
    for name in ("disabled","enabled"):
        config=folder/(name+"-config.json")
        write(config,{**settings(),"action_translation":asdict(TranslationOptions()) if name=="enabled" else None})
        args=["--config",config,"--blocks",blocks,"--root",root,"--source",revision]
        operation(name+"-play","scripts.evaluate_native_hu100_baseline",
                  [*args,"--out",folder/name,*(["--reference-run",folder/"disabled"] if name=="enabled" else [])],
                  folder,deadline)
        operation(name+"-audit","scripts.audit_native_hu100_baseline",
                  ["--run",folder/name,"--out",folder/(name+"-audit.json")],folder,deadline)
        operation(name+"-reproduce","scripts.evaluate_native_hu100_baseline",
                  [*args,"--out",folder/(name+"-reproduction"),"--reproduce",folder/name,
                   *(["--reference-run",folder/"disabled-reproduction"] if name=="enabled" else [])],folder,deadline)
    return folder


def quote(folder, revision):
    play=[];audit=[];repeat=[]
    for name in ("disabled","enabled"):
        play.append(read(folder/name/"complete.json"))
        audit.append(read(folder/(name+"-audit.json")))
        repeat.append(read(folder/(name+"-reproduction")/"complete.json"))
    loads=sum(r["model_load_seconds"] for r in play+repeat)
    scalable=sum(max(0,r["wall_seconds"]-r["model_load_seconds"]) for r in play+repeat)+sum(r["seconds"] for r in audit)
    projected=loads+scalable*2048/16
    return {"source":revision,"blocks_per_opponent":2048,"pilot_blocks":16,
            "final_root":FINAL_ROOT,"pilot_root":PILOT_ROOT,"loads_seconds":loads,
            "pilot_scalable_seconds":scalable,"projected_seconds":projected,
            "execution_budget_seconds":math.ceil(3*projected+120),
            "formula":"3*(measured loads + per-block play/replay/reproduction projected to 2048)+120",
            "pilot_outcomes_read":False,"distinct_final_hands":61440,
            "target_hands":40960,"uniform_reference_hands":20480}


def trace_rows(path):
    with gzip.open(path,"rt") as f:
        for line in f:yield json.loads(line)


def summarize_final():
    freeze=read(OUT/"frozen-final.json");blocks=freeze["blocks_per_opponent"]
    result={"source":freeze["source"],"freeze":freeze,"opponents":{},"telemetry":[],
            "all_hands_reproduced":True,"all_hands_replayed":True}
    for op in OPPONENTS:
        values={};control=[]
        for name in ("disabled","enabled"):
            folder=OUT/"final"/name/op;coords={};hands=[]
            for line in (folder/"hands.jsonl").read_text().splitlines():
                row=json.loads(line)
                if row["arm"]=="candidate":
                    coord=(row["block"],row["rotation"])
                    if coord in coords or row["status"]!="completed":raise ValueError("Invalid comparison hand")
                    coords[coord]=row["candidate_chips"];hands.append(row)
            if set(coords)!={(b,r) for b in range(blocks) for r in (0,1)}:raise ValueError("Incomplete block")
            values[name]=[(coords[b,0]+coords[b,1])/2 for b in range(blocks)]
            control.append(hands)
            groups=defaultdict(list)
            for d in trace_rows(folder/"decisions.jsonl.gz"):
                if d["arm"]=="candidate" and d["logical_player"]==0:groups[d["street"]].append(d["translation"])
            for street,rows in groups.items():
                counts=Counter(d["mode"] for d in rows)
                latency=[1000*d["lookup_seconds"] for d in rows]
                hist=Counter()
                for d in rows:
                    if d["mode"]=="translated":
                        distance=d["distance"]
                        band=next((label for upper,label in ((.05,"0-.05"),(.1,".05-.1"),(.25,".1-.25"),(.5,".25-.5"),(1.,".5-1")) if distance<=upper),">1")
                        hist[band]+=1
                result["telemetry"].append({"option":name,"opponent":op,"street":street,"decisions":len(rows),
                    "counts":dict(counts),"rates":{m:counts[m]/len(rows) for m in ("exact","translated","uniform")},
                    "translation_distance_histogram":dict(hist),
                    "all_in_changes":sum(d["all_in_changes"] for d in rows),
                    "bound_reached":sum(d["bound_reached"] for d in rows),
                    "max_states":max(d["states"] for d in rows),
                    "latency_ms":{"mean":float(np.mean(latency)),"p50":float(np.quantile(latency,.5)),
                        "p95":float(np.quantile(latency,.95)),"p99":float(np.quantile(latency,.99)),"max":max(latency)}})
        difference=[a-b for a,b in zip(values["enabled"],values["disabled"],strict=True)]
        cell={**{name:interval(v) for name,v in values.items()},"enabled_minus_disabled":interval(difference),
              "identical_candidate_hands":control[0]==control[1]}
        if op in ("check_call","tight_aggressive","loose_aggressive") and not cell["identical_candidate_hands"]:
            raise ValueError("On-menu descriptive control changed")
        result["opponents"][op]=cell
    result["primary_passed"]=result["opponents"]["pot_pressure"]["enabled_minus_disabled"]["interval"][0]>0
    result["random_safeguard_passed"]=result["opponents"]["random"]["enabled_minus_disabled"]["interval"][0]>-20
    result["resources"]=[read(p) for p in sorted((OUT/"final").glob("*-guard/receipt.json"))]
    write(OUT/"result.json",result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("pilot","final"));a=p.parse_args()
    revision=source()
    if a.stage=="pilot":
        write(OUT/"freshness.json",freshness(settings()))
        folder=panels("pilot",16,PILOT_ROOT,revision)
        write(OUT/"frozen-final.json",quote(folder,revision))
        write(OUT/"frozen-schedule.json",frozen_schedule({**settings(),"models":[settings()["model"]]},2048,FINAL_ROOT))
        print(json.dumps(read(OUT/"frozen-final.json")))
    else:
        frozen=read(OUT/"frozen-final.json")
        if frozen["source"]!=revision:raise ValueError("Pilot and final source differ")
        review=read(OUT/"source-review.json")
        if review["status"]!="passed" or review["source"]!=revision:raise ValueError("Independent exact-source review required")
        admission=read(OUT/"final-admission.json")
        if admission["source"]!=revision or admission["budget_seconds"]!=frozen["execution_budget_seconds"]:
            raise ValueError("Final admission differs from posted freeze")
        started=time();deadline=started+frozen["execution_budget_seconds"]
        write(OUT/"final-intent.json",{"started":started,"deadline":deadline,"source":revision,
              "freeze_sha256":file_hash(OUT/"frozen-final.json"),"schedule_sha256":file_hash(OUT/"frozen-schedule.json")})
        panels("final",2048,FINAL_ROOT,revision,deadline)
        summarize_final()
        if time()>=deadline:raise TimeoutError("Reporting exceeded frozen budget")
        write(OUT/"final-complete.json",{"status":"complete","started":started,"finished":time(),
              "deadline":deadline,"source":revision,"all_hands_replayed_and_reproduced":True})
        print(json.dumps({"status":"complete","seconds":time()-started}))
    return 0


if __name__=="__main__":raise SystemExit(main())

