"""M4-only HU100 compact-table campaign operations and complete resource guards."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from time import monotonic, sleep, time

import psutil
from scripts.hu20_scaling_supervise import terminate_child
from scripts.run_tp20_campaign import swap_bytes
from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/hu100-1b"
BINARY = ROOT / "native/hu20-trainer/target/release/hu20-trainer"
CHECKPOINT_SHA = "792a675ce6d45d4de8d1b7f3fc6d976548610810f11da8f336c3926f37f8d416"
AVERAGE_SHA = "ba62d13536120a9d549f2f3ff84bcb2a96fbd8143ac2fc8477addab368dee0c4"
GIB = 1024**3
FAMILY_SOFT = 6*GIB
FAMILY_HARD = 8*GIB
DISK_FLOOR = int(15.5*GIB)
SWAP_GROWTH = 512*1024**2

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

def host():
    free_raw = subprocess.check_output(["memory_pressure", "-Q"], text=True, timeout=10)
    free = re.search(r"System-wide memory free percentage:\s*(\d+)%", free_raw)
    if free is None:
        raise ValueError("Unreadable free memory")
    return {
        "pressure_level": int(subprocess.check_output(["sysctl","-n","kern.memorystatus_vm_pressure_level"],text=True,timeout=10)),
        "free_percent": int(free[1]),
        "swap_bytes": swap_bytes(subprocess.check_output(["sysctl","vm.swapusage"],text=True,timeout=10)),
        "ac": "AC Power" in subprocess.check_output(["pmset","-g","batt"],text=True,timeout=10),
        "disk_free_bytes": shutil.disk_usage(OUT).free,
        "system_used_bytes": psutil.virtual_memory().used,
    }

def identity():
    if Path.cwd() != ROOT:
        raise ValueError("Campaign checkout required")
    if subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip() != "Apple M4":
        raise ValueError("Only the free M4 is authorized")
    if int(subprocess.check_output(["sysctl","-n","hw.memsize"],text=True)) != 16*GIB:
        raise ValueError("Expected 16 GiB M4")
    return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()

def limits(sample, swap0, rss):
    if rss >= FAMILY_HARD:
        return "hard whole-family RSS"
    if sample["pressure_level"] != 1 or sample["free_percent"] < 15:
        return "system pressure/headroom"
    if sample["swap_bytes"]-swap0 > SWAP_GROWTH:
        return "swap growth"
    if sample["disk_free_bytes"] <= DISK_FLOOR:
        return "disk floor"
    if not sample["ac"]:
        return "AC power"
    return None

def memory_capacity():
    # Forecast includes the supervisor and save workspace; leave 2 GiB between
    # the soft complete-iteration stop and hard family ceiling.
    return math.floor((FAMILY_SOFT-100_000_000)/110)

def operation(name, command, *, stop_file=None, accepted=(0,), deadline=None):
    OUT.mkdir(parents=True, exist_ok=True)
    guard = OUT / "operations" / name
    guard.mkdir(parents=True)
    admission = host()
    baseline = OUT / "baseline.json"
    if not baseline.exists():
        write(baseline, {"at":time(), "host":admission, "source":identity(),
              "soft_family_bytes":FAMILY_SOFT, "hard_family_bytes":FAMILY_HARD,
              "disk_floor_bytes":DISK_FLOOR, "swap_growth_bytes":SWAP_GROWTH})
    swap0 = read(baseline)["host"]["swap_bytes"]
    failure = limits(admission, swap0, 0)
    if failure or admission["free_percent"]*16*GIB/100 < FAMILY_SOFT+2*GIB:
        write(guard/"admission.json",admission)
        raise RuntimeError("Resource admission refused: "+str(failure or "8 GiB free required"))
    write(guard/"admission.json",admission)
    write(guard/"intent.json",{"command":list(map(str,command)),"started":time(),"source":identity(),
          "binary_sha256":file_hash(BINARY),"deadline":deadline})
    started=time(); peak=0; count=0; child=None; failure=None; soft=False
    original={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def interrupt(signum,frame):
        raise RuntimeError("Supervisor interrupted")
    for s in original:
        signal.signal(s,interrupt)
    try:
        with (guard/"worker.log").open("x") as log, (guard/"resources.jsonl").open("x") as stream:
            child=subprocess.Popen(["/usr/bin/time","-l",*map(str,command)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            parent=psutil.Process(os.getpid()); tick=monotonic()
            while child.poll() is None:
                rss=0
                for p in [parent,*parent.children(recursive=True)]:
                    try:
                        rss+=p.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak=max(peak,rss)
                sample=host(); now=time()
                stream.write(json.dumps({"at":now,"family_rss_bytes":rss,"swap_growth_bytes":sample["swap_bytes"]-swap0,**sample})+"\n")
                stream.flush(); count+=1
                violation=limits(sample,swap0,rss)
                if violation:
                    raise RuntimeError("Guard breach: "+violation)
                if rss >= FAMILY_SOFT:
                    if stop_file is None:
                        raise RuntimeError("Guard breach: tool family exceeds soft ceiling")
                    if not soft:
                        write(stop_file,{"reason":"soft family RSS", "at":now,"rss_bytes":rss})
                        soft=True
                if deadline is not None and now >= deadline:
                    raise TimeoutError("Measured frozen budget exhausted")
                tick+=.2
                sleep(max(0,tick-monotonic()))
            if child.returncode not in accepted:
                raise RuntimeError(f"Worker {name} exited {child.returncode}")
    except BaseException as exc:
        failure=repr(exc)
        raise
    finally:
        if child is not None and child.poll() is None:
            terminate_child(child)
        for s,h in original.items():
            signal.signal(s,h)
        log_path=guard/"worker.log"
        high=re.search(r"(\d+)\s+maximum resident set size",log_path.read_text()) if log_path.exists() else None
        write(guard/"receipt.json",{"name":name,"command":list(map(str,command)),"started":started,"finished":time(),
              "seconds":time()-started,"peak_family_rss_bytes":peak,"kernel_command_peak_rss_bytes":int(high[1]) if high else None,
              "samples":count,"sampling_target_seconds":.2,"soft_stop_requested":soft,
              "status":"failed" if failure else "complete","failure":failure,
              "returncode":child.returncode if child else None,"deadline":deadline})
    return read(guard/"receipt.json")

def export_audit(folder, name, nodes):
    operation(name+"-export",[BINARY,"export",folder/"checkpoint.gz","--current",folder/"current.gz",
                             "--average",folder/"average.gz","--zero-mass","uniform"])
    operation(name+"-audit",[sys.executable,"-m","scripts.audit_native_hu_checkpoint",
              "--checkpoint",folder/"checkpoint.gz","--current",folder/"current.gz",
              "--average",folder/"average.gz","--stack-bb","100","--target-nodes",nodes,"--out",folder/"audit.json"])

def gate():
    folder=OUT/"gate"; folder.mkdir(parents=True)
    operation("gate-train",[BINARY,"train","--stack-bb","100","--seed","2026100601",
              "--roots-per-seat","1","--average-rule","opponent-sampled","--nodes","1000000000",
              "--max-entries","7642767","--out",folder/"checkpoint.gz","--telemetry",folder/"telemetry.jsonl",
              "--stop-file",folder/"stop.json"],stop_file=folder/"stop.json",accepted=(3,))
    if file_hash(folder/"checkpoint.gz") != CHECKPOINT_SHA:
        write(folder/"mismatch.json",{"kind":"checkpoint","actual":file_hash(folder/"checkpoint.gz"),"expected":CHECKPOINT_SHA})
        raise ValueError("STOP: exact checkpoint gate mismatch")
    operation("gate-export",[BINARY,"export",folder/"checkpoint.gz","--average",folder/"average.gz",
                             "--current",folder/"current.gz","--zero-mass","uniform"])
    if file_hash(folder/"average.gz") != AVERAGE_SHA:
        write(folder/"mismatch.json",{"kind":"average","actual":file_hash(folder/"average.gz"),"expected":AVERAGE_SHA})
        raise ValueError("STOP: exact parent average gate mismatch")
    write(folder/"gate.json",{"status":"passed","source":identity(),"checkpoint_sha256":CHECKPOINT_SHA,
          "average_sha256":AVERAGE_SHA,"binary_sha256":file_hash(BINARY)})
    operation("gate-audit",[sys.executable,"-m","scripts.audit_native_hu_checkpoint",
              "--checkpoint",folder/"checkpoint.gz","--current",folder/"current.gz",
              "--average",folder/"average.gz","--stack-bb","100","--target-nodes","39438279","--out",folder/"audit.json"])


PILOT_ROOT = 2026100810011
FINAL_ROOT = 2026100810012
BLOCKS = 2048

def gate_spec():
    from scripts.native_hu100_model_metadata import audited_average_spec
    folder=OUT/"gate"
    if read(folder/"gate.json")["status"] != "passed":
        raise ValueError("Both gates required")
    return audited_average_spec(folder/"average.gz",read(folder/"audit.json"),
                                checkpoint_sha256=CHECKPOINT_SHA,actual_nodes=39438279)

def config(spec, translated=False):
    from dataclasses import asdict
    from src.blueprint.average import TranslationOptions
    settings=read(ROOT/"configs/arena/hu100-playing-baseline-v1.json")
    settings.update(model=spec,pilot_root=PILOT_ROOT,final_root=FINAL_ROOT,
                    action_translation=asdict(TranslationOptions()) if translated else None)
    return settings

def freshness():
    from scripts.report_native_hu100_learning_curves import frozen_schedule
    from scripts.run_hu100_action_translation import PRIOR_ROOTS
    settings=config(gate_spec())
    settings["models"]=[settings["model"]]
    old=read(ROOT/"configs/arena/hu100-learning-curves-v1.json")
    roots=[(old,old["pilot_root"],16),(old,old["final_root"],2048)]
    roots += [(settings,r,b) for r,b in [*PRIOR_ROOTS,(2026100820521,16),(2026100820512,2048),
                                       (PILOT_ROOT,16),(FINAL_ROOT,BLOCKS)]]
    seen={}
    for cfg,root,blocks in roots:
        doc=frozen_schedule(cfg,blocks,root)
        seeds={b["deal_seeds"][0] for p in doc["panels"].values() for b in p["blocks"]}
        for earlier,previous in seen.items():
            if seeds & previous:
                raise ValueError(f"Physical deal collision {earlier}/{root}")
        seen[root]=seeds
    return {"status":"verified","roots":[{"root":r,"blocks":b} for _,r,b in roots],
            "all_pairwise_disjoint":True}

def panel(prefix, label, spec, blocks, root, revision, *, translated=False,
          reference=None, repeat_reference=None, deadline=None):
    folder=OUT/prefix
    folder.mkdir(parents=True,exist_ok=True)
    cfg=folder/(label+"-config.json")
    write(cfg,config(spec,translated))
    args=["--config",cfg,"--source",revision,"--blocks",blocks,"--root",root]
    operation(prefix+"-"+label+"-play",[sys.executable,"-m","scripts.evaluate_native_hu100_baseline",
              *args,"--out",folder/label,*(["--reference-run",reference] if reference else [])],deadline=deadline)
    operation(prefix+"-"+label+"-audit",[sys.executable,"-m","scripts.audit_native_hu100_baseline",
              "--run",folder/label,"--out",folder/(label+"-audit.json")],deadline=deadline)
    operation(prefix+"-"+label+"-reproduce",[sys.executable,"-m","scripts.evaluate_native_hu100_baseline",
              *args,"--out",OUT/(prefix+"-reproduction")/label,"--reproduce",folder/label,
              *(["--reference-run",repeat_reference] if repeat_reference else [])],deadline=deadline)

def pilot():
    revision=identity()
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],text=True).strip():
        raise ValueError("Clean committed source required before play")
    write(OUT/"freshness.json",freshness())
    spec=gate_spec()
    for translated,label in ((False,"off"),(True,"on")):
        panel("pilot",label,spec,16,PILOT_ROOT,revision,translated=translated,
              reference=OUT/"pilot/off" if translated else None,
              repeat_reference=OUT/"pilot-reproduction/off" if translated else None)
    write(OUT/"pilot-complete.json",{"source":revision,"outcomes_inspected":False,
          "pilot_blocks_per_opponent":16,"completed":time()})
    prepare()

def prepare():
    pilot=read(OUT/"pilot-complete.json")
    gate=read(OUT/"gate/audit.json")
    telemetry=read_telemetry(OUT/"gate/telemetry.jsonl")[0]
    entries=gate["entries"]
    forecast_entries=[entries,13291612,
                      math.ceil(19242803*(250_000_000/200_000_000)**.53),
                      math.ceil(19242803*(500_000_000/200_000_000)**.53),
                      math.ceil(19242803*(1_000_000_000/200_000_000)**.53)]
    assets=gate["files"]
    combined_bpe=sum(v["bytes"] for v in assets.values())/entries
    # Gate-compressed bytes are measured, not a fixed elapsed cutoff.
    # Reserve 40 B/entry checkpoints, observed export B/entry, and 10% growth.
    models_bytes=math.ceil((40+sum(v["bytes"] for p,v in assets.items()
                                  if Path(p).name!="checkpoint.gz")/entries)*sum(forecast_entries)*1.10)
    raw_bytes=sum(p.stat().st_size for root in ("pilot","pilot-reproduction")
                  for p in (OUT/root).rglob("*") if p.is_file())
    raw_forecast=math.ceil(raw_bytes*BLOCKS/16*3)
    disk=host()["disk_free_bytes"]
    # Account for already retained gate bytes once: archive requires their
    # second copy; subsequent model sets require both originals and ZIP bytes.
    gate_bytes=sum(v["bytes"] for v in assets.values())
    required_disk=2*models_bytes-gate_bytes+2*raw_forecast+512*1024**2
    op=lambda name:read(OUT/"operations"/name/"receipt.json")
    training_seconds=max(.001,telemetry["elapsed_seconds_including_writes"]-telemetry["write_seconds"])
    train_quote=3*(1_000_000_000/telemetry["completed_nodes"])*training_seconds
    save_quote=2*telemetry["write_seconds"]*sum(forecast_entries[1:])/entries
    tool_quote=2*(op("gate-export")["seconds"]+op("gate-audit")["seconds"])*sum(forecast_entries[1:])/entries
    loads=0;scalable=0
    for label in ("off","on"):
        for root in ("pilot","pilot-reproduction"):
            complete=read(OUT/root/label/"complete.json")
            loads+=complete["model_load_seconds"]
            scalable+=max(0,complete["wall_seconds"]-complete["model_load_seconds"])
        scalable+=read(OUT/"pilot"/(label+"-audit.json"))["seconds"]
    # Six policy arms (five untranslated checkpoints plus terminal translated)
    # versus the two parent pilot arms; model reads scale by forecast entries.
    model_scale=(sum(forecast_entries)+forecast_entries[-1])/(2*entries)
    play_quote=3*(loads*model_scale+scalable*BLOCKS/16*3)
    total=train_quote+save_quote+tool_quote+play_quote+600
    receipt={"status":"admitted" if disk-required_disk>=DISK_FLOOR else "storage-refused",
             "source":pilot["source"],"blocks_per_opponent":BLOCKS,"pilot_root":PILOT_ROOT,
             "final_root":FINAL_ROOT,"pilot_outcomes_inspected":False,
             "memory_entry_ceiling":memory_capacity(),"memory_formula":"110 B/entry + 100 MB",
             "family_soft_bytes":FAMILY_SOFT,"family_hard_bytes":FAMILY_HARD,
             "forecast_entries_parent_100m_250m_500m_1b":forecast_entries,
             "gate_combined_asset_bytes_per_entry":combined_bpe,"retained_models_forecast_bytes":models_bytes,
             "pilot_raw_bytes":raw_bytes,"retained_final_raw_forecast_bytes":raw_forecast,
             "additional_disk_required_bytes":required_disk,"disk_free_bytes":disk,
             "disk_floor_bytes":DISK_FLOOR,"disk_shortfall_bytes":max(0,required_disk+DISK_FLOOR-disk),
             "cost_seconds":{"training":train_quote,"saves":save_quote,"exports_and_audits":tool_quote,
                             "play_replay_reproduction":play_quote,"report_archive_reserve":600,"total":total},
             "measured_pilot_loads_seconds":loads,"measured_pilot_scalable_seconds":scalable,
             "budget_formula":"3x slower-node training; 2x save/tool entry scaling; 3x measured play/load scaling; 600s closeout",
             "limits":"Entry and storage projections are extrapolations; measured guards remain authoritative."}
    write(OUT/"preflight.json",receipt)
    print(json.dumps(receipt,indent=2))

def read_telemetry(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("gate","pilot"))
    a=p.parse_args()
    identity()
    (gate if a.stage=="gate" else pilot)()

if __name__=="__main__":
    main()

