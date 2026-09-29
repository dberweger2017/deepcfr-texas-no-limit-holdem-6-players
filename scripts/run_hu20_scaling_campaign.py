"""One coordinator, immutable deadline, sequential workers and whole-block shards."""

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from time import sleep, time

from scripts.hu20_scaling_common import inventory, specification
from scripts.train_hu20 import write_json
from src.blueprint.windowed import _hash


def remote(plan, code):
    host = plan["hosts"]["m4"]
    command = shlex.join([host["python"], "-c", code])
    return subprocess.check_output(["ssh", "-o", "ConnectTimeout=10", "m4", command], text=True, timeout=30)


def read_host(plan, host, relative):
    base = Path(plan["hosts"][host]["source"])
    path = base / relative
    if host == "m1": return json.loads(path.read_text()) if path.exists() else None
    return json.loads(remote(plan, f"from pathlib import Path; p=Path({str(path)!r}); print(p.read_text() if p.exists() else 'null')"))


def transfer(plan, source_host, relative, target_host, expected):
    """Never publish an incomplete transfer. Failed copies remain explicit attempts."""
    source = Path(plan["hosts"][source_host]["source"]) / relative
    target = Path(plan["hosts"][target_host]["source"]) / relative
    tmp = target.with_name(target.name+".transfer")
    if target_host == "m1":
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", "m4:"+str(source), str(tmp)], check=True, timeout=300)
        if _hash(tmp) != expected: raise ValueError("Transferred artifact hash")
        tmp.replace(target)
    else:
        remote(plan, f"from pathlib import Path; Path({str(target.parent)!r}).mkdir(parents=True,exist_ok=True)")
        subprocess.run(["scp", str(source), "m4:"+str(tmp)], check=True, timeout=300)
        code = ("from pathlib import Path; from hashlib import sha256; "
                f"p=Path({str(tmp)!r}); h=sha256(); "
                "f=p.open('rb'); [h.update(c) for c in iter(lambda:f.read(1048576),b'')]; f.close(); "
                f"assert h.hexdigest()=={expected!r}; p.replace({str(target)!r})")
        remote(plan, code)


def launch(plan, host, name, jobs, deadline):
    settings = plan["hosts"][host]; source = Path(settings["source"])
    root = source / plan["root"]; supervisor = root / (name+"-supervisor")
    jobpath = root / (name+"-jobs.json")
    command = ["/usr/bin/caffeinate", "-dims", settings["python"], "-m", "scripts.hu20_scaling_supervise",
               "--jobs", str(jobpath), "--out", str(supervisor), "--deadline", str(deadline),
               "--swap-baseline", plan["swap_baselines"][host],"--require-ac"]
    if host=="m1": command.extend(["--coordinator-pid",str(os.getpid())])
    code = ("import subprocess,json; from pathlib import Path; "
            f"s=Path({str(supervisor)!r}); assert not s.exists(), 'Duplicate job prevented'; "
            f"Path({str(jobpath)!r}).write_text(json.dumps({jobs!r})); "
            f"log=open({str(root/(name+'-launcher.log'))!r},'w'); "
            f"p=subprocess.Popen({command!r},cwd={str(source)!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)")
    if host == "m4": return int(remote(plan, code).strip())
    return int(subprocess.check_output([sys.executable, "-c", code], text=True).strip())


def wait_workers(plan, phase, record, root):
    remaining = set(plan["hosts"])
    while remaining:
        if time() >= plan["deadline"]: raise TimeoutError("Original absolute deadline")
        for host in list(remaining):
            state = read_host(plan, host, f'{plan["root"]}/{phase}-supervisor/campaign.json')
            if state:
                record.setdefault("worker_status", {})[f"{host}:{phase}"] = state
                write_json(root / "campaign.json", record)
                if state["status"] not in ("running", "preflight"):
                    if state["status"] != "complete": raise RuntimeError(f"{host} {phase} failed; retained worker attempt")
                    remaining.remove(host)
        if remaining: sleep(20)


def stop_owned_workers(plan, record):
    for state in record.get("worker_status", {}).values():
        if state.get("status") != "running": continue
        host = "m4" if state["identity"]["host"] == plan["host_names"]["m4"] else "m1"
        for attempt in state["attempts"]:
            if attempt["status"] != "running" or "pid" not in attempt: continue
            pid = attempt["pid"]
            code = ("import subprocess,os,signal; "
                    f"s=subprocess.run(['ps','-o','command=','-p',{str(pid)!r}],capture_output=True,text=True).stdout; "
                    f"os.killpg({pid},signal.SIGTERM) if 'scripts.' in s and ('hu20_scaling' in s) else None")
            try:
                if host == "m4": remote(plan, code)
                else: subprocess.run([sys.executable,"-c",code],check=True)
            except Exception: pass


def run(plan, plan_path):
    root = Path(plan["root"]); recordpath = root / "campaign.json"
    if recordpath.exists(): raise FileExistsError("Coordinator already owns this campaign; inspect before restart")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Frozen coordinator source must be clean")
    remote_revision = remote(plan, "import subprocess; print(subprocess.check_output(['git','-C',"+repr(plan['hosts']['m4']['source'])+",'rev-parse','HEAD'],text=True).strip())").strip()
    if revision != remote_revision: raise ValueError("Host source pin mismatch")
    record = {"status": "training", "started": plan["started"], "deadline": plan["deadline"],
              "coordinator_pid": os.getpid(),
              "source_revision": revision, "frozen_plan_sha256": _hash(plan_path), "attempts": [], "failure": None}
    write_json(recordpath, record)
    try:
        # The portable M1 was discovered on battery. Never rely on an idle
        # battery-duration estimate for the measured sustained compute path.
        latest_start=plan["training_deadline"]-plan["resource_decision"]["choice"]["training_seconds"]
        while 'AC Power' not in subprocess.check_output(['pmset','-g','batt'],text=True):
            record["status"]="waiting_for_m1_ac"
            record["latest_safe_training_start"]=latest_start;write_json(recordpath,record)
            if time()>=latest_start:
                raise RuntimeError("M1 AC power unavailable before the frozen training reserve; no main work started")
            sleep(20)
        record["status"]="training";record["main_started"]=time();write_json(recordpath,record)
        for host, seeds in plan["training_assignment"].items():
            python = plan["hosts"][host]["python"]
            jobs = [{"name": f"train-{seed}", "command": [python,"-m","scripts.train_hu20_scaling",
                     "--plan",str(plan_path),"--parent",f'{plan["root"]}/inputs/parent-main-{seed}.json',
                     "--out",f'{plan["root"]}/training/B-{seed}',"--deadline",str(plan["training_deadline"])],
                     "deadline": plan["training_deadline"]} for seed in seeds]
            pid = launch(plan,host,"training",jobs,plan["deadline"])
            record["attempts"].append({"host":host,"phase":"training","launcher_pid":pid,"started":time()})
            write_json(recordpath, record)
        wait_workers(plan,"training",record,root)
        all_models = []
        for host, seeds in plan["training_assignment"].items():
            for seed in seeds:
                relative = f'{plan["root"]}/training/B-{seed}'
                r = read_host(plan,host,relative+"/result.json")
                if r["status"] != "complete" or r["parent"]["checkpoint_sha256"] != plan["parents"][str(seed)]["checkpoint_sha256"]:
                    raise ValueError("Training lineage/terminal work failure")
                for m in r["milestones"]:
                    nodes = m["requested_total_nodes"]
                    cp = relative+f"/checkpoint-{nodes}.json.gz"; policy = relative+f"/current-{nodes}.json.gz"
                    other = "m1" if host == "m4" else "m4"
                    for path,h in ((cp,m["checkpoint_sha256"]),(policy,m["policy_sha256"])):
                        transfer(plan,host,path,other,h)
                    all_models.append(specification(seed,nodes,m["iteration"],cp,policy,m["checkpoint_sha256"],m["policy_sha256"]))
        all_models = plan["baseline_models"] + sorted(all_models,key=lambda s:(s["seed"],s["nodes"])) + plan["reference_models"]
        write_json(Path(plan["coordinator_models"]),all_models)
        transfer(plan,"m1",plan["coordinator_models"],"m4",_hash(Path(plan["coordinator_models"])))
        record["status"] = "evaluation"; write_json(recordpath,record)
        for host in plan["hosts"]:
            python=plan["hosts"][host]["python"]
            jobs=[{"name":"evaluation","command":[python,"-m","scripts.evaluate_hu20_scaling","--plan",str(plan_path),
                   "--models",plan["coordinator_models"],"--host",host,"--out",f'{plan["root"]}/evaluation-{host}',
                   "--deadline",str(plan["evaluation_deadline"]),"--swap-baseline",plan["swap_baselines"][host]],"deadline":plan["evaluation_deadline"]},
                  {"name":"audit","command":[python,"-m","scripts.report_hu20_scaling","--plan",str(plan_path),
                   "--evaluation",f'{plan["root"]}/evaluation-{host}',"--out",f'{plan["root"]}/audit-{host}'],"deadline":plan["deadline"]-900}]
            pid=launch(plan,host,"evaluation",jobs,plan["deadline"])
            record["attempts"].append({"host":host,"phase":"evaluation+audit","launcher_pid":pid,"started":time()});write_json(recordpath,record)
        wait_workers(plan,"evaluation",record,root)
        relative=f'{plan["root"]}/audit-m4/results.json'
        value=read_host(plan,"m4",relative);write_json(root/"audit-m4-retained.json",value)
        record["status"]="ready_for_final_report"
    except Exception as exc:
        record.update(status="incomplete",failure=f"{type(exc).__name__}: {exc}")
        stop_owned_workers(plan,record)
    record["finished"]=time();write_json(recordpath,record)
    write_json(root.with_name(root.name+"-coordinator-inventory.json"),inventory(root))
    print(json.dumps({k:v for k,v in record.items() if k != "worker_status"}));return record


def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);a=p.parse_args()
    r=run(json.loads(a.plan.read_text()),a.plan);return r["status"] != "ready_for_final_report"


if __name__ == "__main__": raise SystemExit(main())
