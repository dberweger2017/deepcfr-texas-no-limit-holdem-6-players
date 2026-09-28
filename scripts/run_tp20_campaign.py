"""One heavy child at a time, from outcome-free preflight to fixed confirmation."""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep, time

from scripts.tp20_common import append, system, validate, write_json
from src.arena.schedule import digest
from src.blueprint.windowed import _hash


def swap_bytes(value):
    match = re.search(r"used\s*=\s*([0-9.]+)([MG])", value or "")
    return float(match[1])*1024**(2 if match[2] == "M" else 3) if match else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("preflight", "main"), required=True)
    a = p.parse_args()
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise ValueError("Campaign source must be committed and clean")
    revision = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    plan = json.loads(a.plan.read_text())
    validate(plan, frozen=a.stage == "main")
    if a.stage == "preflight":
        if a.root.exists():
            raise FileExistsError(a.root)
        a.root.mkdir(parents=True)
        now = time()
        record = {"schema":"tp20-campaign-v1", "status":"preflight",
            "started_unix_seconds":now,
            "deadline_unix_seconds":now+plan["limits"]["max_campaign_seconds"],
            "preflight_plan_sha256":digest(plan), "attempts":[],
            "swap_baseline":system(["sysctl","vm.swapusage"])}
    else:
        record = json.loads((a.root/"campaign.json").read_text())
        if record["status"] != "awaiting_plan":
            raise ValueError("Campaign is not waiting for a frozen resource decision")
        if plan.get("resource_preflight_sha256") != _hash(a.root/"preflight"/"result.json"):
            raise ValueError("Frozen plan must pin outcome-free resource preflight")
        record.update(frozen_plan_sha256=digest(plan), source_revision=revision,
                      status="running", training_started_unix_seconds=time())
        record["training_deadline_unix_seconds"] = min(
            time()+plan["limits"]["training_seconds"], record["deadline_unix_seconds"]-7200)
    write_json(a.root/"campaign.json",record)
    phase_deadline = record["deadline_unix_seconds"]-plan["limits"]["report_reserve_seconds"]
    python = sys.executable
    observed = a.root/"independent"/"reached.json"
    args_plan = ["--plan",str(a.plan)]

    def launch(stage, name, command, deadline=phase_deadline):
        attempt = {"stage":stage,"name":name,"command":command,
            "source_revision":revision,"started_unix_seconds":time(),"status":"running",
            "deadline_unix_seconds":deadline}
        record["attempts"].append(attempt)
        write_json(a.root/"campaign.json",record)
        print(f"START {stage} {name}",flush=True)
        next_system = 0
        peak = 0
        with (a.root/f"{stage}-{name}.log").open("w") as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            attempt["pid"] = child.pid
            write_json(a.root/"campaign.json",record)
            try:
                while child.poll() is None:
                    sample = subprocess.run(["ps","-o","rss=","-p",str(child.pid)],
                                            text=True,capture_output=True)
                    current = int(sample.stdout.strip() or 0)*1024
                    peak = max(peak,current)
                    reason = None
                    if current >= plan["limits"]["max_rss_gib"]*1024**3:
                        reason = "Process RSS ceiling"
                    if time() >= deadline:
                        reason = "Absolute phase deadline"
                    if shutil.disk_usage(a.root).free < plan["limits"]["min_free_gib"]*1024**3:
                        reason = "Free-disk ceiling"
                    if monotonic() >= next_system:
                        swap = system(["sysctl","vm.swapusage"])
                        pressure = system(["memory_pressure","-Q"])
                        row = {"unix_seconds":time(),"stage":stage,"name":name,
                            "rss_bytes":current,"swap":swap,"memory_pressure":pressure,
                            "free_disk_bytes":shutil.disk_usage(a.root).free}
                        append(a.root/"resources.jsonl",row)
                        baseline, used = swap_bytes(record["swap_baseline"]), swap_bytes(swap)
                        if baseline is not None and used is not None and (
                                used-baseline > plan["limits"]["max_swap_growth_gib"]*1024**3):
                            reason = "System swap-growth ceiling"
                        free = re.search(r"free percentage:\s*(\d+)%",pressure or "")
                        if free and int(free[1]) < plan["limits"]["min_memory_free_percent"]:
                            reason = "System memory-pressure floor"
                        next_system = monotonic()+30
                    if reason:
                        attempt["error"] = reason
                        child.terminate()
                        try:
                            child.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait()
                        break
                    sleep(1)
                attempt["returncode"] = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                raise
        attempt.update(status="complete" if attempt["returncode"] == 0 and not attempt.get("error")
                       else "failed", finished_unix_seconds=time(), sampled_peak_rss_bytes=peak)
        write_json(a.root/"campaign.json",record)
        print(f"END {stage} {name} {attempt['status']}",flush=True)
        if attempt["status"] != "complete":
            raise RuntimeError(f"{stage}/{name} failed; log and partial artifacts retained")

    def evaluate(phase, lineup, arm, resource_only=False):
        out = a.root/phase/f"{lineup}-{arm}"
        out.parent.mkdir(exist_ok=True)
        command = [python,"-m","scripts.evaluate_tp20",*args_plan,
            "--training-root",str(a.root/("preflight-training" if phase == "preflight" else "training")),
            "--arm",arm,"--lineup",lineup,"--phase",phase,"--out",str(out),
            "--deadline",str(phase_deadline)]
        if resource_only:
            command += ["--resource-only"]
        launch(phase,f"{lineup}-{arm}",command)

    try:
        if a.stage == "preflight":
            evaluate("independent","uniform-observations","uniform",True)
            # Store the immutable decision set outside all checkpoint-dependent trajectories.
            observed.parent.mkdir(exist_ok=True)
            observed.write_bytes((a.root/"independent"/"uniform-observations-uniform"/"reached.json").read_bytes())
            write_json(a.root/"independent"/"manifest.json", {
                "source_schedule_sha256":json.loads((a.root/"independent"/
                    "uniform-observations-uniform"/"manifest.json").read_text())["schedule_sha256"],
                "observations_sha256":_hash(observed), "policy":"same-game untrained uniform",
                "excluded_from_outcome_evaluation":True})
            seed = plan["training_seeds"][0]
            out = a.root/"preflight-training"/str(seed)
            out.parent.mkdir(exist_ok=True)
            launch("preflight-training",str(seed),[python,"-m","scripts.train_tp20",*args_plan,
                "--seed",str(seed),"--out",str(out),"--observations",str(observed),
                "--deadline",str(phase_deadline),"--preflight"])
            for lineup in plan["primary_lineups"]:
                evaluate("preflight",lineup,f"C{seed}",True)
            rows = [json.loads(line) for line in (out/"iterations.jsonl").read_text().splitlines()]
            training = json.loads((out/"result.json").read_text())
            checkpoint = json.loads((out/"checkpoints.jsonl").read_text().splitlines()[-1])
            evaluation = [json.loads((a.root/"preflight"/f"{lineup}-C{seed}"/"result.json").read_text())
                          for lineup in plan["primary_lineups"]]
            (a.root/"preflight").mkdir(exist_ok=True)
            write_json(a.root/"preflight"/"result.json", {
                "schema":"tp20-outcome-free-preflight-v1", "training":training,
                "checkpoint":checkpoint,
                "nodes_per_second":sum(r["nodes"] for r in rows)/sum(r["traversal_seconds"] for r in rows),
                "entries_per_node":training["entries"]/training["completed_nodes"],
                "evaluation_hands_per_second":sum(r["attempts"] for r in evaluation)/sum(
                    r["elapsed_seconds"] for r in evaluation), "evaluation":evaluation,
                "independent_observations_sha256":_hash(observed)})
            record["status"] = "awaiting_plan"
        else:
            training_deadline = record["training_deadline_unix_seconds"]
            for seed in plan["training_seeds"]:
                out = a.root/"training"/str(seed)
                out.parent.mkdir(exist_ok=True)
                launch("training",str(seed),[python,"-m","scripts.train_tp20",*args_plan,
                    "--seed",str(seed),"--out",str(out),"--observations",str(observed),
                    "--deadline",str(training_deadline)],deadline=training_deadline)
            for lineup in plan["primary_lineups"]:
                arms = ["uniform"]+[f"E{s}-{i}" for s in plan["training_seeds"]
                                        for i in range(len(plan["checkpoints"]))]
                for arm in arms:
                    evaluate("development",lineup,arm)
            for phase, lineups in (("confirmation",plan["primary_lineups"]),
                                  ("secondary",plan["secondary_lineups"])):
                for lineup in lineups:
                    for arm in ["uniform"]+[f"C{s}" for s in plan["training_seeds"]]:
                        evaluate(phase,lineup,arm)
            for i,seed in enumerate(plan["training_seeds"]):
                for arm in (f"E{seed}-0",f"C{seed}"):
                    evaluate("crossplay",f"hero-{i}",arm)
            record["status"] = "measurements_complete"
    except Exception as exc:
        record.update(status="failed",error=f"{type(exc).__name__}: {exc}")
    record["updated_unix_seconds"] = time()
    write_json(a.root/"campaign.json",record)
    if a.stage == "main":
        try:
            launch("report","final",[python,"-m","scripts.report_tp20",*args_plan,
                "--root",str(a.root),"--out",str(a.root/"final-report.json")],
                deadline=record["deadline_unix_seconds"])
            if record["status"] == "measurements_complete":
                record["status"] = "complete"
        except Exception as exc:
            record.update(status="failed",report_error=str(exc))
        record["updated_unix_seconds"] = time()
        write_json(a.root/"campaign.json",record)
        # Seal after the report child and its log have stopped changing.
        report_path = a.root/"final-report.json"
        if report_path.exists():
            final_report = json.loads(report_path.read_text())
            final_report["campaign"] = record
            final_report["resources"] = [json.loads(line) for line in
                (a.root/"resources.jsonl").read_text().splitlines()]
            write_json(report_path, final_report)
        inventory = {str(path.relative_to(a.root)):_hash(path) for path in a.root.rglob("*")
                     if path.is_file() and path.name != "artifact-manifest.json"}
        write_json(a.root/"artifact-manifest.json", {"schema":"tp20-artifact-inventory-v1",
            "files":inventory,"files_sha256":digest(inventory)})
        if time() > record["deadline_unix_seconds"]:
            record.update(status="failed", error="Final inventory exceeded the absolute deadline")
            write_json(a.root/"campaign.json",record)
            inventory["campaign.json"] = _hash(a.root/"campaign.json")
            write_json(a.root/"artifact-manifest.json", {"schema":"tp20-artifact-inventory-v1",
                "files":inventory,"files_sha256":digest(inventory)})
    print(json.dumps({"status":record["status"],"error":record.get("error")}),flush=True)
    return 2 if record["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
