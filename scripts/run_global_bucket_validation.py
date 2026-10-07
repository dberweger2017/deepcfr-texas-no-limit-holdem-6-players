"""Sequential M1 witness construction using unchanged external solver bytes."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import io
import zipfile
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from time import sleep, time

from src.diagnostics.board_pooling import pool_statistics
from src.diagnostics.board_pooling_results import completion, check_lock_only, check_replay, check_locked_br_parity
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.flop_check_runtime import append, machine_snapshot, rss_for_tree, swap_usage
from src.diagnostics.global_bucket_validation import global_labels, project_compact, load_tables, ALIASES
from src.diagnostics.saved_hu20 import file_hash

GIB = 1024**3
BINARY_HASH = "fb32974d9d211fa66001d1efb330ec4af2d825005d24b287dc6cf3c37fa8812f"


def load(path):
    return json.loads(Path(path).read_text())


def response(path, *, include_statistics=False):
    path = Path(path)
    rows = []
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        stream = io.TextIOWrapper(archive.open("response.jsonl"))
    else:
        archive = None
        stream = (gzip.open if path.suffix == ".gz" else open)(path, "rt")
    with stream:
        for line in stream:
            if not include_statistics and '"event":"pooling_statistics"' in line[:100]:
                continue
            row = json.loads(line)
            if include_statistics or row["event"] != "pooling_statistics":
                rows.append(row)
    if archive is not None:
        archive.close()
    return rows


def seal(path):
    """Losslessly retain exact raw bytes; release only owned transient buffers."""
    path = Path(path)
    out = path.with_name(path.name + ".gz")
    digest = file_hash(path)
    with path.open("rb") as source, gzip.open(out, "wb", compresslevel=6) as target:
        shutil.copyfileobj(source, target, 8 * 1024**2)
    h = hashlib.sha256()
    with gzip.open(out, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            h.update(block)
    if h.hexdigest() != digest:
        raise ValueError("Response sealing readback differs")
    receipt = {"raw_sha256": digest, "raw_bytes": path.stat().st_size,
               "gzip_sha256": file_hash(out), "gzip_bytes": out.stat().st_size}
    path.unlink()
    return receipt


def retained_response(folder, *, include_statistics=False):
    record=load(Path(folder)/"result.json")
    if record["response"].get("zip_path"):
        path=Path(record["response"]["zip_path"])
        if file_hash(path)!=record["response"]["zip_sha256"]:raise ValueError("Retained native ZIP differs")
    else:
        path=Path(folder)/"response.jsonl.gz"
        if file_hash(path)!=record["response"]["gzip_sha256"]:raise ValueError("Retained response gzip differs")
    return response(path,include_statistics=include_statistics)


def archive_native(root, path, destination):
    """The first persistent native payload is an immutable member-hashed ZIP."""
    archive_dir=Path.home()/"Local/Research-Cloud/PR-190-HU20-bucket-validation"
    archive_dir.mkdir(parents=True,exist_ok=True)
    parts=destination.relative_to(root/"run").parts
    archive_path=archive_dir/("native-"+"-".join(parts)+".zip")
    digest=file_hash(path);size=path.stat().st_size
    manifest={"members":[{"path":"response.jsonl","bytes":size,"sha256":digest}],
              "source":str(destination),"binary_sha256":BINARY_HASH}
    with zipfile.ZipFile(archive_path,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        z.write(path,"response.jsonl")
        z.writestr("ARCHIVE-MANIFEST.json",json.dumps(manifest,sort_keys=True)+"\n")
    h=hashlib.sha256()
    with zipfile.ZipFile(archive_path) as z,z.open("response.jsonl") as source:
        for block in iter(lambda:source.read(8*1024**2),b""):h.update(block)
    if h.hexdigest()!=digest:raise ValueError("Native ZIP member readback differs")
    receipt={"raw_sha256":digest,"raw_bytes":size,"zip_path":str(archive_path),
             "zip_sha256":file_hash(archive_path),"zip_bytes":archive_path.stat().st_size,
             "member":"response.jsonl","member_readback_verified":True}
    path.unlink()
    return receipt


def prepared_compact(job):
    path=Path(load(job["request"])["compact_path"])
    if job.get("compact_overlay"):
        if file_hash(job["base_compact_path"])!=job["base_compact_sha256"]:
            raise ValueError("Frozen overlay base compact differs")
        overlay=Path(job["compact_overlay"])
        if file_hash(overlay)!=job["overlay_sha256"]:raise ValueError("Global card overlay differs")
        data=load(job["base_compact_path"])
        with gzip.open(overlay,"rt") as source:patch=json.load(source)
        data.update(crossfit_labels=patch["crossfit_labels"],global_bucket_transport=patch["global_bucket_transport"])
        data["pool_keys"]={"v1":data["pool_keys"]["v1"],**patch["pool_keys"]}
        return data
    with gzip.open(path.with_name(path.name+".gz"),"rt") as source:return json.load(source)


def materialize_compact(job):
    request=load(job["request"]);path=Path(request["compact_path"])
    if not path.exists():
        if job.get("compact_overlay"):atomic_json(path,prepared_compact(job))
        else:
            with gzip.open(path.with_name(path.name+".gz"),"rb") as source,path.open("xb") as target:
                shutil.copyfileobj(source,target,8*1024**2)
    if file_hash(path)!=job["compact_sha256"]:raise ValueError("Materialized global compact hash differs")
    return request,path


def guard(root, budget, pid, *, worker=False):
    rss = rss_for_tree(pid)
    swap = swap_usage()
    if rss > (7 if worker else 8) * GIB:
        raise RuntimeError("Owned RSS ceiling")
    if swap - budget["swap_baseline_bytes"] > GIB:
        raise RuntimeError("Swap growth ceiling")
    free = shutil.disk_usage(root).free
    if free < 15 * GIB:
        raise RuntimeError("15-GiB free-disk floor")
    if time() >= budget["deadline_epoch"] - 3600:
        raise RuntimeError("48-hour cap reached closeout reserve")
    state = {"rss_bytes": rss, "swap_bytes": swap, "free_disk_bytes": free}
    if time() - getattr(guard, "last_log", 0) >= 5:
        append(root / "family-resources.jsonl", dict(state, timestamp=time(), pid=pid, worker=worker))
        guard.last_log = time()
    return state


def native(root, request, destination):
    budget = load(root / "budget.json")
    binary = root / "inputs/pr149/pooling-engineering-05-mac"
    if file_hash(binary) != BINARY_HASH:
        raise ValueError("Qualified native binary differs")
    if request["memory_budget_bytes"] != 4 * GIB:
        raise ValueError("Frozen 4-GiB arena differs")
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "request.json"
    atomic_json(path, request)
    output = destination / "response.jsonl"
    started = time(); peak = 0; last = 0; child = None
    try:
        with (destination / "stderr.log").open("w") as log:
            child = subprocess.Popen(["nice", "-n", "10", str(binary), str(path), str(output)],
                env=dict(os.environ, RAYON_NUM_THREADS="6"), stdout=log, stderr=log, start_new_session=True)
            while child.poll() is None:
                state = guard(root, budget, os.getpid(), worker=True)
                peak = max(peak, state["rss_bytes"])
                if time() - started > request["seconds"] + 300:
                    raise RuntimeError("Frozen per-job wall cap")
                if time() - last >= 5:
                    append(destination / "resources.jsonl", dict(state, timestamp=time(), elapsed=time()-started))
                    last = time()
                sleep(.5)
            if child.returncode:
                raise RuntimeError(f"Native exit {child.returncode}")
        guard(root, budget, os.getpid(), worker=True)
        rows = response(output)
        if any(r.get("event") == "gate" and not r["passed"] for r in rows):
            raise ValueError("Native scientific gate failed")
        sealed = archive_native(root, output, destination) if destination.is_relative_to(root/"run") else seal(output)
        result = {"binary_sha256": BINARY_HASH, "elapsed_seconds": time()-started,
                  "peak_owned_rss_bytes": peak, "request_sha256": file_hash(path), "response": sealed,
                  "rows": rows}
        if destination.is_relative_to(root/"run"):
            result["request_archive"]=seal(path)
        atomic_json(destination / "result.json", result)
        return result
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        # Keep failed/partial output unchanged; no automatic retry.
        atomic_json(destination / "failure.json", {"error": str(error), "elapsed_seconds": time()-started,
                    "peak_owned_rss_bytes": peak})
        raise


def original_job(root, job):
    folder = root / "inputs/pr149/prepared-03/jobs" / job["job"]
    for name, field in (("request.json", "request_sha256"), ("compact.json", "compact_sha256")):
        if file_hash(folder / name) != job[field]:
            raise ValueError("Restored prepared job differs")
    request = load(folder / "request.json")
    request["compact_path"] = str(folder / "compact.json")
    return request


def compare_collection(root, job, rows, *, exact=False):
    recorded = load(root / "baseline/main-06/collect" / job["job"] / "result.json")
    current = completion(rows)
    previous = recorded["completion"]
    if current["iterations"] != previous["iterations"]:
        raise ValueError("Frozen deterministic iteration count differs")
    for name in ("current_ev_chips", "mes_ev_chips", "exploitability_pct_pot"):
        left = current[name] if isinstance(current[name], list) else [current[name]]
        right = previous[name] if isinstance(previous[name], list) else [previous[name]]
        if any(abs(a-b) > (0 if exact else 1e-5*200) for a,b in zip(left,right,strict=True)):
            raise ValueError(f"Frozen equilibrium {name} differs")
    fresh = {(r["metric"], r["target_solver_seat"]): r for r in rows if r["event"] == "pooling_metric"}
    if set(fresh) != {(m,s) for m in ("e_bp", "e_root_v1") for s in (0,1)}:
        raise ValueError("Missing/duplicate collect measurements")
    for r in recorded["metrics"]:
        value = fresh[r["metric"],r["target_solver_seat"]]
        for field in ("gain_bb", "gain_pct_pot", "responder_br_chips", "reference_responder_value_chips"):
            if abs(value[field]-r[field]) > (0 if exact else 1e-5*(2 if field=="gain_bb" else 200)):
                raise ValueError("Original blueprint/per-root metric differs")
    return {"passed": True, "exact": exact, "iterations": current["iterations"]}


def prepare_board(root, spot):
    jobs = [j for j in load(root / "inputs/pr149/prepared-03/manifest.json")["jobs"] if j["spot"] == spot]
    data = load(root / "inputs/pr149/prepared-03/jobs" / jobs[0]["job"] / "compact.json")
    labels, counts = global_labels(data, load_tables(root / "inputs/tables"))
    folder = root / "prepared"; folder.mkdir(exist_ok=True)
    atomic_json(folder / f"labels-{spot}.json", {"labels": labels, "counts": counts})
    output = []; patch=None; overlay=folder/f"overlay-{spot}.json.gz"
    for job in jobs:
        request = original_job(root, job)
        original = load(root / "inputs/pr149/prepared-03/jobs" / job["job"] / "compact.json")
        if any(original[field] != data[field] for field in ("board", "boards", "holdings", "codes")):
            raise ValueError("Lineage card features differ")
        projected = project_compact(request, original, labels)
        leaf = folder / "jobs" / job["job"]; leaf.mkdir(parents=True, exist_ok=False)
        atomic_json(leaf / "compact.json", projected)
        digest=file_hash(leaf/"compact.json")
        candidate={field:projected[field] for field in ("crossfit_labels","global_bucket_transport")}
        candidate["pool_keys"]={alias:projected["pool_keys"][alias] for alias in ALIASES.values()}
        if patch is None:
            patch=candidate
            with gzip.open(overlay,"wt",compresslevel=6) as target:
                json.dump(patch,target,sort_keys=True,allow_nan=False)
        elif patch!=candidate:raise ValueError("Global card overlay differs across lineages")
        base=root/"inputs/pr149/prepared-03/jobs"/job["job"]/"compact.json"
        # Verify lossless recomposition before releasing the transient full compact.
        composed=dict(original,crossfit_labels=patch["crossfit_labels"],global_bucket_transport=patch["global_bucket_transport"],
                      pool_keys={"v1":original["pool_keys"]["v1"],**patch["pool_keys"]})
        if composed!=projected:raise ValueError("Global overlay changes non-card information")
        (leaf/"compact.json").unlink()
        request["compact_path"] = str(leaf / "compact.json")
        atomic_json(leaf / "request.json", request)
        output.append(dict(job, request=str(leaf/"request.json"), request_sha256=file_hash(leaf/"request.json"),
                           compact_sha256=digest, compact_overlay=str(overlay), overlay_sha256=file_hash(overlay),
                           base_compact_path=str(base), base_compact_sha256=job["compact_sha256"]))
    atomic_json(folder / f"jobs-{spot}.json", output)


def worker(root, phase, job_id):
    manifest = load(root / "prepared/manifest.json")
    job = next(j for j in manifest["jobs"] if j["job"] == job_id)
    request, active_compact = materialize_compact(job)
    if file_hash(job["request"]) != job["request_sha256"] or file_hash(request["compact_path"]) != job["compact_sha256"]:
        raise ValueError("Global prepared inputs differ")
    if phase == "collect":
        result = native(root, request, root / "run/collect" / job_id)
        result["reference_gate"] = compare_collection(root, job, result["rows"])
    else:
        recorded = load(root / "inputs/pr149/main-06/collect" / job_id / "result.json")
        policy = root / "run" / f'crossfit-{job["evaluation_fold"]}-{job["lineage"]}.json'
        inventory = load(root / "run/pool-inventory.json")
        policy_receipt=inventory[policy.name]
        if isinstance(policy_receipt,dict):
            if file_hash(policy.with_name(policy.name+".gz"))!=policy_receipt["gzip_sha256"]:
                raise ValueError("Frozen fitted witness gzip differs")
            if not policy.exists():
                with gzip.open(policy.with_name(policy.name+".gz"),"rb") as source,policy.open("xb") as target:
                    shutil.copyfileobj(source,target,8*1024**2)
            expected_policy_hash=policy_receipt["raw_sha256"]
        else:expected_policy_hash=policy_receipt
        if file_hash(policy)!=expected_policy_hash:raise ValueError("Frozen fitted witness hash differs")
        request.update(pooling_phase="lock-only", max_iterations=0,
            reference_equilibrium_ev_chips=recorded["completion"]["current_ev_chips"],
            reference_response_sha256=recorded["runtime"]["response_sha256"],
            pooling_measurements=[{"metric":"e_recomputed_v1", "projection_metric":"v1", "policy_path":str(policy), "allow_missing":True}] + [{"metric":f"e_global{k}", "projection_metric":alias,
                                   "policy_path":str(policy), "allow_missing":True} for k,alias in ALIASES.items()])
        result = native(root, request, root / "run/relock" / job_id)
        # Original references are kept raw and hash-verified in the restored archive.
        original_rows = response(root / "inputs/pr149/main-06/collect" / job_id / "solver/reference.jsonl.gz")
        result["reference_gate"] = check_lock_only(original_rows, result["rows"], request["pot"], request["reference_response_sha256"])
        identities = [(r["metric"],r["target_solver_seat"]) for r in result["rows"] if r["event"]=="pooling_metric"]
        if len(identities)!=6 or set(identities)!={(m,s) for m in ("e_recomputed_v1","e_global50","e_global200") for s in (0,1)}:
            raise ValueError("Missing/duplicate global lock measurements")
        original_lock=load(root/"baseline/main-06/relock"/job_id/"result.json")
        expected=[r for r in original_lock["metrics"] if r["metric"]=="e_cross_v1"]
        actual=[dict(r,metric="e_cross_v1") for r in result["rows"] if r.get("metric")=="e_recomputed_v1"]
        result["v1_witness_reproduction_gate"]=check_locked_br_parity(actual,expected,request["pot"])
    if phase == "relock" and job["replay_sample"]:
        replay_request = dict(request, pooling_phase="relock", max_iterations=recorded["completion"]["iterations"], target_pct_pot=-1)
        replay = native(root, replay_request, root / "run/replay" / job_id)
        collected = retained_response(root / "run/collect" / job_id, include_statistics=True)
        replayed = retained_response(root / "run/replay" / job_id, include_statistics=True)
        result["replay_gates"] = [check_replay(collected, replayed, request["pot"]),
                                    check_locked_br_parity(result["rows"], replay["rows"], request["pot"])]
    result["job"] = job
    atomic_json(root / "run" / phase / job_id / "result.json", result)
    if job.get("compact_overlay") or active_compact.with_name(active_compact.name+".gz").exists():
        active_compact.unlink()
    if phase=="relock" and policy.with_name(policy.name+".gz").exists():policy.unlink()


def fit(root, lineage, fold):
    jobs = [j for j in load(root / "prepared/manifest.json")["jobs"] if j["lineage"] == lineage and j["evaluation_fold"] != fold]
    if len(jobs)!=20:
        raise ValueError("Opposite-half witness requires exactly 20 roots")
    def records():
        for job in jobs:
            path = root / "run/collect" / job["job"]
            receipt = load(path / "result.json")
            if not receipt["reference_gate"]["passed"]:
                raise ValueError("Collection is not reference-qualified")
            rows = retained_response(path, include_statistics=True)
            stats = [r["groups"] for r in rows if r["event"] == "pooling_statistics"]
            if len(stats)!=1:
                raise ValueError("Missing/duplicate global statistics")
            yield dict(job, groups=[r for r in stats[0] if r["metric"] in ("v1", *ALIASES.values())])
    policy = pool_statistics(records())
    policy.update(evaluation_fold=fold, training_fold=1-fold, training_spots=sorted(j["spot"] for j in jobs),
                  global_transport_aliases={alias:f"global-k{k}" for k,alias in ALIASES.items()})
    path=root/"run"/f"crossfit-{fold}-{lineage}.json"
    atomic_json(path,policy)
    receipt=seal(path)
    atomic_json(path.with_name(path.name+".receipt.json"),receipt)


def child(root, mode, *, phase=None, job=None, spot=None, lineage=None, fold=None):
    command = [sys.executable, "-m", "scripts.run_global_bucket_validation", "--root", str(root), "--mode", mode]
    for name,value in (("phase",phase),("job",job),("spot",spot),("lineage",lineage),("fold",fold)):
        if value is not None:command += ["--"+name,str(value)]
    budget = load(root / "budget.json")
    logs = root / "logs"; logs.mkdir(exist_ok=True)
    label = "-".join(str(x) for x in (mode,phase,job,spot,lineage,fold) if x is not None)
    with (logs / (label+".log")).open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        try:
            while process.poll() is None:
                guard(root,budget,os.getpid());sleep(1)
            if process.returncode:raise RuntimeError(f"{label} failed; see retained log")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=8)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise


def prepare(root):
    original = load(root / "inputs/pr149/prepared-03/manifest.json")
    spots = list(dict.fromkeys(j["spot"] for j in original["jobs"]))
    for i,spot in enumerate(spots):
        child(root,"prepare-board",spot=spot)
        atomic_json(root / "status.json", {"phase":"prepare", "done":i+1,"total":40,"timestamp":time()})
    by_id = {j["job"]:j for spot in spots for j in load(root / "prepared" / f"jobs-{spot}.json")}
    jobs = [by_id[j["job"]] for j in original["jobs"]]
    atomic_json(root / "prepared/manifest.json", {"jobs":jobs,"aliases":ALIASES,"original_manifest_sha256":file_hash(root/"inputs/pr149/prepared-03/manifest.json")})


def pilot(root):
    original = load(root / "inputs/pr149/prepared-03/manifest.json")
    job = original["jobs"][0]
    request = original_job(root,job)
    first = load(root/"pilot/legacy-collect/result.json") if (root/"pilot/legacy-collect/result.json").exists() else native(root,request,root/"pilot/legacy-collect")
    if first["binary_sha256"] != BINARY_HASH or first["request_sha256"] != file_hash(root/"pilot/legacy-collect/request.json"):
        raise ValueError("Legacy pilot identity differs")
    gate = compare_collection(root,job,first["rows"],exact=True)
    old = load(root / "baseline/main-06/collect" / job["job"] / "result.json")
    request.update(pooling_phase="lock-only",max_iterations=0,
                   reference_equilibrium_ev_chips=old["completion"]["current_ev_chips"],
                   reference_response_sha256=old["runtime"]["response_sha256"])
    policy=root/f'inputs/pr149/main-06/crossfit-0-{job["lineage"]}.json'
    request["pooling_measurements"]=[{"metric":"e_cross_v1","projection_metric":"v1","policy_path":str(policy),"allow_missing":True},
        {"metric":"e_cross_eq50","projection_metric":"eq50-fit1","policy_path":str(policy),"allow_missing":True}]
    second=native(root,request,root/"pilot/legacy-lock")
    previous=response(root/"inputs/pr149/main-06/relock"/job["job"]/"solver/reference.jsonl.gz")
    clean=lambda row:{k:v for k,v in row.items() if k!="solver_peak_rss_bytes"}
    expected=[clean(r) for r in previous if r["event"]=="pooling_metric" and r["metric"] in ("e_cross_v1","e_cross_eq50")]
    actual=[clean(r) for r in second["rows"] if r["event"]=="pooling_metric"]
    if actual!=expected:raise ValueError("Legacy v1/fitted-equity50 metrics do not reproduce exactly")
    atomic_json(root/"pilot/legacy-parity.json",{"collect":gate,"locked_metrics_exact":True,"metrics":actual,"binary_unchanged":True})


def singleton_pilot(root):
    job=load(root/"prepared/manifest.json")["jobs"][0]
    collect=root/"run/collect"/job["job"]
    record=load(collect/"result.json")
    if not record["reference_gate"]["passed"]:raise ValueError("Global collection pilot unqualified")
    rows=retained_response(collect,include_statistics=True)
    statistics=[r["groups"] for r in rows if r["event"]=="pooling_statistics"]
    if len(statistics)!=1:raise ValueError("Global collection statistics absent")
    policy=pool_statistics([dict(job,groups=[r for r in statistics[0] if r["metric"] in ALIASES.values()])])
    policy_path=root/"pilot/singleton-global-policy.json";atomic_json(policy_path,policy)
    del rows,statistics,policy
    original=load(root/"baseline/main-06/collect"/job["job"]/"result.json")
    request,active_compact=materialize_compact(job)
    request.update(pooling_phase="lock-only",max_iterations=0,
        reference_equilibrium_ev_chips=original["completion"]["current_ev_chips"],
        reference_response_sha256=original["runtime"]["response_sha256"],
        pooling_measurements=[{"metric":f"e_singleton_global{k}","projection_metric":alias,
                              "policy_path":str(policy_path),"allow_missing":False} for k,alias in ALIASES.items()])
    result=native(root,request,root/"pilot/global-singleton-lock")
    result["reference_gate"]=check_lock_only(response(root/"inputs/pr149/main-06/collect"/job["job"]/"solver/reference.jsonl.gz"),
        result["rows"],request["pot"],request["reference_response_sha256"])
    atomic_json(root/"pilot/global-singleton-lock/result.json",result)
    if job.get("compact_overlay") or active_compact.with_name(active_compact.name+".gz").exists():active_compact.unlink()


def main_run(root):
    if not load(root/"pilot/legacy-parity.json")["locked_metrics_exact"]:
        raise ValueError("Legacy parity prerequisite missing")
    jobs=load(root/"prepared/manifest.json")["jobs"]
    # Main is admitted separately after the first global collection pilot.
    admission=load(root/"main-admission.json")
    if not admission["passed"] or admission["prepared_manifest_sha256"]!=file_hash(root/"prepared/manifest.json"):
        raise ValueError("Measured main admission absent or changed")
    (root/"run").mkdir(exist_ok=True)
    for phase in ("collect","fit","relock"):
        if phase=="fit":
            paths=[]
            for lineage in sorted({j["lineage"] for j in jobs}):
                for fold in (0,1):
                    child(root,"fit",lineage=lineage,fold=fold)
                    paths.append(root/"run"/f"crossfit-{fold}-{lineage}.json")
            atomic_json(root/"run/pool-inventory.json",{p.name:load(p.with_name(p.name+".receipt.json")) for p in paths})
            continue
        for i,job in enumerate(jobs):
            # The outcome-blind first collection pilot is the identical first main job.
            dest=root/"run"/phase/job["job"]
            if phase=="collect" and i==0 and dest.exists():
                r=load(dest/"result.json")
                if not r["reference_gate"]["passed"] or r["job"]!=job:raise ValueError("Pilot collection differs")
            else:child(root,"worker",phase=phase,job=job["job"])
            atomic_json(root/"status.json",{"phase":phase,"done":i+1,"total":120,"timestamp":time()})
    atomic_json(root/"status.json",{"phase":"complete","timestamp":time()})


def main():
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Owned runner terminated")))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=["singleton","legacy-collect","pilot","prepare","prepare-board","worker","fit","run"],required=True)
    for name in ("phase","job","spot"):p.add_argument("--"+name)
    for name in ("lineage","fold"):p.add_argument("--"+name,type=int)
    a=p.parse_args();root=a.root.resolve()
    if a.mode=="legacy-collect":
        job=load(root/"inputs/pr149/prepared-03/manifest.json")["jobs"][0]
        result=native(root,original_job(root,job),root/"pilot/legacy-collect")
        atomic_json(root/"pilot/legacy-collect-parity.json",compare_collection(root,job,result["rows"],exact=True))
    elif a.mode=="prepare-board":prepare_board(root,a.spot)
    elif a.mode=="worker":worker(root,a.phase,a.job)
    elif a.mode=="fit":fit(root,a.lineage,a.fold)
    elif a.mode=="singleton":singleton_pilot(root)
    elif a.mode=="pilot":pilot(root)
    elif a.mode=="prepare":prepare(root)
    else:main_run(root)


if __name__=="__main__":
    main()
