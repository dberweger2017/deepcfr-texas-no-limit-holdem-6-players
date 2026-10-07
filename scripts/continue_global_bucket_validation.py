"""Explicit one-shot continuation after PR190's preserved resource stop.

Preparation verifies retained evidence and measures admission without scoring.
The run command preserves the interrupted attempt and never retries a failure.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import zipfile

from scripts.run_global_bucket_validation import (
    BINARY_HASH, GIB, child, compare_collection, guard, load, materialize_compact,
)
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.saved_hu20 import file_hash

ATTEMPT = "continuation-01"
ORIGINAL_BUDGET_SHA256 = "27003a9889f227890e15910833ba1fd4abb5cc827f47ab4e650153ec954830cc"
PREPARED_SHA256 = "476d0c08e043056e2073b1b858d36c1291541c16c5a81e1b908daa8f3cdcfaf6"
SOURCES = ("scripts/continue_global_bucket_validation.py", "scripts/run_global_bucket_validation.py")


def verified_prefix(root, jobs):
    """Only the complete qualified frozen prefix is eligible for reuse."""
    retained = []
    missing_seen = False
    for job in jobs:
        path = root/"run/collect"/job["job"]/"result.json"
        if not path.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise ValueError("Completed collections are not a frozen prefix")
        result = load(path)
        if (result["job"] != job or not result["reference_gate"]["passed"]
                or result["binary_sha256"] != BINARY_HASH
                or result["request_sha256"] != job["request_sha256"]):
            raise ValueError("Retained collection identity/gate differs")
        compare_collection(root, job, result["rows"])
        retained.append({"job": job["job"], "result_sha256": file_hash(path)})
    return retained


def verify_payload(receipt):
    path = Path(receipt["zip_path"])
    if file_hash(path) != receipt["zip_sha256"]:
        raise ValueError("Retained native ZIP hash differs")
    h = hashlib.sha256()
    size = 0
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("ARCHIVE-MANIFEST.json"))
        if manifest["binary_sha256"] != BINARY_HASH:
            raise ValueError("Retained native manifest binary differs")
        member = next(m for m in manifest["members"] if m["path"] == "response.jsonl")
        with archive.open("response.jsonl") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                h.update(block)
                size += len(block)
    if (h.hexdigest() != receipt["raw_sha256"] or size != receipt["raw_bytes"]
            or member["sha256"] != h.hexdigest() or member["bytes"] != size):
        raise ValueError("Retained native member differs")


def measured_resources(root, budget):
    commands = subprocess.check_output(["ps", "-axo", "pid,command"], text=True)
    for item in load(root/"background-roots.json")["roots"]:
        if any(line.strip().startswith(str(item["pid"])+" ")
               and item["command_fragment"] in line for line in commands.splitlines()):
            raise ValueError("An owned process remains active")
    state = guard(root, budget, os.getpid())
    pressure = subprocess.check_output(["memory_pressure", "-Q"], text=True)
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
    if not match:
        raise ValueError("Memory availability measurement unavailable")
    return dict(state, memory_available_percent=int(match.group(1)),
                swap_headroom_bytes=budget["swap_baseline_bytes"]+GIB-state["swap_bytes"],
                measured_epoch=time.time())


def prepare(root):
    if (root/"continuation-start.lock").exists():
        raise ValueError("Continuation was already dispatched")
    if (file_hash(root/"budget.json") != ORIGINAL_BUDGET_SHA256
            or file_hash(root/"prepared/manifest.json") != PREPARED_SHA256
            or file_hash(root/"inputs/pr149/pooling-engineering-05-mac") != BINARY_HASH):
        raise ValueError("Original budget, prepared manifest or qualified binary changed")
    budget = load(root/"budget.json")
    jobs = load(root/"prepared/manifest.json")["jobs"]
    stop = load(root/"stop-receipt.json")
    if (len(jobs) != 120 or len({j["job"] for j in jobs}) != 120
            or stop["reason"] != "Swap growth ceiling" or load(root/"status.json")["phase"] != "stopped"):
        raise ValueError("Unexpected study or stop state")
    resources = measured_resources(root, budget)
    prefix = verified_prefix(root, jobs)
    if len(prefix) != 19 or stop["qualified_reference_gates"] != 19:
        raise ValueError("Resource-stop prefix differs")
    for retained in prefix:
        folder = root/"run/collect"/retained["job"]
        result = load(folder/"result.json")
        verify_payload(result["response"])
        request = result["request_archive"]
        path = folder/"request.json.gz"
        if file_hash(path) != request["gzip_sha256"] or request["raw_sha256"] != result["request_sha256"]:
            raise ValueError("Retained request differs")
    # Each worker still materializes and hashes its exact full compact before solving.
    checked = set()
    for job in jobs:
        if file_hash(job["request"]) != job["request_sha256"]:
            raise ValueError("Prepared request differs")
        for path_field, hash_field in (("compact_overlay", "overlay_sha256"),
                                      ("base_compact_path", "base_compact_sha256")):
            if job.get(path_field) and job[path_field] not in checked:
                if file_hash(job[path_field]) != job[hash_field]:
                    raise ValueError("Prepared overlay/base differs")
                checked.add(job[path_field])
    materialize_compact(jobs[19])
    for item in stop["partial_files_retained"]:
        path = root/item["path"]
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError("Interrupted evidence differs")
    archive_dir = Path.home()/"Local/Research-Cloud/PR-190-HU20-bucket-validation"
    for phase in ("collect", "relock", "replay"):
        planned = jobs[19:] if phase == "collect" else jobs
        for job in planned:
            if (archive_dir/f"native-{phase}-{job['job']}.zip").exists():
                raise ValueError("A future native archive already exists; never overwrite")
    if any((root/"run").glob("crossfit-*")) or (root/"run/relock").exists() or (root/"run/replay").exists():
        raise ValueError("Unexpected later-phase evidence")
    admission = load(root/"main-admission.json")
    first = load(root/"run/collect"/jobs[0]["job"]/"result.json")
    historical = {j["job"]: load(root/"baseline/main-06/collect"/j["job"]/"result.json") for j in jobs}
    scale = first["elapsed_seconds"]/historical[jobs[0]["job"]]["runtime"]["elapsed_seconds"]
    collect = sum(historical[j["job"]]["runtime"]["elapsed_seconds"] for j in jobs[19:])*scale
    other = admission["timing_components_seconds"]["locks"]+admission["timing_components_seconds"]["replays"]
    members = {m["path"]: m for m in load(root/"pr149-manifest.json")["files"]}
    raw = sum(members[f"main-06/collect/{j['job']}/solver/response.jsonl"]["bytes"]
              for j in jobs[19:]+[j for j in jobs if j["replay_sample"]])
    zip_forecast = raw*first["response"]["zip_bytes"]/members[f"main-06/collect/{jobs[0]['job']}/solver/response.jsonl"]["bytes"]
    disk_quote = zip_forecast*1.2+1.5*GIB
    resources = measured_resources(root, budget)
    conservative = 1.5*(collect+other)+7200
    available = budget["deadline_epoch"]-3600-time.time()
    checks = dict(time=conservative < available,
                  disk=resources["free_disk_bytes"]-disk_quote >= 15*GIB,
                  memory=resources["memory_available_percent"] >= 50,
                  swap_headroom=resources["swap_headroom_bytes"] >= GIB/2)
    proof = dict(passed=all(checks.values()), checks=checks, attempt=ATTEMPT,
                 created_epoch=time.time(), resources=resources, retained_prefix=prefix,
                 prepared_manifest_sha256=PREPARED_SHA256, budget_sha256=ORIGINAL_BUDGET_SHA256,
                 stop_receipt_sha256=file_hash(root/"stop-receipt.json"),
                 original_deadline_epoch=budget["deadline_epoch"], expected_remaining_seconds=collect+other+7200,
                 conservative_remaining_seconds=conservative, available_scientific_seconds=available,
                 incremental_disk_bytes=disk_quote,
                 conservative_final_free_disk_bytes=resources["free_disk_bytes"]-disk_quote,
                 source_sha256={p: file_hash(root.parents[1]/p) for p in SOURCES},
                 original_limits_unchanged=True, outcome_values_inspected=False)
    path = root/"continuation-admission.json"
    with path.open("x") as stream:
        json.dump(proof, stream, sort_keys=True)
        stream.write("\n")
    print(json.dumps(proof, sort_keys=True), flush=True)
    if not proof["passed"]:
        raise RuntimeError("Continuation admission failed; preserve evidence, do not launch")


def run(root):
    proof = load(root/"continuation-admission.json")
    if (not proof["passed"] or proof["attempt"] != ATTEMPT
            or time.time()-proof["created_epoch"] > 3600
            or file_hash(root/"budget.json") != proof["budget_sha256"]
            or file_hash(root/"prepared/manifest.json") != proof["prepared_manifest_sha256"]
            or file_hash(root/"stop-receipt.json") != proof["stop_receipt_sha256"]
            or any(file_hash(root.parents[1]/p) != digest for p, digest in proof["source_sha256"].items())):
        raise ValueError("Continuation admission absent, expired or changed")
    jobs = load(root/"prepared/manifest.json")["jobs"]
    prefix = verified_prefix(root, jobs)
    if prefix != proof["retained_prefix"]:
        raise ValueError("Retained evidence changed after admission")
    resources = guard(root, load(root/"budget.json"), os.getpid())
    if resources["free_disk_bytes"]-proof["incremental_disk_bytes"] < 15*GIB:
        raise ValueError("Continuation disk admission no longer holds")
    if time.time()+proof["conservative_remaining_seconds"] >= proof["original_deadline_epoch"]-3600:
        raise ValueError("Continuation timing admission no longer holds")
    with (root/"continuation-worker-start.lock").open("x") as stream:
        stream.write(str(time.time())+"\n")
    interrupted = root/"run/collect"/jobs[len(prefix)]["job"]
    retained = root/"attempts/initial-resource-stop/collect"/interrupted.name
    retained.parent.mkdir(parents=True, exist_ok=True)
    if retained.exists():
        raise ValueError("Interrupted evidence already relocated")
    interrupted.rename(retained)
    atomic_json(root/"interrupted-attempt-relocation.json",
                dict(original=str(interrupted), retained=str(retained), stop_receipt_sha256=proof["stop_receipt_sha256"]))
    try:
        for i, job in enumerate(jobs[len(prefix):], len(prefix)+1):
            child(root, "worker", phase="collect", job=job["job"], log_tag=ATTEMPT)
            atomic_json(root/"status.json", dict(phase="collect", done=i, total=120, attempt=ATTEMPT, timestamp=time.time()))
        policies = []
        for lineage in sorted({j["lineage"] for j in jobs}):
            for fold in (0, 1):
                child(root, "fit", lineage=lineage, fold=fold, log_tag=ATTEMPT)
                policies.append(root/"run"/f"crossfit-{fold}-{lineage}.json")
        atomic_json(root/"run/pool-inventory.json",
                    {p.name: load(p.with_name(p.name+".receipt.json")) for p in policies})
        for i, job in enumerate(jobs, 1):
            child(root, "worker", phase="relock", job=job["job"], log_tag=ATTEMPT)
            atomic_json(root/"status.json", dict(phase="relock", done=i, total=120, attempt=ATTEMPT, timestamp=time.time()))
        atomic_json(root/"status.json", dict(phase="complete", attempt=ATTEMPT, timestamp=time.time()))
    except BaseException as error:
        atomic_json(root/"continuation-failure.json", dict(error=str(error), timestamp=time.time(), automatic_restart=False))
        atomic_json(root/"status.json", dict(phase="stopped", reason=str(error), attempt=ATTEMPT, timestamp=time.time()))
        raise


def main():
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Owned continuation terminated")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else run)(args.root.resolve())


if __name__ == "__main__":
    main()
