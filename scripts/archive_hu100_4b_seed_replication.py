"""PR236 measured multiGB ZIP calibration and member-verified archival."""
import argparse
import json
import math
import re
from pathlib import Path
import shutil
import subprocess
import sys
from time import time

from scripts import run_hu100_4b_seed_replication as campaign
from scripts.archive_hu100_4b_seed_ladder import seal
from scripts.evaluate_hu100_direct import put
from src.policies.files import file_hash

ROOT, OUT = campaign.ROOT, campaign.OUT
MODULE = "scripts.archive_hu100_4b_seed_replication"
GIB = 1024**3
read = campaign.read


def inventory():
    indexed = {a["sha256"]: a for a in read(campaign.INDEX)["assets"]}
    files, omitted, aliases = [], [], []
    seen = {}
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(OUT)
        if rel.as_posix() in ("archive-receipt.json", "archive-admission.json"):
            continue
        if rel.parts[0] == "archive-calibrations" and path.suffix == ".zip":
            continue
        if rel.parts[:3] == ("guards", "operations", "archive-final"):
            continue
        if (len(rel.parts) >= 3 and rel.parts[:2] == ("guards", "operations")
                and rel.parts[2].startswith("archive-calibration-")
                and not (OUT / "guards/operations" / rel.parts[2] / "receipt.json").exists()):
            continue
        # Snapshot copies, inputs and training checkpoints are model payloads.
        # Traces are ordinary gzip streams and remain raw scientific evidence.
        model = path.suffix == ".gz" and (rel.parts[0] in ("inputs", "training")
                                          or "models" in rel.parts)
        if model:
            sha = file_hash(path)
            if sha in indexed:
                pin = indexed[sha]
                if path.stat().st_size != pin["bytes"]:
                    raise ValueError("STOP indexed model changed before archive")
                omitted.append({"local_path": str(path), "relative_path": str(rel), "asset": pin})
                continue
        stat = path.stat()
        inode = stat.st_dev, stat.st_ino
        member = "research/" + rel.as_posix()
        if inode in seen:
            aliases.append({"member": member, "canonical_member": seen[inode],
                            "bytes": stat.st_size, "sha256": file_hash(path)})
            continue
        seen[inode] = member
        files.append((path, member))
    # Preserve the owner authorization, issue resolution, source clearance and
    # exact PR quote receipts even though their working location is ignored.
    for path in sorted((ROOT / "planning").glob("*")):
        if path.is_file() and path.suffix in (".json", ".md"):
            if path.stat().st_size > 1024**2:
                raise ValueError("Compact planning receipt exceeds admitted size")
            files.append((path, "provenance/planning/" + path.name))
    files.append((campaign.BINARY, "research/bin/hu20-trainer"))
    changed = subprocess.check_output(["git","diff","--name-only",
        "c3869c9d8884cf7eaab66790b17b3e36398a3ac3","HEAD"],text=True).splitlines()
    for name in changed:
        path=ROOT/name
        if path.is_file() and path.stat().st_size<=1024**2:
            files.append((path,"source/"+name))
    return files,omitted,aliases


def calibration_receipt():
    paths = sorted((OUT / "archive-calibrations").glob("*.json"))
    if not paths:
        raise ValueError("Fresh actual-byte archive calibration required")
    return max((read(path) for path in paths), key=lambda item: item["measured_at"])


def workload_requires_refresh(logical, members, calibration):
    return (logical > 2 * calibration["logical_member_bytes"]
            or members > 2 * calibration["members"])


def calibrate(label):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", label):
        raise ValueError("Unique calibration label required")
    source=campaign.source()
    files,omitted,aliases=inventory()
    # Calibrate on both actual new terminal sets plus pilot traces, metadata
    # and closed guards. Indexed old model bytes are omitted by inventory().
    selected = files
    logical=sum(p.stat().st_size for p,m in selected)
    if logical<3*GIB:
        raise ValueError("Actual multiGB calibration workload required")
    if shutil.disk_usage(OUT).free<16*GIB+logical+GIB:
        raise ValueError("Preventive calibration ZIP storage admission")
    result=seal(selected,OUT/"archive-calibrations"/(label+".zip"),{
        "pr":236,"source":source,"scope":"measured multiGB archival calibration",
        "model_dependencies":omitted,"hardlink_aliases":aliases})
    result.update(measured_at=time(), label=label,
                  receipt_path=str(OUT/"archive-calibrations"/(label+".json")))
    put(OUT/"archive-calibrations"/(label+".json"),result)


def archive_admission():
    freeze=read(OUT/"frozen-final.json")
    calibration=calibration_receipt()
    files,omitted,aliases=inventory()
    logical=sum(p.stat().st_size for p,m in files)
    free=shutil.disk_usage(OUT).free
    # Count the current actual bytes immediately before launch, with3x fresh
    # source-hash/write/full-member-readback/whole-hash timing headroom.
    allowance=3*calibration["seconds_including_source_hash_pack_readback_whole_hash"]*max(
        1,logical/calibration["logical_member_bytes"])
    if free<16*GIB+logical+2*GIB:
        raise ValueError("Preventive final archive storage admission")
    if workload_requires_refresh(logical, len(files), calibration):
        raise ValueError("Archive workload differs materially; refresh calibration before launch")
    record={"at":time(),"logical_member_bytes":logical,"members":len(files),
            "free_bytes":free,"required_free_bytes":16*GIB+logical+2*GIB,
            "allowance_seconds":allowance,"calibration":calibration,
            "source":freeze["source"],"headroom_multiplier":3,
            "quote_not_based_on_payoffs":True}
    put(OUT/"archive-admission.json",record)
    return record


def pack():
    source=campaign.source()
    freeze=read(OUT/"frozen-final.json")
    if source!=freeze["source"] or read(OUT/"summary.json")["status"]!="verified":
        raise ValueError("Frozen source and fully verified results required")
    files,omitted,aliases=inventory()
    cloud=Path.home()/"Local/Research-Cloud/PR-236-hu100-4b-seed-replication"
    cloud.mkdir(parents=True,exist_ok=True)
    logical=sum(p.stat().st_size for p,m in files)
    if shutil.disk_usage(cloud).free<16*GIB+logical+2*GIB:
        raise ValueError("Preventive archive destination storage admission")
    receipt=seal(files,cloud/"hu100-4b-seed-replication-M4-20261011.zip",{
        "pr":236,"source":source,"source_base":"c3869c9d8884cf7eaab66790b17b3e36398a3ac3",
        "binary_sha256":file_hash(campaign.BINARY),"model_dependencies":omitted,
        "hardlink_aliases":aliases,"originals_retained":True,
        "calibration_zip_excluded_to_avoid_duplicate_model_bytes":True,
        "live_archive_guard_excluded":"research/guards/operations/archive-final"})
    put(OUT/"archive-receipt.json",receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation",choices=("calibrate","calibrate-worker","pack","pack-worker"))
    p.add_argument("--label",default="initial")
    a=p.parse_args()
    if a.operation=="calibrate":
        # A workload refresh is a new timing measurement with exclusive paths,
        # never an overwrite or retry of a failed scientific operation.
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", a.label):
            raise ValueError("Unique calibration label required")
        campaign.guarded("archive-calibration-"+a.label,
                         [sys.executable,"-m",MODULE,"calibrate-worker","--label",a.label])
    elif a.operation=="calibrate-worker":
        calibrate(a.label)
    elif a.operation=="pack":
        admission=archive_admission()
        campaign.guarded("archive-final",[sys.executable,"-m",MODULE,"pack-worker"],
                         admission["allowance_seconds"])
    else:
        pack()


if __name__=="__main__":
    main()
