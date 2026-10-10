"""Member-hashed PR226 ZIPs; indexed gate models remain archive dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_STORED

from scripts import run_hu100_4b_seed_ladder as campaign
from scripts.evaluate_hu100_direct import put
from src.policies.files import file_hash

ROOT, OUT = campaign.ROOT, campaign.OUT
CHUNK = 1024*1024


def stream_hash(stream):
    sha, size = hashlib.sha256(), 0
    while chunk := stream.read(CHUNK):
        sha.update(chunk)
        size += len(chunk)
    return size, sha.hexdigest()


def seal(files, destination, metadata):
    destination.parent.mkdir(parents=True, exist_ok=True)
    pins = {}
    tick = perf_counter()
    for path, member in files:
        if member in pins:
            raise ValueError("Duplicate archive member")
        stat = path.stat()
        pins[member] = {"bytes": stat.st_size, "sha256": file_hash(path),
            "source_path": str(path), "source_mtime_ns": stat.st_mtime_ns}
    manifest = {"version": 1, "metadata": metadata, "members": pins}
    raw = (json.dumps(manifest, sort_keys=True, indent=2)+"\n").encode()
    with ZipFile(destination, "x", compression=ZIP_STORED, allowZip64=True) as archive:
        for path, member in files:
            pin = pins[member]
            if path.stat().st_size != pin["bytes"] or path.stat().st_mtime_ns != pin["source_mtime_ns"]:
                raise ValueError("Archive source changed before write")
            archive.write(path, member)
        archive.writestr("ARCHIVE-MANIFEST.json", raw)
    with ZipFile(destination) as archive:
        if set(archive.namelist()) != set(pins) | {"ARCHIVE-MANIFEST.json"}:
            raise ValueError("Archive member inventory differs")
        if archive.read("ARCHIVE-MANIFEST.json") != raw:
            raise ValueError("Archive manifest readback differs")
        for member, pin in pins.items():
            with archive.open(member) as stream:
                size, sha = stream_hash(stream)
            if size != pin["bytes"] or sha != pin["sha256"]:
                raise ValueError("Archive member readback differs")
    whole = file_hash(destination)
    return {"status": "locally-verified", "path": str(destination), "bytes": destination.stat().st_size,
        "sha256": whole, "manifest_path": "ARCHIVE-MANIFEST.json",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(), "members": len(pins),
        "all_member_sizes_and_hashes_readback": True,
        "seconds_including_source_hash_pack_readback_whole_hash": perf_counter()-tick,
        "logical_member_bytes": sum(pin["bytes"] for pin in pins.values()),
        "cloud_upload_accepted": False, "remote_archive_bytes_downloaded": False}


def pilot():
    model = OUT/"timing-training/checkpoint.gz"
    result = seal([(model, "research/timing-training/checkpoint.gz")],
        OUT/"archive-timing-pilot.zip", {"scope": "timing-only", "pr": campaign.PR})
    put(OUT/"archive-pilot-cost.json", result)


def pack():
    revision = campaign.source()
    freeze = campaign.read(OUT/"frozen-final.json")
    if freeze["source"] != revision:
        raise ValueError("Frozen source required for archive")
    if campaign.read(OUT/"summary.json")["status"] != "verified":
        raise ValueError("Final verified summary required")
    exclusions = {item["path"]: item for item in freeze["indexed_model_exclusions"]}
    files = []
    excluded = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(OUT)
        name = rel.as_posix()
        # The current archive's own live guard stream cannot be an immutable
        # member. Its closed receipt/summary is retained separately in Git.
        if rel.parts[:3] == ("guards", "operations", "archive-final"):
            continue
        if name in exclusions:
            pin = exclusions[name]
            if path.stat().st_size != pin["bytes"] or file_hash(path) != pin["sha256"]:
                raise ValueError("STOP indexed model exclusion changed")
            excluded.append(pin)
            continue
        if name in ("archive-receipt.json", "archive-timing-pilot.zip"):
            continue
        files.append((path, "research/"+name))
    files.append((campaign.BINARY, "research/bin/hu20-trainer"))
    # Exact scientific source is restored from Git, without duplicating legacy
    # research blobs. Include the newly authored control/protocol bytes.
    import subprocess
    changed = subprocess.check_output(["git", "diff", "--name-only",
        "ff984da9a16bf1e76e891fab239d0785188a93af", "HEAD"], text=True).splitlines()
    for name in changed:
        path = ROOT/name
        if path.is_file() and path.stat().st_size <= CHUNK:
            files.append((path, "source/"+name))
    expected = sum(path.stat().st_size for path, member in files)+2*campaign.GIB
    cloud = Path.home()/f"Local/Research-Cloud/PR-{campaign.PR}-hu100-4b-seed-ladder"
    cloud.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cloud).free-expected < 16*campaign.GIB:
        raise ValueError("Preventive archive disk admission")
    destination = cloud/"hu100-4b-seed-ladder-M4-20261010.zip"
    result = seal(files, destination, {"pr": campaign.PR, "source": revision,
        "source_base": "ff984da9a16bf1e76e891fab239d0785188a93af",
        "binary_sha256": file_hash(campaign.BINARY),
        "model_dependencies": excluded,
        "dependency_indexes": ["docs/reports/hu100-3b-ladder-artifacts/model-input-index.json",
            "docs/reports/hu100-independent-stages-artifacts/model-index.json", "RESULTS_INDEX.md"],
        "restore_source": "git clone the repository into a fresh checkout; git checkout "+revision,
        "excluded_active_archive_guard": "research/guards/operations/archive-final",
        "originals_retained": True})
    put(OUT/"archive-receipt.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "pack", "pack-worker"))
    args = parser.parse_args()
    if args.command == "pack":
        quote = campaign.read(OUT/"frozen-final.json")
        campaign.guarded("archive-final", [sys.executable, "-m", "scripts.archive_hu100_4b_seed_ladder", "pack-worker"],
            quote["archive_quote_seconds"])
    elif args.command == "pack-worker":
        pack()
    else:
        pilot()
