"""Archive-only recovery of the completed HU100 campaign; retain the failed ZIP."""
import hashlib
import json
from pathlib import Path
import sys
import os
import re
import subprocess
from time import time

import psutil
import urllib.request
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

from scripts import run_native_hu100_growth_1b as campaign
from src.policies.files import file_hash

DEST = Path.home()/"Local/Research-Cloud/PR-207-hu100-1b/hu100-1b-campaign-M4-retry-20261009.zip"
READMISSION = campaign.OUT/"archive-readmission.json"
OPERATION = "archive-retry"
COMMAND = [sys.executable, "-m", "scripts.archive_native_hu100_growth_1b", "--pack"]
SWAP_LIMIT = 3*campaign.GIB


def admitted_swap_limit(name, command, path):
    """Owner approval applies only to this pack command, never training or play."""
    if name != OPERATION or list(map(str, command)) != COMMAND or Path(path) != READMISSION:
        raise ValueError("Archive-only readmission required")
    a = campaign.read(path)
    if (a["owner_instruction"] != "You can use up 3 gb swap. You can try again"
        or a["scope"] != "archive-only" or a["swap_growth_bytes"] != SWAP_LIMIT
        or a["absolute_swap_ceiling_bytes"] != 3_000_000_000
        or a["destination"] != str(DEST)):
        raise ValueError("Owner archive readmission differs")
    for key, source in (("failure_sha256", campaign.OUT/"campaign-failure.json"),
                        ("baseline_sha256", campaign.OUT/"baseline.json"),
                        ("scientific_complete_sha256", campaign.OUT/"evaluation/complete.json")):
        if a[key] != file_hash(source):
            raise ValueError("Retained readmission input changed: "+key)
    if campaign.read(campaign.OUT/"campaign-failure.json")["operation"] != "archive":
        raise ValueError("Only the original archive failure can be readmitted")
    if campaign.read(campaign.OUT/"evaluation/complete.json")["status"] != "verified":
        raise ValueError("Verified completed science required")
    if (campaign.OUT/"operations"/OPERATION/"retry-failure.json").exists():
        raise ValueError("Archive retry is stopped")
    return min(SWAP_LIMIT, a["absolute_swap_ceiling_bytes"]-campaign.read(campaign.OUT/"baseline.json")["host"]["swap_bytes"])


def source_paths():
    paths = {}
    for folder, prefix in ((campaign.OUT, "research"), (campaign.ROOT/"results/qualification", "qualification")):
        for p in folder.rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(folder)
            # The current supervisor stream and launch log are still changing.
            # All original failure logs are frozen and included.
            if prefix == "research" and (rel.parts[:2] == ("operations", OPERATION)
                                        or rel.name == "archive-retry-launch.log"):
                continue
            paths[prefix+"/"+str(rel)] = p
    paths["runtime/hu20-trainer"] = campaign.BINARY
    return paths


def pack():
    admitted_swap_limit(OPERATION, COMMAND, READMISSION)
    url = "https://api.github.com/repos/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pulls/207"
    with urllib.request.urlopen(url, timeout=20) as response:
        pr = json.load(response)
    campaign.write(campaign.OUT/"owning-pr-before-archive-retry.json",
                   {"url":url, "state":pr["state"], "merged":pr["merged"], "at":time()})
    if pr["state"] != "open" or pr["merged"]:
        raise ValueError("This owner's open campaign only")
    paths = source_paths()
    members = []
    for name, p in sorted(paths.items()):
        stat = p.stat()
        members.append({"path":name, "bytes":stat.st_size, "sha256":file_hash(p),
                        "mtime_ns":stat.st_mtime_ns, "original_path":str(p)})
    upper = sum(m["bytes"] for m in members)+len(members)*1024+1024**2
    if campaign.host()["disk_free_bytes"]-upper <= campaign.DISK_FLOOR:
        raise ValueError("Measured worst-case archive does not fit above disk floor")
    manifest = {
        "source":campaign.read(campaign.OUT/"training/intent.json")["source"],
        "archive_source":campaign.identity(), "source_root":str(campaign.ROOT), "created":time(),
        "members":members, "originals_retained":True, "owning_pr":207,
        "preserved_external_partial":campaign.read(READMISSION)["preserved_partial"],
        "later_receipts":"Current archive lifecycle, upload acceptance and evidence review remain compact Git receipts"
    }
    encoded = json.dumps(manifest, sort_keys=True).encode()
    if not DEST.parent.is_dir():
        raise ValueError("Confirmed native Drive folder required")
    # Exclusive creation preserves the original failed synced ZIP and any retry.
    with ZipFile(DEST, "x", compression=ZIP_STORED, allowZip64=True) as z:
        for m in members:
            p = paths[m["path"]]
            stat = p.stat()
            if stat.st_size != m["bytes"] or stat.st_mtime_ns != m["mtime_ns"]:
                raise ValueError("Archive source changed: "+m["path"])
            compression = ZIP_STORED if p.suffix in (".gz", ".zip") or p == campaign.BINARY else ZIP_DEFLATED
            z.write(p, m["path"], compress_type=compression,
                    compresslevel=1 if compression == ZIP_DEFLATED else None)
        z.writestr("ARCHIVE-MANIFEST.json", encoded, compress_type=ZIP_DEFLATED, compresslevel=1)
    with ZipFile(DEST) as z:
        if z.read("ARCHIVE-MANIFEST.json") != encoded:
            raise ValueError("Manifest readback mismatch")
        for m in members:
            digest = hashlib.sha256(); size = 0
            with z.open(m["path"]) as stream:
                while chunk := stream.read(8*1024**2):
                    digest.update(chunk); size += len(chunk)
            if size != m["bytes"] or digest.hexdigest() != m["sha256"]:
                raise ValueError("Archive/member exactness mismatch: "+m["path"])
    campaign.write(campaign.OUT/"archive-receipt.json", {
        "status":"locally-verified", "archive":str(DEST), "bytes":DEST.stat().st_size,
        "sha256":file_hash(DEST), "manifest_member":"ARCHIVE-MANIFEST.json",
        "manifest_sha256":hashlib.sha256(encoded).hexdigest(), "members_verified":len(members),
        "source":manifest["source"], "archive_source":manifest["archive_source"],
        "originals_retained":True, "remote_bytes_downloaded":False,
        "finished":time(), "cloud_acceptance":"pending"
    })




def archive_conflicts(archive):
    """Query the documented Foundation API, rather than an absent provider field."""
    before=DEST.stat()
    script='''function run(argv) {
        ObjC.import("Foundation");
        const url = $.NSURL.fileURLWithPath(argv[0]);
        const versions = $.NSFileVersion.unresolvedConflictVersionsOfItemAtURL(url);
        return JSON.stringify({
            api: "NSFileVersion.unresolvedConflictVersionsOfItemAtURL",
            exists: Boolean($.NSFileManager.defaultManager.fileExistsAtPath(argv[0])),
            nil: versions.isNil(),
            unresolved_count: versions.isNil() ? null : Number(versions.count)
        });
    }'''
    result=json.loads(subprocess.check_output(["/usr/bin/osascript","-l","JavaScript","-e",script,str(DEST)],
                                             text=True,timeout=20))
    after=DEST.stat()
    if (not result["exists"] or result["nil"] or not isinstance(result["unresolved_count"],int)
        or before.st_size != archive["bytes"] or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns):
        raise ValueError("Native archive conflict query/path could not be verified")
    return {**result,"at":time(),"path":str(DEST),"bytes":after.st_size,"mtime_ns":after.st_mtime_ns}


def native_status():
    """Record one guarded native snapshot; the owner will accept the upload."""
    swap_limit=admitted_swap_limit(OPERATION, COMMAND, READMISSION)
    if campaign.read(campaign.OUT/"operations"/OPERATION/"receipt.json")["status"] != "complete":
        raise ValueError("Successful local archive operation required")
    archive=campaign.read(campaign.OUT/"archive-receipt.json")
    if archive["archive"] != str(DEST) or archive["bytes"] != DEST.stat().st_size:
        raise ValueError("Verified native archive path/size differs")
    swap0=campaign.read(campaign.OUT/"baseline.json")["host"]["swap_bytes"]
    started=time(); failure=None; native=None
    directory=campaign.OUT/"operations/archive-upload-status"; directory.mkdir()
    try:
        parent=psutil.Process(os.getpid());rss=0
        for process in [parent,*parent.children(recursive=True)]:
            try:
                rss+=process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        sample=campaign.host()
        campaign.write(directory/"resource-snapshot.json",{"at":time(),"family_rss_bytes":rss,
                       "swap_growth_bytes":sample["swap_bytes"]-swap0,**sample})
        breach=campaign.limits(sample,swap0,rss,swap_limit=swap_limit)
        if breach or rss >= campaign.FAMILY_SOFT:
            raise RuntimeError("Archive upload guard breach: "+str(breach or "soft family RSS"))
        raw=subprocess.check_output(["fileproviderctl","evaluate",str(DEST)],text=True,timeout=20)
        (directory/"native.txt").write_text(raw)
        def flag(key):
            found=re.search(r"\b"+key+r"\s*=\s*(\d+)\s*;",raw)
            return int(found[1]) if found else None
        if re.search(r"\buploadingError\s*=",raw):
            raise RuntimeError("Native provider reported uploadingError")
        uploaded=flag("isUploaded"); uploading=flag("isUploading");size=flag("documentSize")
        if uploaded not in (0,1) or uploading not in (0,1) or size != archive["bytes"]:
            raise ValueError("Native upload state/path/size unreadable")
        conflicts=archive_conflicts(archive)
        if conflicts["unresolved_count"] != 0:
            raise ValueError("Unresolved archive file versions")
        actual_id=None
        if uploaded==1 and uploading==0:
            actual_id=subprocess.check_output(["xattr","-p","com.google.drivefs.item-id#S",str(DEST)],text=True).strip()
        native={"status":"native-uploaded" if uploaded==1 and uploading==0 else "upload-pending",
                "actual_drive_id":actual_id,"is_uploaded":bool(uploaded),"is_uploading":bool(uploading),
                "unresolved_conflict_versions":conflicts,"reported_uploading_error":False,
                "bytes":size,"name":DEST.name,"at":time(),"raw_receipt":str(directory/"native.txt"),
                "owner_instruction":"You just have to start the drive upload, don’t wait for it to finish I will delete the originals as soon as it’s uploaded not before making sure the hashes match",
                "cloud_acceptance":"delegated to owner; not claimed","originals_retained":True}
        campaign.write(campaign.OUT/"native-upload-status.json",native)
    except BaseException as exc:
        failure=repr(exc)
        campaign.write(directory/"failure.json",{"failure":failure,"at":time()})
        raise
    finally:
        campaign.write(directory/"receipt.json",{"status":"failed" if failure else "complete",
            "failure":failure,"started":started,"finished":time(),"seconds":time()-started,
            "samples":1,"snapshot_only":True,
            "swap_growth_limit_bytes":swap_limit,"absolute_swap_ceiling_bytes":3_000_000_000,
            "archive_sha256":archive["sha256"],"native_status":native})


def main():
    if sys.argv[1:] == ["--pack"]:
        pack()
    elif sys.argv[1:] == ["--native-status"]:
        native_status()
    elif not sys.argv[1:]:
        campaign.clean_source()
        try:
            campaign.operation(OPERATION, COMMAND, archive_readmission=READMISSION)
        except BaseException as exc:
            path = campaign.OUT/"operations"/OPERATION/"retry-failure.json"
            if not path.exists():
                campaign.write(path, {"failure":repr(exc), "at":time()})
            raise
    else:
        raise ValueError("Archive-only command required")


if __name__ == "__main__":
    main()
