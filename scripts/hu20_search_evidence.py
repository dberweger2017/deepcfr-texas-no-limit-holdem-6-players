"""Hash-before-delete parity profiles and lossless bounded evidence retrieval.

Existing Mac evidence is never a deletion target. Finalization requires an
explicit owned-pod root and happens only after full parity comparison.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from scripts.hu20_search_arena_control import durable_json


def file_hash(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            sha.update(block)
    return sha.hexdigest()


def finalize_profiles(root, owned_pod_root):
    from src.blueprint.hu20_turn_solver import sampled_profile, PROFILE_RETENTION_RULE
    root, owned_pod_root = root.resolve(), owned_pod_root.resolve()
    if not (root == owned_pod_root or owned_pod_root in root.parents) or owned_pod_root == Path("/"):
        raise ValueError("Finalization must stay inside the explicitly owned pod directory")
    rows = []
    for profile in sorted(root.rglob("profile.jsonl")):
        request = profile.parent / "request.json"
        request_sha = file_hash(request)
        selected = sampled_profile(request_sha)
        row = {"rule": PROFILE_RETENTION_RULE, "request_sha256": request_sha,
               "profile_sha256": file_hash(profile), "profile_bytes": profile.stat().st_size,
               "sample_selected": selected, "body_retained": selected,
               "profile_generated": True}
        # The durable receipt is the deletion authorization, including partial bodies.
        durable_json(profile.parent / "profile-retention.json", row)
        if not selected:
            profile.unlink()
        rows.append({"profile": str(profile.relative_to(root)), **row})
    return rows


def verify_retention(root):
    from src.blueprint.hu20_turn_solver import sampled_profile, PROFILE_RETENTION_RULE
    counts = {"requests": 0, "profile_bodies": 0, "deleted_profiles": 0, "absent_profiles": 0}
    for request in root.rglob("request.json"):
        folder = request.parent
        sidecar = folder / "profile-retention.json"
        receipt = folder / "receipt.json"
        if sidecar.exists():
            rule = json.loads(sidecar.read_text())
        elif receipt.exists():
            rule = json.loads(receipt.read_text()).get("profile_retention")
        else:
            continue  # Non-solve metadata request, or unchanged reference without a generated profile.
        if rule is None:
            if (folder / "profile.jsonl").exists():
                raise ValueError("Solve with a profile lacks retention authorization")
            continue
        counts["requests"] += 1
        request_sha = file_hash(request)
        selected = sampled_profile(request_sha)
        if rule["rule"] != PROFILE_RETENTION_RULE or rule["request_sha256"] != request_sha or rule["sample_selected"] != selected:
            raise ValueError("Retention identity differs")
        profile = folder / "profile.jsonl"
        generated = rule["profile_generated"]
        if profile.exists() != (selected and generated):
            raise ValueError("Profile body differs from the fixed retention rule")
        if profile.exists():
            if file_hash(profile) != rule["profile_sha256"] or profile.stat().st_size != rule["profile_bytes"]:
                raise ValueError("Retained profile hash/size differs")
            counts["profile_bodies"] += 1
        elif generated:
            if not rule["profile_sha256"] or rule["profile_bytes"] < 0:
                raise ValueError("Deleted profile lacks hash/size")
            counts["deleted_profiles"] += 1
        else:
            if rule["profile_sha256"] is not None or rule["profile_bytes"] != 0:
                raise ValueError("Absent profile metadata differs")
            counts["absent_profiles"] += 1
    return counts


def pack(root, out, *, limit=10**9):
    root, out = root.resolve(), out.resolve()
    if out == root or root in out.parents:
        raise ValueError("Archive output must be outside the evidence tree")
    out.mkdir(parents=True, exist_ok=False)
    members, archives, current = {}, [], None
    try:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.is_symlink():
                raise ValueError("Evidence contains a symlink")
            if current is None:
                archive = out / f"evidence-{len(archives):03}.tar.gz"
                current = tarfile.open(archive, "w:gz", compresslevel=6)
                names = []
            name = str(path.relative_to(root))
            members[name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
            current.add(path, arcname=name, recursive=False)
            names.append(name)
            if archive.stat().st_size >= limit * .8:
                current.close()
                current = None
                if archive.stat().st_size > limit:
                    raise ValueError("Archive chunk exceeds predeclared transfer cap")
                archives.append({"path": archive.name, "bytes": archive.stat().st_size,
                                 "sha256": file_hash(archive), "members": names})
        if current is not None:
            current.close()
            current = None
            if archive.stat().st_size > limit:
                raise ValueError("Archive chunk exceeds predeclared transfer cap")
            archives.append({"path": archive.name, "bytes": archive.stat().st_size,
                             "sha256": file_hash(archive), "members": names})
        manifest = {"members": members, "archives": archives, "raw_bytes": sum(m["bytes"] for m in members.values())}
        durable_json(out / "manifest.json", manifest)
        return manifest
    finally:
        if current is not None:
            current.close()


def verify_archives(out):
    manifest = json.loads((out / "manifest.json").read_text())
    seen = set()
    for row in manifest["archives"]:
        path = out / row["path"]
        if path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            raise ValueError("Retrieved archive hash/size differs")
        names = []
        with tarfile.open(path, "r|gz") as stream:
            for member in stream:
                if not member.isfile() or member.name in seen:
                    raise ValueError("Unexpected/duplicate archive member")
                source = stream.extractfile(member)
                sha, n = hashlib.sha256(), 0
                for block in iter(lambda: source.read(1024**2), b""):
                    sha.update(block)
                    n += len(block)
                if {"bytes": n, "sha256": sha.hexdigest()} != manifest["members"][member.name]:
                    raise ValueError("Retrieved member hash/size differs")
                seen.add(member.name)
                names.append(member.name)
        if names != row["members"]:
            raise ValueError("Archive member list differs")
    if seen != set(manifest["members"]):
        raise ValueError("Evidence retrieval is incomplete")
    return {"files": len(seen), "archives": len(manifest["archives"]), "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("root", type=Path)
    finalize.add_argument("--owned-pod-root", type=Path, required=True)
    check = sub.add_parser("retention-check")
    check.add_argument("root", type=Path)
    archive = sub.add_parser("pack")
    archive.add_argument("root", type=Path)
    archive.add_argument("out", type=Path)
    verify = sub.add_parser("verify-archives")
    verify.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.op == "finalize":
        result = finalize_profiles(args.root, args.owned_pod_root)
    elif args.op == "retention-check":
        result = verify_retention(args.root)
    elif args.op == "pack":
        result = pack(args.root, args.out)
    else:
        result = verify_archives(args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
