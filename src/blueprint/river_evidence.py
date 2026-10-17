"""Outcome-independent retention of complete river solver evidence."""

import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

COMPACT_RETENTION_RULE = "hu200-scientific-request-lowest-1pct-v1"


def scientific_request(request):
    return {key: value for key, value in request.items()
            if key not in ("mode", "dump_path", "seconds", "profile_format")}


def sample_key(request):
    payload = json.dumps(scientific_request(request), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return sha256(b"hu200-river-native-body-v1\0" + payload).hexdigest()


def selected_body(key):
    if len(key) != 64:
        raise ValueError("Sample key must contain 64 hexadecimal digits")
    return int(key, 16) < 2**256 // 100


def file_pin(path):
    hasher = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            hasher.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=hasher.hexdigest())


def seal_evidence(directory, request, record, files):
    """Durably record every generated hash before removing successful scratch.

    Parsed policy arrays own their values. Retention never modifies a matrix or
    request sent to the solver. A failed solve retains every generated file.
    """
    directory = Path(directory)
    files = dict(files)
    key = sample_key(request)
    if record.get("retention_sample_key", key) != key:
        raise ValueError("Scientific request changed after sample selection")
    sampled = selected_body(key)
    success = record.get("status") == "completed"
    body = {name: pin for name, pin in files.items()
            if name in ("profile.jsonl", "profile.f64")}
    original = files["request.json"]
    if record.get("request_sha256", original["sha256"]) != original["sha256"]:
        raise ValueError("Request changed after process launch")
    record["profile_retention"] = dict(rule=COMPACT_RETENTION_RULE,
        sample_key=key, sample_selected=sampled, failure_kept=not success,
        body_retained=bool(body) and (sampled or not success), generated_files=body,
        request_uncompressed=original)
    removed = []
    if success:
        compressed = directory / "request.json.gz"
        with compressed.open("xb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target,
                               compresslevel=1, mtime=0) as zipped:
                with (directory / "request.json").open("rb") as source:
                    shutil.copyfileobj(source, zipped, 1 << 20)
            target.flush()
            os.fsync(target.fileno())
        files[compressed.name] = file_pin(compressed)
        record["profile_retention"]["request_stored"] = compressed.name
        removed.append("request.json")
        if not sampled:
            removed.extend(body)
    else:
        record["profile_retention"]["request_stored"] = "request.json"
    # A durable receipt must precede scratch deletion, including both compact
    # body files. Preserve failed/partial bodies irrespective of sample choice.
    with (directory / "receipt.json").open("w") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
        for name in removed:
            (directory / name).unlink()
            files.pop(name)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return files


def read_request(directory):
    directory = Path(directory)
    raw = directory / "request.json"
    if raw.exists():
        return json.loads(raw.read_text())
    with gzip.open(directory / "request.json.gz", "rb") as stream:
        payload = stream.read()
    receipt = json.loads((directory / "receipt.json").read_text())
    expected = receipt["profile_retention"]["request_uncompressed"]
    if len(payload) != expected["bytes"] or sha256(payload).hexdigest() != expected["sha256"]:
        raise ValueError("Decompressed request checksum differs")
    return json.loads(payload)
