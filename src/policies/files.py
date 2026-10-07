"""Streaming checksums without loading models or research tooling."""

from hashlib import sha256
from pathlib import Path


def file_hash(path: Path | str) -> str:
    result = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()
