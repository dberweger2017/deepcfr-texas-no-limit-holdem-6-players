"""Local artifact paths and streaming checksums without research dependencies."""

from hashlib import sha256
from pathlib import Path


def local_cache(path: Path | str) -> Path:
    """Derived play files must not be built inside synced research storage."""
    resolved = Path(path).expanduser().resolve()
    synced = Path.home() / 'Local/Research-Cloud'
    if resolved.is_relative_to(synced.resolve()) or any(
            'GoogleDrive' in part or part in ('CloudStorage', 'Mobile Documents') for part in resolved.parts):
        raise ValueError('Policy cache must be outside synced storage')
    return resolved


def file_hash(path: Path | str) -> str:
    result = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()
