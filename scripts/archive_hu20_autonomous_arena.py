"""Seal research into a new ZIP with stream-verified member hashes; preserve originals."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import time
import zipfile

from scripts.hu20_search_arena_control import durable_json
from scripts.hu20_search_evidence import file_hash

# Authentication material is operational access, never a research artifact.
SECRET_NAMES = {'control-token', 'github-token', 'runpod-credential.json'}


def archive_research(root, destination):
    root = root.resolve()
    destination = destination.resolve()
    if destination == root or root in destination.parents or destination.exists():
        raise ValueError('Use a new archive outside the research tree')
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name not in SECRET_NAMES
                   and '__pycache__' not in p.parts)
    if any(p.is_symlink() for p in paths):
        raise ValueError('Research symlink requires explicit handling')
    required = sum(p.stat().st_size for p in paths) + 1024**3
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required:
        raise OSError('Insufficient real free disk for lossless research ZIP')
    members = {}
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in paths:
            name = str(path.relative_to(root))
            members[name] = {'bytes': path.stat().st_size, 'sha256': file_hash(path)}
            archive.write(path, name)
        archive.writestr('RESEARCH_MEMBER_HASHES.json', json.dumps(members, sort_keys=True, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if set(archive.namelist()) != set(members) | {'RESEARCH_MEMBER_HASHES.json'}:
            raise ValueError('Research ZIP member list differs')
        for name, spec in members.items():
            sha, count = hashlib.sha256(), 0
            with archive.open(name) as source:
                for block in iter(lambda: source.read(1024**2), b''):
                    sha.update(block); count += len(block)
            if {'bytes': count, 'sha256': sha.hexdigest()} != spec:
                raise ValueError('Research ZIP member differs: ' + name)
    receipt = {'at': time(), 'archive': str(destination), 'bytes': destination.stat().st_size,
               'sha256': file_hash(destination), 'members': len(members), 'verified': True,
               'source': str(root), 'authentication_files_excluded': sorted(SECRET_NAMES)}
    durable_json(destination.with_suffix('.verified.json'), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive_research(args.root, args.destination), sort_keys=True))


if __name__ == '__main__':
    main()
