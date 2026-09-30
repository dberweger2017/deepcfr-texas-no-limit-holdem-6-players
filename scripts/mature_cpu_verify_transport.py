"""One guarded M4 transport/hash/parity phase, after the Linux logs close."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from scripts.mature_cpu_rental_guard import write


def execute(args):
    runtime = Path('/Users/dberweger/Local/trainer-observation-reuse-20260930')
    sys.path.insert(0, str(runtime))
    from scripts.hu20_scaling_supervise import run
    folder = args.folder.resolve()
    archive = folder / 'results.tar'
    expected = (folder / 'results.tar.sha256').read_text().split()[0]
    digest = sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            if time.time() >= args.deadline or shutil.disk_usage(folder).free < 8 * 2**30:
                raise RuntimeError('Original cutoff or disk guard during retrieval verification')
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError('Linux/M4 archive transport SHA differs')
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            target = folder / member.name
            if (member.issym() or member.islnk() or not target.resolve().is_relative_to(folder)
                    or not (member.isfile() or member.isdir())):
                raise ValueError('Unexpected archive member')
        source.extractall(folder, filter='data')
    deadline = min(args.deadline, time.time() + 600)
    command = [sys.executable, '-m', 'scripts.verify_mature_cpu_linux', '--root',
               str(folder / 'results'), '--reference', str(args.reference),
               '--out', str(folder / 'platform-verification.json')]
    result = run([{'name': 'independent-platform-verification', 'command': command}],
                 folder / 'm4-verification-supervisor', deadline, require_ac=True)
    write(folder / 'transport-verification.json', {'passed': result['status'] == 'complete',
          'archive_sha256': expected, 'archive_bytes': archive.stat().st_size,
          'finished': time.time(), 'deadline': deadline})
    return result['status'] == 'complete'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    raise SystemExit(0 if execute(parser.parse_args()) else 1)
