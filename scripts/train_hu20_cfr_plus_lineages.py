"""Run the three predeclared CFR+ lineages, preserving milestones and native exports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from time import time, sleep

SEEDS = (2026100601, 2026100602, 2026100603)
MILESTONES = (100_000_000, 500_000_000, 1_000_000_000)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def guarded(command, log, resources, deadline, root):
    if log.exists() or resources.exists():
        raise FileExistsError('Preserve earlier attempts; use a fresh log path')
    with log.open('x') as out, resources.open('x') as err:
        process = subprocess.Popen(['/usr/bin/time', '-l', *map(str, command)], stdout=out, stderr=err,
                                   start_new_session=True)
        while process.poll() is None:
            if time() > deadline or shutil.disk_usage(root).free < 15 * 1024**3:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                raise RuntimeError('Training deadline or disk guard')
            sleep(1)
        if process.returncode:
            raise RuntimeError(f'Native command failed: {process.returncode}; see {log}')


def run(binary, root, seed, deadline):
    training, policies = root / 'training', root / 'policies'
    pattern = training / f'F-{seed}-{{nodes}}.json.gz'
    guarded([binary, 'train', '--nodes', '1000000000', '--milestones', '100000000,500000000',
             '--seed', seed, '--roots-per-seat', '1', '--average-rule', 'traverser-reach',
             '--regret-floor', '0', '--out', pattern], training / f'{seed}.log', training / f'{seed}.time', deadline, root)
    records = []
    for milestone in MILESTONES:
        path = training / f'F-{seed}-{milestone}.json.gz'
        with gzip.open(path, 'rt') as f:
            header = json.loads(f.readline())
            assert header['training_options'] == 'regret-floor-0' and 'average_rule' not in header
            entries = 0
            minimum = float('inf')
            for row in map(json.loads, f):
                minimum = min(minimum, *row[2])
                entries += 1
            assert minimum >= 0
        records.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha(path),
                        'milestone': milestone, 'iteration': header['iteration'], 'entries': entries,
                        'minimum_regret': minimum, 'training_options': header['training_options'], 'header': header})
    (training / f'{seed}.manifest.json').write_text(json.dumps(records, indent=2, sort_keys=True) + '\n')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    root, binary = a.out.resolve(), a.binary.resolve()
    for d in ('training', 'policies'):
        (root / d).mkdir(parents=True, exist_ok=True)
    deadline = time() + 7200
    provenance = {'source_main_commit': a.source, 'binary_sha256': sha(binary), 'binary': str(binary),
                  'seeds': list(SEEDS), 'milestones': list(MILESTONES), 'started_at': time(),
                  'deadline': deadline, 'regret_floor': 0, 'average_rule': 'traverser-reach', 'paid_compute': False}
    (root / 'training-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda seed: run(binary, root, seed, deadline), SEEDS))
    # Export sequentially: the combined current/average native export has the largest memory peak.
    exports = []
    for seed in SEEDS:
        checkpoint = root / 'training' / f'F-{seed}-1000000000.json.gz'
        current = root / 'policies' / f'C-{seed}.current.json.gz'
        average = root / 'policies' / f'O-{seed}.average.jsonl.gz'
        guarded([binary, 'export', checkpoint, '--current', current, '--average', average],
                root / 'training' / f'{seed}.export.log', root / 'training' / f'{seed}.export.time', deadline, root)
        with gzip.open(current, 'rt') as f:
            document = json.load(f)
            assert document['training_options'] == 'regret-floor-0'
            iteration = document['iteration']
            entries = len(document['entries'])
            del document
            gc.collect()
        with gzip.open(average, 'rt') as f:
            metadata = json.loads(f.readline())
            assert metadata['checkpoint_header']['training_options'] == 'regret-floor-0'
            assert metadata['source_checkpoint_sha256'] == sha(checkpoint)
            assert metadata['extraction'] == 'normalize-lifetime-iteration-own-reach-accumulator-v1'
        for path in (current, average):
            exports.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha(path),
                            'seed': seed, 'iteration': iteration, 'entries': entries, 'training_options': 'regret-floor-0'})
    result = {'status': 'complete', 'provenance': provenance, 'checkpoints': sum(records, []), 'exports': exports,
              'finished_at': time()}
    (root / 'checkpoints-manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'complete', 'checkpoints': len(result['checkpoints']), 'exports': len(exports)}), flush=True)


if __name__ == '__main__':
    main()
