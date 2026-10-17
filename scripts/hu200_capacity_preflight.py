"""Measure an unchanged qualified table and independently verify its no-step save."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import subprocess
import time

from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.diagnostics.cfr_average import audit
from src.policies.files import file_hash

QUALIFIED = Path('/home/dberweger/cygnus-qualification-20261014')
BINARY_SHA = 'd7049a7a024f142d1a82f75d3c673e427a80c516afd9c00b7769265dfad16436'
PREFIX_SHA = '2c4c1484b7e4ebefca2bb364223d6cd12e886f98b20784b51ff6ac6678f4312d'


def put(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def verify_rows(original, saved, *, expected_entries=69_084_815, old_cap=85_373_862, new_cap=301_153_954):
    counts = Counter()
    with gzip.open(original, 'rb') as left, gzip.open(saved, 'rb') as right:
        before, after = json.loads(next(left)), json.loads(next(right))
        if before['config'].pop('max_entries') != old_cap:
            raise ValueError('Old cap differs')
        if after['config'].pop('max_entries') != new_cap:
            raise ValueError('New cap differs')
        if before != after:
            raise ValueError('Non-cap header difference')
        for a, b in zip(left, right, strict=True):
            if a != b:
                raise ValueError('Table row bytes differ')
            row = json.loads(a)
            counts['entries'] += 1
            counts['action_slots'] += len(row[1])
            counts['zero_visits'] += row[4] == 0
            counts['visits'] += row[4]
            counts['regret_free_slots'] += len(row[1]) if all(value == 0 for value in row[2]) else 0
            counts['menu_' + str(len(row[1]))] += 1
    if counts['entries'] != expected_entries:
        raise ValueError('Wrong prefix cardinality')
    return dict(counts)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    binary = QUALIFIED / 'trainer-first'
    original = QUALIFIED / 'benchmark-results/500m.gz'
    tables = QUALIFIED / 'tables'
    if file_hash(binary) != BINARY_SHA or file_hash(original) != PREFIX_SHA:
        raise ValueError('Qualified binary or original checkpoint differs')
    table_pins = {street: file_hash(tables / (street + '-k50.bin')) for street in EQUITY_K50_SHA256}
    if table_pins != EQUITY_K50_SHA256:
        raise ValueError('K50 table hashes differ')
    source_stat = original.stat()
    put(out / 'inputs.json', dict(binary_sha256=BINARY_SHA, original_sha256=PREFIX_SHA,
        original_path=str(original), original_bytes=source_stat.st_size,
        original_mtime_ns=source_stat.st_mtime_ns, table_hashes=table_pins,
        note='Read-only merged PR285 input; no #292 paths used'))
    saved = out / 'memory-500m.gz'
    telemetry = out / 'telemetry.jsonl'
    # The saved iteration is already the iteration limit: zero traversal steps.
    command = [str(binary), 'train', '--stack-bb', '200', '--menu', 'v1',
        '--seed', '2026101302', '--roots-per-seat', '1', '--average-rule', 'opponent-sampled',
        '--card-buckets', str(tables), '--resume', str(original), '--resume-sha256', PREFIX_SHA,
        '--max-entries', '301153954', '--freeze-at-cap', '--recovery',
        '--iterations', '321139', '--nodes', '500000141', '--max-seconds', '1200',
        '--out', str(saved), '--telemetry', str(telemetry),
        '--stop-file', str(out / 'stop.request')]
    put(out / 'command.json', command)
    started = time.monotonic()
    code = subprocess.run(command).returncode
    # The iteration limit ends normally even though the next-node target is unreached.
    if code != 0:
        raise ValueError(f'No-step native save returned {code}, expected0')
    record = json.loads(telemetry.read_text())
    if record['completed_nodes'] != 500_000_140 or record['iteration'] != 321139:
        raise ValueError('Unexpected training step')
    if saved.stat().st_size != record['checkpoint_bytes'] or file_hash(saved) != record['checkpoint_sha256']:
        raise ValueError('Post-write size/SHA differs from native receipt')
    counts = verify_rows(original, saved)
    put(out / 'row-identity.json', dict(status='verified', counts=counts,
        allowed_header_difference='config.max_entries only', all_row_bytes_equal=True,
        original_sha256=PREFIX_SHA, saved_sha256=record['checkpoint_sha256'],
        native=record, elapsed_seconds=time.monotonic() - started))
    current, average = out / 'memory-500m.current.gz', out / 'memory-500m.average.gz'
    subprocess.run([str(binary), 'export', str(saved), '--current', str(current),
                    '--average', str(average), '--zero-mass', 'uniform'], check=True)
    spec = dict(name='capacity-memory', seed=2026101302, iteration=321139,
        checkpoint_sha256=record['checkpoint_sha256'], sha256=file_hash(current))
    checked = audit(saved, current, average, spec, file_hash(average), expected_schema=HU200_EQUITY_SCHEMA)
    if checked['all_nodes_verified'] != counts['entries']:
        raise ValueError('Incomplete every-row audit')
    after_stat = original.stat()
    if (after_stat.st_size, after_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns) or file_hash(original) != PREFIX_SHA:
        raise ValueError('Original input changed')
    put(out / 'audit.json', dict(status='verified', audit=checked,
        files={path.name: dict(bytes=path.stat().st_size, sha256=file_hash(path))
               for path in (saved, current, average)}, elapsed_seconds=time.monotonic() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out)


if __name__ == '__main__':
    main()
