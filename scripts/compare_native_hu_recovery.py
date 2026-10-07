"""Full HU20 legacy recovery equivalence; metadata exclusions are explicit and bounded.

Run only under the campaign resource supervisor. Never infer legacy completed
nodes from a filename: reference telemetry must bind the retained parent hash.
"""
import argparse
from gzip import open as gzip_open
from itertools import zip_longest
import json
from pathlib import Path
import struct

from scripts.audit_native_hu_checkpoint import inspect
from scripts.prepare_native_hu_campaign import REFERENCE, SEED, file_hash
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.blueprint.average import checked_header, checked_row

PARENT_SHA = 'a5318cd586c0b170a68334e4236111faddabaf7f686c071958757db888afab47'


def exact(a, b):
    """Compare IEEE-754 bits, including signed zero, rather than tolerant probabilities."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return struct.pack('!d', a) == struct.pack('!d', b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def read_header(path):
    with gzip_open(path, 'rt') as source:
        h = json.loads(source.readline())
    checked_header(h, {'seed': SEED, 'iteration': h['iteration']}, expected_schema=HU20_UNCAPPED_SCHEMA)
    if (h.get('average_rule') != 'opponent-sampled' or 'training_options' in h
        or h['config']['roots_per_seat'] != 1 or h['table']['button'] != 0):
        raise ValueError('Recovery comparison requires the frozen native linear recipe')
    return h


def receipt(path, checkpoint):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    matches = [r for r in rows if Path(r['path']).resolve() == checkpoint.resolve()]
    if len(matches) != 1:
        raise ValueError('Exactly one successful checkpoint receipt is required')
    r = matches[0]
    if (r['status'] != 'saved' or r['checkpoint_sha256'] != file_hash(checkpoint)
        or r['checkpoint_bytes'] != checkpoint.stat().st_size
        or r['completed_nodes'] < r['requested_nodes']):
        raise ValueError('Checkpoint receipt identity/completeness differs')
    return r


def compare(reference, resumed, parent, reference_telemetry, resumed_telemetry,
            reference_current, reference_average, resumed_current, resumed_average):
    parent_hash = file_hash(parent)
    if parent_hash != PARENT_SHA:
        raise ValueError('Retained historical 500M parent differs')
    # Locate the new 500M save by its independently retained member hash, then
    # validate its bytes/receipt before trusting its lifetime node counter.
    records = [json.loads(line) for line in reference_telemetry.read_text().splitlines()]
    parent_records = [r for r in records if r['checkpoint_sha256'] == parent_hash]
    if len(parent_records) != 1:
        raise ValueError('Reference 500M save does not reproduce the retained parent')
    parent_receipt = receipt(reference_telemetry, Path(parent_records[0]['path']))
    if parent_receipt['requested_nodes'] != 500000000 or parent_receipt['iteration'] != 1095942:
        raise ValueError('Wrong retained parent endpoint')
    r, s = receipt(reference_telemetry, reference), receipt(resumed_telemetry, resumed)
    if (r['requested_nodes'] != 1000000000 or s['requested_nodes'] != 1000000000
        or r['completed_nodes'] != s['completed_nodes'] or r['iteration'] != s['iteration']):
        raise ValueError('Recovery final actual nodes/iterations differ')
    h, recovered, ph = read_header(reference), read_header(resumed), read_header(parent)
    if 'native_state' in h or 'native_state' in ph:
        raise ValueError('Reference and retained parent must preserve legacy headers')
    state = recovered.get('native_state')
    scientific = {k: v for k, v in recovered.items() if k != 'native_state'}
    if not exact(h, scientific):
        raise ValueError('Recovery changed scientific header/configuration')
    pd, rd = parent_receipt['diagnostics'], r['diagnostics']
    expected_state = {'version': 1, 'completed_nodes': r['completed_nodes'],
        'coverage_start': [ph['iteration'], parent_receipt['completed_nodes'], pd['traverser_visits']],
        'decisions_by_street': [x-y for x, y in zip(rd['decisions_by_street'], pd['decisions_by_street'], strict=True)],
        'traverser_visits_by_street': [x-y for x, y in zip(rd['traverser_visits_by_street'], pd['traverser_visits_by_street'], strict=True)]}
    if (not exact(state, expected_state) or any(v < 0 for v in expected_state['decisions_by_street'])
        or any(v < 0 for v in expected_state['traverser_visits_by_street'])
        or s['diagnostics']['coverage_start'] != expected_state['coverage_start']
        or s['diagnostics']['decisions_by_street'] != expected_state['decisions_by_street']
        or s['diagnostics']['traverser_visits_by_street'] != expected_state['traverser_visits_by_street']):
        raise ValueError('Recovery-only coverage metadata differs from reference deltas')
    entries = visits = 0
    previous = None
    with gzip_open(reference, 'rt') as left, gzip_open(resumed, 'rt') as right:
        next(left); next(right)
        for a, b in zip_longest(left, right):
            if a is None or b is None:
                raise ValueError('Recovery table length differs')
            x, y = json.loads(a), json.loads(b)
            checked_row(x, h['iteration'], False); checked_row(y, h['iteration'], False)
            if previous is not None and x[0] <= previous:
                raise ValueError('Reference keys must be unique and sorted')
            if not exact(x, y):
                raise ValueError(f'Recovery complete training row differs: {x[0]}')
            previous = x[0]; entries += 1; visits += x[4]
    if (entries != rd['entries'] or entries != s['diagnostics']['entries']
        or visits != rd['traverser_visits'] or visits != s['diagnostics']['traverser_visits']
        or visits != state['coverage_start'][2] + sum(state['traverser_visits_by_street'])):
        raise ValueError('Recovery telemetry/table counts differ')
    if file_hash(reference_current) != file_hash(resumed_current):
        raise ValueError('Current inference bytes differ')
    # These audits independently recompute every output row from training state.
    original_audit = inspect(reference, reference_current, reference_average, 20)
    recovered_audit = inspect(resumed, resumed_current, resumed_average, 20, hu20_reference=False)
    average_rows = 0
    with gzip_open(reference_average, 'rt') as left, gzip_open(resumed_average, 'rt') as right:
        lm, rm = json.loads(next(left)), json.loads(next(right))
        exclusions = {'checkpoint_header', 'source_checkpoint_sha256'}
        if (not exact({k:v for k,v in lm.items() if k not in exclusions},
                      {k:v for k,v in rm.items() if k not in exclusions})
            or not exact(lm['checkpoint_header'], h) or not exact(rm['checkpoint_header'], recovered)
            or lm['source_checkpoint_sha256'] != file_hash(reference)
            or rm['source_checkpoint_sha256'] != file_hash(resumed)):
            raise ValueError('Unexpected average metadata difference')
        for a, b in zip_longest(left, right):
            if a is None or b is None or not exact(json.loads(a), json.loads(b)):
                raise ValueError('Average probabilities/mass/visits differ')
            average_rows += 1
    if average_rows != entries:
        raise ValueError('Average export does not cover the complete training table')
    return {'status': 'verified', 'scope': 'full HU20 legacy recovery state and current/average policy equivalence',
        'seed': SEED, 'completed_nodes': r['completed_nodes'], 'iteration': h['iteration'],
        'entries': entries, 'training_float_equality': 'IEEE-754 bits, including signed zero',
        'current_equality': 'identical file bytes by SHA256; both independently audited',
        'average_equality': 'all probabilities/mass/visits exact; validated metadata differences only',
        'metadata_differences': {'checkpoint': ['native_state'],
            'average': ['source_checkpoint_sha256', 'checkpoint_header.native_state'],
            'validated_recovery_state': state},
        'reference_average_sha256': REFERENCE, 'reference_average_bytes': reference_average.stat().st_size,
        'reference_audit': original_audit, 'resumed_audit': recovered_audit,
        'files': {str(p.resolve()): {'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in
            (reference, resumed, parent, reference_telemetry, resumed_telemetry,
             reference_current, reference_average, resumed_current, resumed_average)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fields = ('reference', 'resumed', 'parent', 'reference_telemetry', 'resumed_telemetry',
              'reference_current', 'reference_average', 'resumed_current', 'resumed_average')
    for field in fields:
        parser.add_argument('--'+field.replace('_','-'), type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve prior audit evidence')
    result = compare(**{k: getattr(args, k) for k in fields})
    with args.out.open('x') as target:
        json.dump(result, target, sort_keys=True, indent=2, allow_nan=False); target.write('\n')


if __name__ == '__main__': main()
