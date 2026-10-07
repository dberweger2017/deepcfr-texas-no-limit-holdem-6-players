"""Audit a future native checkpoint/current/average set and summarize training coverage."""
import argparse
from collections import Counter
from gzip import open as gzip_open
import json
from pathlib import Path

from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, HU100_SCHEMA
from src.diagnostics.cfr_average import audit, checked_header, checked_row, average_rule
from src.policies.files import file_hash
from scripts.prepare_native_hu_campaign import REFERENCE


def inspect(checkpoint, current, average, bb, target_nodes=None, *, hu20_reference=True):
    schema = HU100_SCHEMA if bb == 100 else HU20_UNCAPPED_SCHEMA
    with gzip_open(checkpoint, 'rt') as source:
        header = json.loads(source.readline())
    if bb == 100 and target_nodes is not None:
        state = header.get('native_state', {})
        if (state.get('completed_nodes', 0) < target_nodes
            or header.get('average_rule') != 'opponent-sampled' or 'training_options' in header
            or state.get('coverage_start') != [0,0,0]
            or len(state.get('traverser_visits_by_street', [])) != 4
            or any(type(v) is not int or v <= 0 for v in state['traverser_visits_by_street'])):
            raise ValueError('Incomplete production HU100 target/coverage; no extension admission')
    spec = {'name': 'native-hu', 'seed': header['config']['seed'], 'iteration': header['iteration'],
            'checkpoint_sha256': file_hash(checkpoint), 'sha256': file_hash(current)}
    checked_header(header, spec, expected_schema=schema)
    receipt = audit(checkpoint, current, average, spec, file_hash(average), expected_schema=schema)
    if bb == 20 and hu20_reference and (receipt['average_sha256'] != REFERENCE or average.stat().st_size != 142677367):
        raise ValueError('HU20 regression differs from the pinned v0.4.1 reference')
    visits = Counter(); positive = 0; total_visits = 0; count = 0
    with gzip_open(checkpoint, 'rt') as source:
        next(source)
        for line in source:
            _, _, _, _, mass, n = checked_row(json.loads(line), header['iteration'], average_rule(header) == 'traverser-reach')
            count += 1; total_visits += n; positive += bool(mass)
            visits['0' if n == 0 else '1' if n == 1 else '2-9' if n < 10 else '10-99' if n < 100 else '100+'] += 1
    return {'status': 'verified', 'hu20_reference_checked': bb == 20 and hu20_reference,
            'audit': receipt, 'stack_bb': bb, 'iteration': header['iteration'],
            'native_state': header.get('native_state'), 'entries': count, 'visits_histogram': dict(visits),
            'mean_traverser_visits_per_key': total_visits/count if count else 0,
            'positive_average_mass_keys': positive, 'zero_average_mass_keys': count-positive,
            'files': {str(p): {'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in (checkpoint, current, average)},
            'scope': 'resource/coverage and serialization only; no poker-strength or convergence claim'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint', 'current', 'average', 'out'): p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--stack-bb', type=int, choices=(20,100), required=True)
    p.add_argument('--target-nodes', type=int, required=True); a = p.parse_args()
    result = inspect(a.checkpoint, a.current, a.average, a.stack_bb, a.target_nodes)
    with a.out.open('x') as target: target.write(json.dumps(result, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__': main()
