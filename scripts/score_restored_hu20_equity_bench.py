"""Run #222's unchanged scoring on verified, relocated inputs; never train."""

import argparse
import json
from pathlib import Path

from scripts import run_hu20_equity_bench as bench
from src.diagnostics.flop_check import atomic_json
from src.policies.files import file_hash


def qualified_evaluator(restored, receipt):
    """Use #222's later scoring qualification, preserving its earlier setup pin."""
    qualification_path = restored/'qualified-final-source.json'
    member = next(p for p in receipt['members']
                  if p.get('path') == 'research/qualified-final-source.json')
    if (qualification_path.stat().st_size != member['bytes']
            or file_hash(qualification_path) != member['sha256']):
        raise ValueError('Archived evaluator qualification differs')
    qualification = json.loads(qualification_path.read_text())
    index = json.loads((bench.ROOT/'docs/reports/hu20-equity-bench-artifacts/model-input-index.json').read_text())
    if qualification['lock_sha256'] != index['lock_evaluator']['sha256']:
        raise ValueError('Final qualification and restoration locator differ')
    return qualification['lock_sha256']


def configure(base):
    restored = base/'restored/research'
    receipt = json.loads((base/'restoration.json').read_text())
    if receipt['status'] != 'verified':
        raise ValueError('Verified restoration required')
    pins = json.loads((restored/'input-pins.json').read_text())
    plan = json.loads(bench.PLAN.read_text())
    for key in ('corpus', 'crossfit'):
        if plan[key] != pins[key] or file_hash(plan[key]['path']) != pins[key]['sha256']:
            raise ValueError('Frozen corpus or folds differ')
    records = {r['spot']: r for r in json.loads(Path(plan['corpus']['path']).read_text())['roots']}
    folds = json.loads(Path(plan['crossfit']['path']).read_text())['folds']
    jobs = sorted(pins['jobs'], key=lambda j: j['spot'])
    if len(jobs) != 40 or {j['spot'] for j in jobs} != set(records):
        raise ValueError('Frozen 40-root set differs')
    if any(j['evaluation_fold'] != folds[j['spot']] or j['policy_index'] != 0
           or j['lineage'] != 2026093001 for j in jobs):
        raise ValueError('Frozen fold or lineage differs')
    freeze = json.loads((restored/'matched-freeze.json').read_text())
    if freeze['iterations'] != {'0': 4000977, '1': 3855889}:
        raise ValueError('Matched freeze differs')
    bench.OUT = base/'work'
    bench.LOCK = base/'restored/lock-evaluator'
    if file_hash(bench.LOCK) != qualified_evaluator(restored, receipt):
        raise ValueError('Lock evaluator differs')
    # The accepted #222 adapters replace canonical-path preparation at this
    # boundary. Only file locations change; evaluation and reporting stay shared.
    bench.inputs = lambda: (plan, records, folds, jobs)
    return restored, receipt, jobs


def prepare(base):
    restored, receipt, jobs = configure(base)
    for pin in receipt['members'] + receipt['aliases']:
        path = Path(pin['restored_path'])
        if path.stat().st_size != pin['bytes'] or file_hash(path) != pin['sha256']:
            raise ValueError('Restored member differs: '+str(path))
    bench.OUT.mkdir(exist_ok=False)
    for directory in ('runs', 'transport', 'references'):
        (bench.OUT/directory).symlink_to((restored/directory).resolve(), target_is_directory=True)
    for name in ('matched-freeze.json', 'visits.json'):
        (bench.OUT/name).symlink_to((restored/name).resolve())
    localized = []
    for job in jobs:
        source = restored/'prepared'/job['job']/'request.json'
        request = json.loads(source.read_text())
        request['compact_path'] = str((source.parent/'compact.json').resolve())
        dest = bench.OUT/'prepared'/job['job']/'request.json'
        atomic_json(dest, request)
        original = json.loads(source.read_text())
        if {k:v for k,v in request.items() if k != 'compact_path'} != {k:v for k,v in original.items() if k != 'compact_path'}:
            raise ValueError('Request behavior changed')
        localized.append({'job':job['job'], 'source_sha256':file_hash(source),
                          'localized_sha256':file_hash(dest), 'changed_fields':['compact_path']})
    for fold in (0,1):
        policies = bench.policies(bench.OUT/'runs'/f'fold-{fold}')
        if len(policies) != 21:
            raise ValueError('Expected exactly 21 frozen policies per fold')
    atomic_json(base/'localization.json', {'requests':localized, 'policies_per_fold':21,
        'science':'unchanged #222 evaluate/report functions', 'training':False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','pilot','evaluate','report'))
    parser.add_argument('--base',type=Path,default=Path('results/equity-bench-scoring-fixed-20261010'))
    args = parser.parse_args()
    base = args.base.resolve()
    if args.command == 'prepare':
        prepare(base)
    else:
        configure(base)
        if args.command == 'report':
            bench.report()
        else:
            bench.evaluate(pilot=args.command == 'pilot')


if __name__ == '__main__':
    main()
