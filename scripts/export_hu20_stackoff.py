"""Publish compact generated evidence, not model binaries or human journals."""
import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

from scripts.evaluate_hu20 import write_json
from scripts.report_hu20_stackoff import records
from src.diagnostics.saved_hu20 import file_hash

FIELDS = ('model', 'panel', 'block', 'rotation', 'index', 'logical_player', 'seat',
          'street', 'position', 'kind', 'raise_to', 'call_amount', 'pot', 'stack',
          'street_bet', 'hole_cards', 'board', 'concrete_category', 'card_bucket',
          'key', 'visits', 'trained', 'large_raise_opportunity', 'jam_opportunity',
          'large_raise_probability', 'jam_probability', 'menu', 'probabilities',
          'preceding_target_off_menu', 'on_target_menu', 'seconds')


def export(run, out):
    out.mkdir(parents=True, exist_ok=False)
    for name in ('plan.json', 'manifest.json', 'result.json', 'attempts.json', 'summary.json',
                 'contexts.json', 'inspection-summary.json', 'environment.json'):
        path = run / name
        if path.exists():
            shutil.copyfile(path, out / name)
    with gzip.open(out / 'generated-hands.jsonl.gz', 'wt') as hands, gzip.open(out / 'decisions.csv.gz', 'wt', newline='') as decisions:
        writer = csv.DictWriter(decisions, FIELDS)
        writer.writeheader()
        for row in records(run):
            for action in row['actions']:
                observed = action['observation']
                item = {**observed, **action, 'model': row['policy'],
                        **{k: row[k] for k in ('panel', 'block', 'rotation')}}
                item = {k: item.get(k) for k in FIELDS}
                for k in ('hole_cards', 'board', 'menu', 'probabilities', 'card_bucket'):
                    item[k] = json.dumps(item[k], separators=(',', ':'))
                writer.writerow(item)
            compact = {k: v for k, v in row.items() if k not in ('reached_keys', 'decision_telemetry')}
            compact['actions'] = [{k: v for k, v in action.items() if k != 'observation'} for action in row['actions']]
            hands.write(json.dumps(compact, sort_keys=True, allow_nan=False) + '\n')
    for path in sorted(run.glob('*.inspection.jsonl.gz')):
        # Every raw holding/key/menu/probability query is retained.
        shutil.copyfile(path, out / path.name)
    provenance = {'source_run': json.loads((run / 'manifest.json').read_text()),
                  'files': {p.name: {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
                            for p in sorted(out.iterdir())},
                  'generated_simulator_evidence_only': True,
                  'warning': 'Join decisions to a hand for context; repeated actions are not independent payoff samples.'}
    write_json(out / 'evidence-manifest.json', provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.run, args.out)
    print(json.dumps({'files': len(result['files']), 'bytes': sum(v['bytes'] for v in result['files'].values())}))


if __name__ == '__main__':
    main()
