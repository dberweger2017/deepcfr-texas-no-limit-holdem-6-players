"""Export verified diagnostic JSON to small, inspectable report tables."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
from statistics import mean, median

from scripts.diagnose_native_hu100 import rows


def export(root, out):
    if json.loads((root / 'analysis/verification.json').read_text())['status'] != 'verified':
        raise ValueError('Verify diagnostics before exporting')
    out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((root / 'analysis/summary.json').read_text())
    cells = json.loads((root / 'analysis/cells.json').read_text())
    fields = ['nodes', 'opponent', 'dimensions', 'street', 'position', 'pot_band',
              'lookup', 'visit_band', 'mass_band', 'decisions', 'covered_decisions',
              'distinct_hands', 'wins', 'losses', 'ties', 'mean_final_net_bb_per_decision',
              'mean_mass', 'mean_visits', 'mean_fold_probability', 'mean_passive_probability', 'mean_raise_probability']
    with (out / 'decision-cells.csv').open('x') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader()
        for c in cells:
            writer.writerow({**c, 'dimensions': '|'.join(c['dimensions'])})
    with (out / 'hand-exposures.csv').open('x') as f:
        writer = csv.DictWriter(f, ['nodes', 'opponent', 'exposure', 'hands', 'net_bb',
                                   'mean_net_bb', 'contribution_bb_per_100'])
        writer.writeheader()
        for p in summary['panels']:
            for exposure, values in p['hand_exposures'].items():
                writer.writerow({'nodes': p['nodes'], 'opponent': p['opponent'], 'exposure': exposure, **values})
    with (out / 'missing-support.csv').open('x') as f:
        writer = csv.DictWriter(f, ['nodes', 'opponent', 'support', 'decisions']); writer.writeheader()
        for p in summary['panels']:
            for support, count in p['missing_support'].items():
                writer.writerow({'nodes': p['nodes'], 'opponent': p['opponent'], 'support': support, 'decisions': count})
    covered = defaultdict(list)
    for r in rows(root / 'analysis/features.jsonl.gz'):
        if r['nodes'] == 11042440 and r['lookup'] == 'positive-mass-known-key':
            covered[r['opponent'], r['street']].append(r)
    readout = []
    for (op, street), selected in sorted(covered.items()):
        readout.append({'opponent': op, 'street': street, 'decisions': len(selected),
            'median_visits': median(r['visits'] for r in selected),
            'visits_lt10_pct': 100 * mean(r['visits'] < 10 for r in selected),
            'median_mass': median(r['mass'] for r in selected),
            'raise_pct': 100 * mean(r['raise_probability'] for r in selected),
            'payoff_bb_per_decision': mean(r['net_bb'] for r in selected)})
    with (out / 'covered-streets.csv').open('x') as f:
        writer = csv.DictWriter(f, list(readout[0])); writer.writeheader(); writer.writerows(readout)
    shutil.copyfile(root / 'failure-examples.json', out / 'failure-examples.json')
    for name in ('budget.json', 'frozen-analysis.json', 'source-review.json', 'native-binary.json',
                 'retrieval-transport-correction.json', 'upstream-pr200-status.json'):
        shutil.copyfile(root / name, out / name)
    shutil.copyfile(root / 'analysis/summary.json', out / 'scientific-summary.json')
    shutil.copyfile(root / 'analysis/verification.json', out / 'verification.json')
    resources = []
    for p in root.glob('guard-*/resources.jsonl'):
        resources.extend(json.loads(line) for line in p.read_text().splitlines())
    snapshot = {'samples': len(resources),
                'sampled_peak_family_rss_bytes': max(x['aggregate_job_rss_bytes'] for x in resources),
                'maximum_swap_growth_bytes': max(x['swap_growth_bytes'] for x in resources),
                'minimum_free_disk_bytes': min(x['free_disk_bytes'] for x in resources),
                'minimum_system_free_percent': min(x['system_memory']['free_percent'] for x in resources),
                'all_normal_pressure': all(x['system_memory']['pressure_level'] == 1 for x in resources),
                'all_ac': all('AC Power' in x['power'] for x in resources),
                'scope': 'sampled five-second family/system measurements, not transient peak proof'}
    (out / 'resources.json').write_text(json.dumps(snapshot, indent=2, sort_keys=True) + '\n')
    (out / 'native-parity.txt').write_text((root / 'guard-analysis/full-native.log').read_text())
    (out / 'qualification.txt').write_text((root / 'guard-analysis/focused-tests.log').read_text()
                                         + (root / 'guard-analysis/artifacts.log').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); args = p.parse_args()
    export(args.root, args.out)
