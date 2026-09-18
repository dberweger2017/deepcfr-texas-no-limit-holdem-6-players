"""Post-hoc checkpoint comparisons from verified M4 outcomes; no training."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

from src.arena.report import estimate
from scripts.report_holdem_branching import decision_counts


def analyze(root, inventory_path):
    inventory = json.loads(inventory_path.read_text())
    hashes = {r['path']: r['sha256'] for r in inventory['files'] if r['group'] == 'campaign'}
    results = []
    for seed in (2026091802, 2026091803):
        for suite in ('styles', 'random'):
            snapshots = {}
            control = None
            for iteration in (64, 128, 192, 256):
                suffix = '-random' if suite == 'random' else ''
                relative = f'seed-{seed}/scenario-0-seed-{seed}/outcomes-{iteration}{suffix}.json'
                raw = (root / relative).read_bytes()
                if hashlib.sha256(raw).hexdigest() != hashes[relative]:
                    raise ValueError(f'Artifact mismatch: {relative}')
                rows = json.loads(raw)
                del raw
                expected = {(arm, block, rotation) for arm in ('candidate', 'baseline')
                            for block in range(1024) for rotation in range(6)}
                keys = [(r['arm'], r['block'], r['rotation']) for r in rows]
                if len(keys) != len(expected) or set(keys) != expected or any(
                    r['status'] != 'completed' or r['big_blind'] != 2 or r['hand'] != 0
                    for r in rows
                ):
                    raise ValueError('Unexpected hand roster')
                baseline = sorted((r for r in rows if r['arm'] == 'baseline'),
                                  key=lambda r: (r['block'], r['rotation']))
                fingerprint = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
                if control is not None and fingerprint != control:
                    raise ValueError('Baseline outcomes differ between checkpoints')
                control = fingerprint
                blocks = defaultdict(list)
                for row in rows:
                    if row['arm'] == 'candidate':
                        blocks[row['block']].append(100 * row['candidate_chips'] / row['big_blind'])
                rates = [mean(blocks[b]) for b in range(1024)]
                counts = decision_counts(rows)
                snapshots[iteration] = {'iteration': iteration, 'profit': estimate(rates),
                                        'behavior': counts, 'candidate_hands': 6144,
                                        'block_rates_bb100': rates}
                del rows, baseline
            comparisons = []
            for earlier, later in ((64, 256), (128, 256), (192, 256)):
                difference = [b-a for a,b in zip(snapshots[earlier]['block_rates_bb100'],
                                                 snapshots[later]['block_rates_bb100'], strict=True)]
                comparisons.append({'earlier': earlier, 'later': later,
                                    'paired_difference': estimate(difference)})
            results.append({'seed': seed, 'suite': suite, 'unchanged_control_sha256': control,
                            'checkpoints': list(snapshots.values()), 'duration_comparisons': comparisons})
    return {'purpose': 'Exploratory post-hoc duration and behavior analysis; no selection or confirmation',
            'intervals': 'Nominal 95%, deal-block paired; repeated comparisons not multiplicity adjusted',
            'source_inventory': str(inventory_path), 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/m4-fullgame-completed'))
    parser.add_argument('--inventory', type=Path, default=Path('docs/reports/holdem-local-fullgame-artifacts.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.inventory)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['results']:
        print(row['seed'], row['suite'], row['duration_comparisons'])
        for c in row['checkpoints']:
            b=c['behavior']
            print(c['iteration'], 'preflop all-in %', 100*b['all_ins_by_street'].get('preflop',0)/6144,
                  'postflop hand %',100*b['hands_with_postflop_decision']/6144)


if __name__ == '__main__':
    main()
