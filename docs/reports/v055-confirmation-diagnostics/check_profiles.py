"""Reconcile the profile CLI and public tables with the archived analyzer JSON.

Analyzer decision summaries include prefixes from abandoned hands; completed-hand
profiles do not. Preserve those differences instead of mixing their denominators.
Analyzer size buckets also pool all streets and use a different raise-size convention.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

ROOT = Path('results/outputs')
PUBLIC = ROOT / 'public-final'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-pending', action='store_true', help='check available analyzer outputs during AIVAT')
    args = parser.parse_args()
    pending = []
    profiles = json.loads((ROOT / 'profiles.json').read_text())
    responses = list(csv.DictReader((PUBLIC / 'flop-responses.csv').open()))
    folds = list(csv.DictReader((PUBLIC / 'flop-folds.csv').open()))
    checks = []
    for name, profile in profiles.items():
        assert profile['hands'] == (30000 if name == 'confirmation-pooled' else 10000)
        for side in ('player', 'slumbot'):
            actual = profile[side]['by_street']['flop']['fold']
            assert actual == sum(int(r['hands']) for r in folds if r['run'] == name and r['folder'] == side)
            expected = Counter()
            for row in responses:
                if row['run'] == name and row['responder'] == side and row['pot'] == 'all':
                    for key in ('fold', 'call', 'raise'):
                        expected[row['size'], key] += int(row[key])
            for key, value in expected.items():
                size, response = key
                assert value == profile[side]['answers_to_bets'].get(f'flop {size}', {}).get(response, 0)
        if name == 'confirmation-pooled':
            source = Path('results/inputs/pr318/analysis/pooled-30000.json')
        elif name.startswith('confirmation-'):
            # The completed AIVAT analyzer also reports the unchanged decision counts.
            source = ROOT / f'aivat-{name}.json'
        elif name == 'exploratory':
            source = Path('results/inputs/pr316/summary-blend.json')
        else:
            source = Path(f'results/inputs/pr304/analysis/v055/seed-{name}-pooled-10000.json')
        if not source.exists():
            if args.allow_pending:
                pending.append(name)
                continue
            raise FileNotFoundError(f'Wait for all account analyzers: {source}')
        analysis = json.loads(source.read_text())
        assert abs(profile['overall']['bb_per_100'] - analysis['bb_per_100']['mean']) < 1e-10
        for street in ('preflop', 'flop', 'turn', 'river'):
            public = profile['player']['by_street'][street]
            recorded = analysis['actions_by_street'].get(street, {})
            for kind in ('fold', 'check', 'call', 'aggressive'):
                a = public['bet'] + public['raise'] if kind == 'aggressive' else public[kind]
                b = recorded.get('raise', 0) + recorded.get('all-in', 0) if kind == 'aggressive' else recorded.get(kind, 0)
                difference = b - a
                assert difference >= 0, (name, street, kind, a, b)
                if name.startswith('confirmation-'):
                    assert difference == 0
                checks.append({'run': name, 'street': street, 'kind': kind,
                               'completed_hand_actions': a, 'analyzer_decisions': b,
                               'extra_prefix_decisions': difference})
    filename = 'decision-reconciliation-pending.csv' if pending else 'decision-reconciliation.csv'
    with (ROOT / filename).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    print('Profiles and all flop opportunities/folds reconcile; available analyzer totals reconcile.')
    if pending:
        print('Pending analyzer cross-checks: ' + ', '.join(pending))


if __name__ == '__main__':
    main()
