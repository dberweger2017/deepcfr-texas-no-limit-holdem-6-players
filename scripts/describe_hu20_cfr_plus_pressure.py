"""Descriptive native-pressure response and whole-hand partitions, after frozen replay audit."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def situations(row):
    """Blind completion is not a response to a raise; jams use actual remaining stack."""
    street = None
    last_raise = None
    off_menu = False
    opponent_raises = 0
    responses = []
    for action in row['actions']:
        observed = action['observation']
        if action['street'] != street:
            street, last_raise = action['street'], None
        if action['logical_player'] == 1:
            if action['kind'] == 'raise':
                opponent_raises += 1
                outside = action['raise_to'] not in {
                    c['raise_to'] for c in observed['menu'] if c['kind'] == 'raise'}
                off_menu |= outside
                jam = action['raise_to'] == observed['street_bet'] + observed['stack']
                last_raise = ('jam' if jam else 'raise', outside)
            continue
        if last_raise is None or observed['call_amount'] <= 0:
            continue
        kind = action['kind']
        assert kind in ('fold', 'call', 'raise')
        paid = (min(observed['call_amount'], observed['stack']) if kind == 'call' else
                action['raise_to'] - observed['street_bet'] if kind == 'raise' else 0)
        assert 0 <= paid <= observed['stack']
        responses.append({'street': street, 'facing': last_raise[0], 'response': kind,
                          'off_menu_history': off_menu, 'last_raise_off_menu': last_raise[1],
                          'committed_chips': paid, 'mass': action['average_mass_status']})
    last = responses[-1] if responses else None
    return responses, {
        'last_response': '/'.join(last[k] for k in ('street', 'facing', 'response')) if last else 'no-facing-response',
        'last_street': last['street'] if last else 'no-facing-response',
        'off_menu': 'off-menu-seen' if off_menu else 'all-menu-sizes',
        'raise_depth': 'more-than-two-opponent-raises' if opponent_raises > 2 else 'at-most-two-opponent-raises'}


def mean_interval(values):
    n = len(values)
    mean = float(np.mean(values))
    margin = float(t.ppf(.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n))
    return {'mean': mean, 'ci95': [mean - margin, mean + margin], 'blocks': n}


def ratio_interval(numerator, denominator, scale=1):
    """Cluster ratio delta interval: each block contains all three lineages and both seats."""
    if not np.sum(denominator):
        return None
    ratio = float(np.sum(numerator) / np.sum(denominator))
    influence = (numerator - ratio * denominator) / np.mean(denominator)
    result = mean_interval(influence * scale)
    margin = result['ci95'][1] - result['mean']
    return {'mean': ratio * scale, 'ci95': [ratio * scale - margin, ratio * scale + margin],
            'blocks': len(denominator)}


def describe(root, out):
    arena = json.loads((root / 'arena/run/summary.json').read_text())
    audit = json.loads((root / 'arena/audit.json').read_text())
    assert audit['status'] == 'verified'
    n = arena['contrasts_bb_per_100']['O-R']['native-pressure']['blocks']
    decisions = defaultdict(lambda: np.zeros((n, 8)))
    partitions = defaultdict(lambda: np.zeros((n, 3)))
    files = []
    total = defaultdict(dict)
    for arm, seeds in [('R', (2026093001, 2026093002, 2026093003)),
                       ('O', (2026100601, 2026100602, 2026100603))]:
        for seed in seeds:
            path = root / f'arena/run/{arm}-{seed}.hands.jsonl.gz'
            with path.open('rb') as stream:
                files.append({'path': path.name, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()})
            with gzip.open(path, 'rt') as stream:
                for row in map(json.loads, stream):
                    if row['panel'] != 'native-pressure':
                        continue
                    block, rotation = row['block'], row['rotation']
                    assert (seed, block, rotation) not in total[arm]
                    total[arm][seed, block, rotation] = row['target_chips']
                    responses, groups = situations(row)
                    for response in responses:
                        for key in [(response['street'], response['facing'], 'all'),
                                    (response['street'], response['facing'], response['response'])]:
                            cell = decisions[arm, *key][block]
                            cell += [1, response['response'] == 'fold', response['response'] == 'call',
                                     response['response'] == 'raise', -row['target_chips'], response['committed_chips'],
                                     response['mass'] == 'missing', response['mass'] == 'zero_mass']
                    for dimension, group in groups.items():
                        partitions[arm, dimension, group][block] += [1, row['target_chips'], max(-row['target_chips'], 0)]
            print(json.dumps({'processed': path.name}), flush=True)
        assert len(total[arm]) == n * 6
    result = {'status': 'verified_descriptive_partitions', 'blocks': n, 'hands_per_arm': n * 6,
              'source_files': files, 'responses': [], 'whole_hand_partitions': [],
              'units': '100 chips = 1 BB; mean chips/hand numerically equals BB/100',
              'interval_method': 'paired deal-block Student-t; conditional ratios use cluster delta method',
              'attribution': 'Each hand appears once per partition dimension. Response-level terminal losses overlap across decisions and are not additive or causal.',
              'translation': 'No action translation exists in the frozen harness. Off-menu compares actual rival raise_to to its recorded uncapped v1 menu.'}
    for (arm, street, facing, response), cell in sorted(decisions.items()):
        base = decisions[arm, street, facing, 'all'][:, 0]
        result['responses'].append({'arm': arm, 'street': street, 'facing': facing, 'response': response,
            'observed_decisions': int(np.sum(cell[:, 0])),
            'frequency_percent': ratio_interval(cell[:, 0], base, 100),
            'mean_terminal_net_chips_lost': ratio_interval(cell[:, 4], cell[:, 0]),
            'mean_chips_committed': ratio_interval(cell[:, 5], cell[:, 0]),
            'missing_percent': ratio_interval(cell[:, 6], cell[:, 0], 100),
            'zero_mass_percent': ratio_interval(cell[:, 7], cell[:, 0], 100)})
    for dimension in ('last_response', 'last_street', 'off_menu', 'raise_depth'):
        groups = sorted({group for arm, d, group in partitions if d == dimension})
        sums = {arm: np.zeros(n) for arm in ('R', 'O')}
        for group in groups:
            cells = {arm: partitions[arm, dimension, group] for arm in ('R', 'O')}
            entry = {'dimension': dimension, 'group': group, 'arms': {}}
            for arm, cell in cells.items():
                sums[arm] += cell[:, 1] / 6
                entry['arms'][arm] = {'observed_hands': int(np.sum(cell[:, 0])),
                    'hand_share_percent': ratio_interval(cell[:, 0], np.full(n, 6), 100),
                    'mean_net_chips_lost_per_exposed_hand': ratio_interval(-cell[:, 1], cell[:, 0]),
                    'net_contribution_bb_per_100': mean_interval(cell[:, 1] / 6),
                    'gross_losses_chips': int(np.sum(cell[:, 2]))}
            entry['candidate_minus_R_contribution'] = mean_interval((cells['O'][:, 1] - cells['R'][:, 1]) / 6)
            result['whole_hand_partitions'].append(entry)
        contrast = mean_interval(sums['O'] - sums['R'])
        expected = arena['contrasts_bb_per_100']['O-R']['native-pressure']
        assert np.isclose(contrast['mean'], expected['bb_per_100'], atol=1e-9)
        assert np.allclose(contrast['ci95'], expected['ci95'], atol=1e-9)
        for arm in ('R', 'O'):
            assert np.isclose(np.mean(sums[arm]), arena['absolute_bb_per_100'][arm]['native-pressure']['bb_per_100'], atol=1e-9)
    result['partitions_reconcile_to_frozen_native_pressure'] = True
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    describe(args.root, args.out)


if __name__ == '__main__':
    main()
