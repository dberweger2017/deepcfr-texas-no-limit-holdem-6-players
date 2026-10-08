"""Independently recompute paired checkpoint statistics from audited raw chip rows."""

import argparse
from collections import defaultdict
import json
from math import isclose, sqrt
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t

from scripts.evaluate_native_hu100_baseline import OPPONENTS, make_plan
from src.arena.artifacts import write_json
from src.arena.schedule import build_schedule, canonical, digest
from src.policies.files import file_hash


def interval(values, alpha=.05):
    center = mean(values)
    margin = float(t.ppf(1 - alpha / 2, len(values) - 1)) * stdev(values) / sqrt(len(values))
    return {'bb_per_100': center, 'interval': [center - margin, center + margin],
            'blocks': len(values), 'alpha': alpha}


def frozen_schedule(settings, blocks, root):
    from scripts.evaluate_native_hu100_baseline import action_seed
    single = {**settings, 'model': settings['models'][0]}
    panels = {}
    deal_seeds, action_seeds = set(), set()
    for opponent in OPPONENTS:
        schedule = build_schedule(make_plan(single, opponent, blocks, root))
        streams = []
        for b in schedule:
            if b.deal_seeds[0] in deal_seeds:
                raise ValueError('Deal collision')
            deal_seeds.add(b.deal_seeds[0])
            for r in (0, 1):
                for arm in ('candidate', 'baseline'):
                    for role in (0, 1):
                        seed = action_seed(root, b, r, arm, role)
                        if seed in action_seeds:
                            raise ValueError('Action stream collision')
                        action_seeds.add(seed)
                        streams.append([b.index, r, arm, role, seed])
        panels[opponent] = {'blocks': [vars_block(b) for b in schedule], 'private_streams': streams}
    return {'root': root, 'blocks_per_opponent': blocks, 'panels': panels,
            'pairing': 'candidate streams identical across checkpoints; baseline distinct; logical player and seat rotation fixed'}


def vars_block(block):
    from dataclasses import asdict
    return asdict(block)


def summarize(run, settings, output):
    frozen = json.loads((run / 'frozen-final.json').read_text())
    blocks, root = frozen['blocks_per_opponent'], frozen['final_root']
    schedule_path = run / 'frozen-schedule.json'
    expected_schedule = frozen_schedule(settings, blocks, root)
    if (file_hash(schedule_path) != frozen['schedule_sha256']
            or canonical(json.loads(schedule_path.read_text())) != canonical(expected_schedule)):
        raise ValueError('Global frozen schedule changed')
    results, curves, contrasts, coverage = {}, [], [], []
    first = settings['models'][0]['actual_nodes']
    source = frozen['source']
    baseline_rows = {}
    hands_replayed = actions_replayed = 0
    for spec in settings['models']:
        nodes = spec['actual_nodes']
        target = run / 'final' / str(nodes)
        audit_path = run / f'final-{nodes}-audit.json'
        a = json.loads(audit_path.read_text())
        if a['status'] != 'verified':
            raise ValueError('Unverified checkpoint')
        hands_replayed += a['hands_replayed']; actions_replayed += a['actions_replayed']
        repeat = json.loads((run / 'final-reproduction' / str(nodes) / 'complete.json').read_text())
        if not repeat['reproduced_all_hands_and_decisions']:
            raise ValueError('Unreproduced checkpoint')
        inputs = json.loads((target / 'inputs.json').read_text())
        if (inputs['source'] != source or inputs['config']['model'] != spec
                or inputs['root'] != root or inputs['blocks_per_opponent'] != blocks):
            raise ValueError('Checkpoint/run identity differs')
        for opponent in OPPONENTS:
            panel = target / opponent
            manifest = json.loads((panel / 'manifest.json').read_text())
            if (manifest['revision'] != source or manifest['dirty']
                    or manifest['plan']['models'][0]['sha256'] != spec['sha256']):
                raise ValueError('Changed/dirty source or model')
            actual_blocks = json.loads((panel / 'schedule.json').read_text())['blocks']
            if canonical(actual_blocks) != canonical(expected_schedule['panels'][opponent]['blocks']):
                raise ValueError('Unpaired physical deals')
            values = defaultdict(lambda: defaultdict(dict))
            baseline = []
            with (panel / 'hands.jsonl').open() as f:
                for line in f:
                    row = json.loads(line)
                    if row['arm'] == 'baseline':
                        baseline.append(row)
                    cell = values[row['arm']][row['block']]
                    if row['rotation'] in cell or row['status'] != 'completed':
                        raise ValueError('Duplicate/failed coordinate')
                    cell[row['rotation']] = row['candidate_chips']
            for arm in ('candidate', 'baseline'):
                if set(values[arm]) != set(range(blocks)) or any(set(v) != {0, 1} for v in values[arm].values()):
                    raise ValueError('Missing paired coordinates')
            if nodes == first:
                baseline_rows[opponent] = baseline
            elif baseline != baseline_rows[opponent]:
                raise ValueError('Uniform rows were not reused exactly')
            # Each independent block averages two seats, then chips/100 -> BB/100.
            candidate = [sum(values['candidate'][b].values()) / 2 for b in range(blocks)]
            reference = [sum(values['baseline'][b].values()) / 2 for b in range(blocks)]
            results[nodes, opponent] = candidate
            report = json.loads((panel / 'report.json').read_text())
            for arm, series in (('candidate', candidate), ('baseline', reference)):
                calculated = interval(series)
                recorded = report['scenarios'][opponent]['comparison'][arm]
                if not isclose(calculated['bb_per_100'], recorded['bb_per_100'], abs_tol=1e-8):
                    raise ValueError('Independent policy mean differs')
            curves.append({'actual_nodes': nodes, 'opponent': opponent, 'sha256': spec['sha256'],
                           'policy': interval(candidate), 'uniform': interval(reference),
                           'policy_minus_uniform': interval([a - b for a, b in zip(candidate, reference, strict=True)]),
                           'baseline_rows_sha256': digest(baseline)})
            for g in report['diagnostics']:
                if g['logical_player'] == 0 and g['arm'] == 'candidate':
                    coverage.append({'actual_nodes': nodes, 'opponent': opponent, **g})
    final = settings['models'][-1]['actual_nodes']
    for spec in settings['models'][:-1]:
        earlier = spec['actual_nodes']
        for opponent in OPPONENTS:
            difference = [a - b for a, b in zip(results[final, opponent], results[earlier, opponent], strict=True)]
            adjusted = interval(difference, .05 / 20)
            low, high = adjusted['interval']
            contrasts.append({'final_nodes': final, 'earlier_nodes': earlier, 'opponent': opponent,
                              'descriptive_95': interval(difference), 'bonferroni_20': adjusted,
                              'formal_label': 'improvement' if low > 0 else 'decline' if high < 0 else 'inconclusive'})
    result = {'status': 'verified', 'source': source, 'frozen_final_sha256': file_hash(run / 'frozen-final.json'),
              'schedule_sha256': frozen['schedule_sha256'], 'blocks_per_opponent': blocks,
              'unique_final_hands': blocks * 60, 'replayed_rows_including_reference_copies': hands_replayed,
              'replayed_actions_including_reference_copies': actions_replayed,
              'uniform_evaluations_per_opponent': 1, 'formal_family_size': 20,
              'curves': curves, 'final_minus_earlier': contrasts, 'coverage': coverage}
    write_json(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    r = summarize(args.run, json.loads(args.config.read_text()), args.out)
    print(json.dumps({k: v for k, v in r.items() if k not in ('curves', 'coverage', 'final_minus_earlier')}))
