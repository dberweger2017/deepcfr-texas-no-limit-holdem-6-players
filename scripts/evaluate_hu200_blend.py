"""Blinded #263 pair execution and a global barrier for the fixed three-seed blend."""
import argparse
from collections import Counter, defaultdict
import gc
import gzip
import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
import subprocess
from time import time

from scipy.stats import t

from scripts import evaluate_hu100_direct as direct
from scripts.run_hu200_frozen_evaluation import series
from src.policies.files import file_hash

SEEDS = (2026101302, 2026101421, 2026101422)
DIRECT_BLOCKS = 131072
PANEL_BLOCKS = 16384


def read(path):
    return json.loads(Path(path).read_text())


def comparisons(pilot=False):
    """Panel roots are shared intentionally; direct and pilot/final roots are disjoint."""
    result = []
    for i, seed in enumerate(SEEDS):
        result.append(dict(label=f'seed-{seed}/direct', seed=seed, opponent=None,
                           blocks=128 if pilot else DIRECT_BLOCKS,
                           root=20261015297010+i if pilot else 20261015297101+i))
        for opponent in direct.PANEL_OPPONENTS:
            result.append(dict(label=f'seed-{seed}/panel-{opponent}', seed=seed,
                               opponent=opponent, blocks=32 if pilot else PANEL_BLOCKS,
                               root=20261015297020 if pilot else 20261015297200))
    return result


def pair(out, specs, blocks, root, opponent, cache, pilot=False):
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    registry, primary = direct.execute(specs, blocks, root, 'v055-blend', out/'play', source,
                                      stack_bb=200, opponent=opponent, cache=cache, cache_read_only=True)
    del registry
    gc.collect()
    # report=False is essential: no final contrast is opened by this per-pair audit.
    checked = direct.audit(out/'play', out/'replay.json', report=False, cost_only=pilot)
    registry, repeat = direct.execute(specs, blocks, root, 'v055-blend', out/'reproduction', source,
                                     reproduce=out/'play', stack_bb=200, opponent=opponent,
                                     cache=cache, cache_read_only=True)
    direct.put(out/'costs.json', dict(primary=primary, repeat=repeat,
                                    audit_seconds=checked['seconds'],
                                    audit_fixed_setup_seconds=checked['fixed_setup_seconds'],
                                    audit_block_work_seconds=checked['block_work_seconds'],
                                    blocks=blocks, pilot=pilot,
                                    **({'paired_sd_bb_per_100': checked['paired_sd_bb_per_100']} if pilot else {})))


def cost_projection(cost, final_blocks):
    """Pay model/table setup once; trace hashing and complete replay scale with blocks."""
    fixed = sum(cost[arm]['load_or_validation_seconds'] for arm in ('primary', 'repeat'))
    fixed += cost['audit_fixed_setup_seconds']
    work = sum(cost[arm]['play_seconds'] for arm in ('primary', 'repeat'))
    work += cost['audit_block_work_seconds']
    return fixed + final_blocks/cost['blocks']*work


def candidate_fingerprints(folder):
    """Hash repeated blend arms without interpreting their outcomes."""
    result = {}
    for filename in ('hands', 'decisions'):
        sha, count = hashlib.sha256(), 0
        with gzip.open(folder/'play'/(filename+'.jsonl.gz'), 'rt') as stream:
            for row in map(json.loads, stream):
                if row['arm'] != 'candidate':
                    continue
                if filename == 'decisions':
                    row = direct.stable_decision(row)
                sha.update((direct.canonical(row)+'\n').encode())
                count += 1
        result[filename] = dict(sha256=sha.hexdigest(), records=count)
    return result


def verify_all(base, plan):
    if plan['comparisons'] != comparisons():
        raise ValueError('Fixed final counts, opponents or roots differ')
    admission = read(base/'execution-admission.json')
    if admission['plan_sha256'] != plan['sha256']:
        raise ValueError('Execution plan differs')
    pins, shared, hands = {}, defaultdict(list), 0
    for comparison in plan['comparisons']:
        label = comparison['label']
        folder = base/'final'/label
        expected = comparison['blocks']*(4 if comparison['opponent'] else 2)
        played, replayed, repeated = (read(folder/'play/complete.json'), read(folder/'replay.json'),
                                     read(folder/'reproduction/complete.json'))
        inputs = read(folder/'play/inputs.json')
        if (played['status'] != 'complete' or replayed['status'] != 'verified'
                or not repeated['reproduced_all_hands_and_decisions']
                or not played['hands'] == replayed['hands_replayed'] == repeated['hands'] == expected
                or played['decisions'] != replayed['actions_replayed']
                or played['decisions'] != repeated['decisions']
                or inputs['source'] != admission['source']
                or inputs['plan']['blocks'] != comparison['blocks']
                or inputs['plan']['root_seed'] != comparison['root']
                or inputs.get('scripted_opponent') != comparison['opponent']
                or read(folder/'costs.json')['pilot']):
            raise ValueError('Incomplete or foreign final verification: '+label)
        models = inputs['models']
        if models != [plan['models']['blend'], plan['models'][str(comparison['seed'])]]:
            raise ValueError('Final model/translation pins differ: '+label)
        for arm, complete in (('play', played), ('reproduction', repeated)):
            for name, pin in complete['files'].items():
                path = folder/arm/name
                if path.stat().st_size != pin['bytes'] or file_hash(path) != pin['sha256']:
                    raise ValueError('Final stream changed: '+str(path))
                pins[str(path.relative_to(base))] = pin
        for path in (folder/'play/inputs.json', folder/'play/complete.json', folder/'replay.json',
                     folder/'reproduction/complete.json', folder/'costs.json'):
            pins[str(path.relative_to(base))] = dict(bytes=path.stat().st_size, sha256=file_hash(path))
        hands += expected
        if comparison['opponent']:
            shared[comparison['opponent']].append(candidate_fingerprints(folder))
    if any(len(items) != 3 or any(item != items[0] for item in items[1:]) for items in shared.values()):
        raise ValueError('The repeated blend arms did not share identical hands/decisions')
    result = dict(status='verified', at=time(), plan_sha256=plan['sha256'], files=pins,
                  recorded_primary_hands=hands, unique_primary_hands=3*DIRECT_BLOCKS*2+7*PANEL_BLOCKS*8,
                  all_final_hands_independently_replayed_and_reproduced=True,
                  shared_blend_arm_fingerprints={op: items[0] for op, items in shared.items()})
    direct.put(base/'all-final-verified.json', result)
    return result


def interval(values, family=1):
    center, sd, n = mean(values), stdev(values), len(values)
    margin = float(t.ppf(1-.05/(2*family), n-1))*sd/sqrt(n)
    return dict(bb_per_100=center, ci95=[center-margin, center+margin], sd=sd, blocks=n,
                family_comparisons=family)


def missing_shares(folder):
    counts = defaultdict(Counter)
    with gzip.open(folder/'play/decisions.jsonl.gz', 'rt') as stream:
        for row in map(json.loads, stream):
            if 'lookup' not in row:
                continue
            cell = counts[row['policy']+'/'+row['street']]
            cell['decisions'] += 1
            cell[row['lookup']] += 1
            if row['lookup'] == 'missing-key':
                cell['translated'] += row['translation']['mode'] == 'translated'
                cell['unresolved'] += row['translation']['selected_key'] is None
    return {key: dict(cell, exact_missing_share=cell['missing-key']/cell['decisions'],
                      unresolved_missing_share=cell['unresolved']/cell['decisions'])
            for key, cell in counts.items()}


def readout(base, plan):
    if not (base/'all-final-verified.json').exists():
        raise ValueError('All final hands must verify before readout')
    barrier = read(base/'all-final-verified.json')
    if barrier.get('status') != 'verified' or barrier['plan_sha256'] != plan['sha256']:
        raise ValueError('All final hands must verify before readout')
    # Include the replay receipts, not just streams, in this second barrier check.
    for name, pin in barrier['files'].items():
        path = base/name
        if path.stat().st_size != pin['bytes'] or file_hash(path) != pin['sha256']:
            raise ValueError('Verified final evidence changed')
    result = dict(status='verified', plan_sha256=plan['sha256'], direct={}, panel={},
                  no_pass_fail_or_slumbot_decision=True,
                  interval_scope='Selected fixed models and independent duplicate blocks; not training-seed population uncertainty')
    for comparison in plan['comparisons']:
        folder = base/'final'/comparison['label']
        seed = str(comparison['seed'])
        if comparison['opponent'] is None:
            values = series(folder, comparison['blocks'])['candidate']
            result['direct'][seed] = dict(contrast=interval(values), missing_key_shares=missing_shares(folder))
        else:
            opponent = comparison['opponent']
            arms = series(folder, comparison['blocks'], panel=True)
            difference = [a-b for a,b in zip(arms['candidate'], arms['baseline'], strict=True)]
            record = result['panel'].setdefault(opponent, dict(models={}, contrasts={}))
            record['models']['blend'] = interval(arms['candidate'])
            record['models'][seed] = interval(arms['baseline'])
            record['contrasts'][seed] = dict(nominal=interval(difference), bonferroni_21=interval(difference,21))
    direct.put(base/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('pair', 'verify', 'readout'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--specs', type=Path)
    parser.add_argument('--blocks', type=int)
    parser.add_argument('--root', type=int)
    parser.add_argument('--opponent', choices=direct.PANEL_OPPONENTS)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.stage == 'pair':
        pair(args.out, read(args.specs), args.blocks, args.root, args.opponent, args.cache, args.pilot)
    else:
        plan = read(args.plan)
        plan['sha256'] = file_hash(args.plan)
        (verify_all if args.stage == 'verify' else readout)(args.out, plan)


if __name__ == '__main__':
    main()
