"""Audit completed continuous runs using saved outcomes; never train or infer."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from src.arena.report import comparison, estimate
from src.arena.schedule import digest


def read_rows(directory, iteration, suffix=''):
    path = directory / f'outcomes-{iteration}{suffix}.json'
    if not path.exists():
        path = path.with_suffix('.json.gz')
        with gzip.open(path, 'rt') as stream:
            rows = json.load(stream)
    else:
        rows = json.loads(path.read_text())
    report = json.loads((directory / f'evaluation-{iteration}{suffix}.json').read_text())
    if report['status'] != 'valid' or digest(rows) != report['outcomes_sha256']:
        raise ValueError(f'Invalid report or outcome digest: {path}')
    candidate = [r for r in rows if r['arm'] == 'candidate']
    keys = {(r['block'], r['rotation']) for r in candidate}
    if len(candidate) != 6144 or keys != {(b, s) for b in range(1024) for s in range(6)}:
        raise ValueError('Incomplete candidate deal/seat roster')
    blocks = defaultdict(list)
    first_shoves = preflop_all_ins = 0
    for row in candidate:
        hero = row['rotation']
        start = row['events'][0]
        if start['player_ids'][hero] != 'player-0':
            raise ValueError('Candidate identity mismatch')
        stack = start['stacks'][hero]
        actions = []
        for event in row['events']:
            if event.get('seat') != hero:
                continue
            if event['event'] == 'BlindPosted':
                stack -= event['amount']
            elif event['event'] == 'ActionTaken':
                actions.append((event['street'], event['action']['kind'], event['paid'] > 0 and event['paid'] == stack))
                stack -= event['paid']
        first_shoves += bool(actions and actions[0] == ('preflop', 'raise', True))
        preflop_all_ins += any(street == 'preflop' and all_in for street, _, all_in in actions)
        blocks[row['block']].append(row['candidate_chips'] / row['big_blind'])
    rates = [100 * sum(blocks[i]) / 6 for i in range(1024)]
    measured = estimate(rates)
    if measured != report['scenarios']['six-100bb']['comparison']['candidate']:
        raise ValueError('Recomputed candidate statistics differ')
    return {'estimate': measured, 'first_action_shoves': first_shoves,
            'preflop_all_in_hands': preflop_all_ins, 'hands': len(candidate),
            'outcomes_sha256': report['outcomes_sha256'], 'schedule_sha256': report['schedule_sha256'],
            'deal_keys': [(r['block'], r['rotation'], r['opponents']) for r in candidate],
            'control_outcomes_sha256': digest([r for r in rows if r['arm'] == 'baseline']), 'block_rates': rates}


def analyze(root):
    result = {'arms': {}, 'paired': {}, 'scope': 'Exploratory single-seed validation; rotations clustered within deal blocks.'}
    for arm in ('baseline', 'branching', 'replay'):
        base = root / f'{arm}-continuous-2026091901'
        directory = next(base.glob('scenario-*'))
        timing = [json.loads(line) for line in (directory/'training-timing.jsonl').read_text().splitlines()]
        reports = [json.loads(line) for line in (directory/'iteration-reports.jsonl').read_text().splitlines()]
        measured = {}
        for iteration in (64, 256, 512, 960, 1024, 1600, 1664):
            if (directory/f'evaluation-{iteration}.json').exists():
                measured[str(iteration)] = read_rows(directory, iteration)
        first = [r for r in timing if r['iteration'] <= 1024]
        coverage = [c for row in first for c in row['collection_coverage']]
        roots = sum(c['roots'] for c in coverage)
        records = sum(sum(c['records_by_street'].values()) for c in coverage)
        fitting = [r['fit'] for report in reports[:1024] for r in report['roles'] if r['fit']]
        result['arms'][arm] = {
            'status': json.loads((base/'status.json').read_text()),
            'supervisor': json.loads((root/f'{base.name}-supervisor.json').read_text()),
            'evaluations': measured,
            'first_1024': {
                'roots': roots, 'nodes': sum(r['nodes'] for r in first),
                'training_seconds': sum(r['total_seconds'] for r in first),
                'collection_seconds': sum(r['collection_seconds'] for r in first),
                'fitting_seconds': sum(r['fitting_seconds'] for r in first),
                'postflop_root_fraction': sum(c['roots_with_postflop'] for c in coverage) / roots,
                'postflop_record_fraction': sum(sum(n for s,n in c['records_by_street'].items() if s != 'preflop') for c in coverage) / records,
                'max_regret_update_bb': max(r['max_regret_update_bb'] for r in reports[:1024]),
                'clipped_steps': sum(f['clipped_steps'] for f in fitting),
                'fitting_steps': sum(f['steps'] for f in fitting),
            },
        }
    for arm in ('branching', 'replay'):
        result['paired'][arm] = {}
        for iteration in ('256', '512', '960', '1024'):
            a = result['arms'][arm]['evaluations'][iteration]
            b = result['arms']['baseline']['evaluations'][iteration]
            if a['schedule_sha256'] != b['schedule_sha256'] or a['deal_keys'] != b['deal_keys'] or a['control_outcomes_sha256'] != b['control_outcomes_sha256']:
                raise ValueError('Paired schedules/deals differ')
            result['paired'][arm][iteration] = comparison(a['block_rates'], b['block_rates'])
    for arm in result['arms'].values():
        for evaluation in arm['evaluations'].values():
            del evaluation['block_rates'], evaluation['deal_keys']
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.write_text(json.dumps(analyze(args.root),indent=2,allow_nan=False)+'\n')
