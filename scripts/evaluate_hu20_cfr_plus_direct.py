"""Frozen direct CFR+ average versus v0.4.0 matches, outside the panel release rule."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from random import Random
import resource
from time import time, perf_counter

from scripts.evaluate_hu20_cfr_average import play
from scripts.evaluate_hu20_v041_arena import load
from src.arena.report import estimate
from src.arena.schedule import digest, stream_seed


class PolicyRival:
    """A frozen policy receives only its own public observation and private action RNG."""
    def __init__(self, source, seed):
        self.source = source
        self.random = Random(seed)

    def choose_action(self, view):
        menu, probabilities, _ = self.source.distribution(view)
        return self.random.choices(menu, weights=probabilities, k=1)[0].action


def run(plan, policies, out, lineage):
    out.mkdir(parents=True, exist_ok=True)
    spec = plan['pairs'][lineage - 1]
    started = perf_counter()
    name = f'direct-lineage-{lineage}'
    hands = 0
    def guard():
        if time() > plan['started_at'] + plan['max_seconds']:
            raise TimeoutError('Frozen direct-match deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 6 * 1024**3:
            raise MemoryError('Direct-match RSS guard')
    failure = None
    try:
        guard()
        candidate = load(spec['candidate'], policies)
        reference = load(spec['reference'], policies)
        with gzip.open(out / f'{name}.hands.jsonl.gz', 'xt') as stream:
            for block in range(plan['blocks']):
                for rotation in (0, 1):
                    rival = PolicyRival(reference, stream_seed(plan['root'], 'test', 'opponent', 2, block, 1))
                    row = play(candidate, spec['candidate'], {'name': 'direct-v040', 'contract': 'frozen-policy',
                               'rule': 'frozen-policy'}, plan['root'], block, rotation, guard, rival=rival)
                    row.update(arm='O', lineage=lineage, reference=spec['reference']['name'])
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                    stream.flush()
                    hands += 1
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    result = {'status': 'incomplete' if failure else 'complete', 'failure': failure, 'hands': hands,
              'seconds': perf_counter() - started, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'plan_sha256': digest(plan), 'lineage': lineage}
    (out / f'{name}.result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def report(plan, out):
    cells = defaultdict(dict)
    for lineage in (1, 2, 3):
        name = f'direct-lineage-{lineage}'
        result = json.loads((out / f'{name}.result.json').read_text())
        if result['status'] != 'complete' or result['plan_sha256'] != digest(plan):
            raise ValueError('Incomplete or foreign direct-match lineage')
        with gzip.open(out / f'{name}.hands.jsonl.gz', 'rt') as stream:
            for row in map(json.loads, stream):
                key = (row['block'], row['rotation'])
                if row['lineage'] != lineage or row['root_seed'] != plan['root'] or key in cells[lineage]:
                    raise ValueError('Foreign or duplicate direct-match coordinate')
                cells[lineage][key] = row['target_chips']
        if set(cells[lineage]) != {(b, r) for b in range(plan['blocks']) for r in (0, 1)}:
            raise ValueError('Incomplete direct-match coverage')
    blocks = range(plan['blocks'])
    overall = estimate([sum(cells[l][b, r] for l in (1, 2, 3) for r in (0, 1)) / 6 for b in blocks])
    lineages = {str(l): estimate([(cells[l][b, 0] + cells[l][b, 1]) / 2 for b in blocks]) for l in (1, 2, 3)}
    positions = {p: estimate([sum(cells[l][b, b % 2 if p == 'button' else 1 - b % 2]
                                   for l in (1, 2, 3)) / 3 for b in blocks]) for p in ('button', 'big_blind')}
    summary = {'plan_sha256': digest(plan), 'hands': 6 * plan['blocks'], 'overall': overall,
               'lineages': lineages, 'positions': positions,
               'scope': 'direct CFR+ average vs matched v0.4.0 lineages; exploratory, outside release rule'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('play', 'report'))
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--policies', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lineage', type=int, choices=(1, 2, 3))
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    result = run(plan, a.policies, a.out, a.lineage) if a.command == 'play' else report(plan, a.out)
    print(json.dumps(result), flush=True)
    return int(result.get('status') == 'incomplete')


if __name__ == '__main__':
    raise SystemExit(main())
