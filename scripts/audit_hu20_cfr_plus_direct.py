"""Independent direct-match replay and raw-chip arithmetic; no policy or production reporter load."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from time import perf_counter

from scipy.stats import t
from src.arena.runner import public_events
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stats(values):
    n = len(values)
    point = math.fsum(values) / n
    deviation = statistics.stdev(values)
    margin = float(t.ppf(.975, n - 1)) * deviation / math.sqrt(n)
    return {'blocks': n, 'bb_per_100': point, 'ci95': [point - margin, point + margin] if deviation else None}


def audit(plan, out, expected_sha256):
    started = perf_counter()
    assert digest(plan) == expected_sha256
    recorded = json.loads((out / 'summary.json').read_text())
    cells = {}
    decisions = 0
    files = []
    for lineage in (1, 2, 3):
        path = out / f'direct-lineage-{lineage}.hands.jsonl.gz'
        with path.open('rb') as f:
            files.append({'path': path.name, 'bytes': path.stat().st_size,
                          'sha256': hashlib.file_digest(f, 'sha256').hexdigest()})
        spec = plan['pairs'][lineage - 1]
        with gzip.open(path, 'rt') as stream:
            for row in map(json.loads, stream):
                coordinate = (lineage, row['block'], row['rotation'])
                assert coordinate not in cells and row['root_seed'] == plan['root'] and row['lineage'] == lineage
                assert row['policy'] == spec['candidate']['name'] and row['reference'] == spec['reference']['name']
                assert row['button'] == row['block'] % 2 and row['native_replay_verified']
                assert row['deal_seed'] == (2 << 62) | (int(digest((2, plan['root'], 'deal', (2, row['block'])))[:16], 16) & (2**62 - 1))
                hand = Hand.start(Table(('seat0', 'seat1'), (2000, 2000), button=row['button']),
                                  hand_id=row['hand_id'], seed=row['deal_seed'])
                for index, action in enumerate(row['actions']):
                    assert hand.actor == action['seat'] and index == action['index']
                    view = hand.observe(hand.actor)
                    observation = action['observation']
                    assert action['street'] == view.street.value and observation['hole_cards'] == list(view.hole_cards)
                    assert observation['board'] == list(view.board) and observation['pot'] == view.pot
                    actual = Action(ActionKind(action['kind']), action['raise_to'])
                    view.legal_actions.validate(actual)
                    hand = hand.apply(actual)
                    decisions += 1
                assert hand.finished
                chips = [player.stack - 2000 for player in hand.observe(0).players]
                assert chips == row['net_chips_by_seat'] and sum(chips) == 0
                assert chips[row['rotation']] == row['target_chips']
                assert digest(public_events(hand.events)) == row['public_events_sha256']
                cells[coordinate] = row['target_chips']
        print(json.dumps({'lineage': lineage, 'replayed': len(cells)}), flush=True)
    assert set(cells) == {(l, b, r) for l in (1, 2, 3) for b in range(plan['blocks']) for r in (0, 1)}
    blocks = range(plan['blocks'])
    result = {'overall': stats([math.fsum(cells[l, b, r] for l in (1, 2, 3) for r in (0, 1)) / 6 for b in blocks]),
              'lineages': {str(l): stats([(cells[l, b, 0] + cells[l, b, 1]) / 2 for b in blocks]) for l in (1, 2, 3)},
              'positions': {p: stats([math.fsum(cells[l, b, b % 2 if p == 'button' else 1 - b % 2]
                                              for l in (1, 2, 3)) / 3 for b in blocks]) for p in ('button', 'big_blind')}}
    def close(left, right):
        assert left['blocks'] == right['blocks'] and math.isclose(left['bb_per_100'], right['bb_per_100'], abs_tol=1e-9)
        assert (left['ci95'] is None) == (right['ci95'] is None)
        if left['ci95'] is not None:
            assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(left['ci95'], right['ci95'], strict=True))
    close(result['overall'], recorded['overall'])
    for dimension in ('lineages', 'positions'):
        for key, value in result[dimension].items():
            close(value, recorded[dimension][key])
    result.update(status='verified', plan_sha256=digest(plan), hands_replayed=len(cells), decisions_checked=decisions,
                  independent_arithmetic_matches=True, raw_files=files, seconds=perf_counter() - started)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    a = p.parse_args()
    result = audit(json.loads(a.plan.read_text()), a.out, a.plan_sha256)
    (a.out / 'audit.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
