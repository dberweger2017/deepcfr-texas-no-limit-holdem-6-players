"""Read verified three-seat duplicate blocks, using deal-cluster t intervals."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

PERMUTATIONS = tuple(itertools.permutations(range(3)))


def validate_blocks(rows):
    blocks = {}
    for row in rows:
        block = row['block']
        permutation = row['permutation']
        if not isinstance(block, int) or block < 0 or not 0 <= permutation < 6:
            raise ValueError('invalid block/permutation')
        if tuple(row['identities']) != PERMUTATIONS[permutation]:
            raise ValueError('incorrect seat permutation')
        rewards = row['rewards']
        if len(rewards) != 3 or any(type(x) is not int for x in rewards) or sum(rewards) != 0:
            raise ValueError('invalid zero-sum chip settlement')
        group = blocks.setdefault(block, {})
        if permutation in group:
            raise ValueError('duplicate permutation')
        group[permutation] = row
    if sorted(blocks) != list(range(len(blocks))) or any(len(g) != 6 for g in blocks.values()):
        raise ValueError('missing block/permutation')
    if len(blocks) < 2:
        raise ValueError('need at least two independent blocks for uncertainty')
    return blocks


def interval(values):
    from scipy.stats import t
    mean = statistics.mean(values)
    error = statistics.stdev(values) / math.sqrt(len(values))
    width = float(t.ppf(0.975, len(values) - 1)) * error
    return {'mean_bb100': mean, 'standard_error_bb100': error,
            'lower95_bb100': mean - width, 'upper95_bb100': mean + width}


def summarize(rows, big_blind):
    groups = validate_blocks(rows)
    singleton, pair = [], []
    seats_a, seats_b = [[] for _ in range(3)], [[] for _ in range(3)]
    for group in groups.values():
        a, b = [[] for _ in range(3)], [[] for _ in range(3)]
        for row in group.values():
            for seat, identity in enumerate(row['identities']):
                (a if identity == 0 else b)[seat].append(row['rewards'][seat] * 100 / big_blind)
        singleton.append(statistics.mean(sum(a, [])))
        pair.append(statistics.mean(sum(b, [])))
        for seat in range(3):
            seats_a[seat].append(statistics.mean(a[seat]))
            seats_b[seat].append(statistics.mean(b[seat]))
    return {'blocks': len(groups), 'hands': len(groups) * 6,
            'interval': 'two-sided 95% Student-t over independent six-hand deal blocks',
            'singleton': interval(singleton), 'pair_per_player': interval(pair),
            'singleton_by_seat': [interval(v) for v in seats_a],
            'pair_by_seat': [interval(v) for v in seats_b]}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verified_readout(trace, replay_receipt, reproduction):
    replay = json.loads(replay_receipt.read_text())
    digest = sha(trace)
    if replay['status'] != 'passed' or replay['trace_sha256'] != digest or sha(reproduction) != digest:
        raise ValueError('all-hand replay and exact reproduction required before readout')
    with trace.open() as f:
        header = json.loads(next(f))
        rows = [json.loads(line) for line in f]
    if header['harness'] != 'nplayer-duplicate-v1' or len(rows) != header['hands'] or len(rows) != replay['hands']:
        raise ValueError('header/count mismatch')
    result = summarize(rows, header['identity']['big_blind'])
    if result['blocks'] != header['blocks']:
        raise ValueError('block count mismatch')
    return {'trace_sha256': digest, 'root': header['seed'], 'policies': header['policies'], **result}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('replay_receipt', type=Path)
    p.add_argument('reproduction', type=Path)
    a = p.parse_args()
    print(json.dumps(verified_readout(a.trace, a.replay_receipt, a.reproduction), indent=2))
