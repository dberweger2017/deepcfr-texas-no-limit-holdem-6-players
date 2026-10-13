"""Read a hash-verified journal prefix and describe hands that touch missing keys.

This selected-path comparison cannot measure a fallback intervention. No equity
enumeration, policy sampling, HTTP requests or writes to the input runs occur.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from math import sqrt
from statistics import mean, stdev

from scipy.stats import t
from src.arena.external.journal import canonical, FAILURES
from src.arena.external.session import hu200_average
from src.arena.external.slumbot import verify_record, verify_terminal

STREETS = ('preflop', 'flop', 'turn', 'river')


def group(values):
    n = len(values)
    center = mean(values) / 100 if n else None
    margin = float(t.ppf(.975, n - 1)) * stdev(values) / (100 * sqrt(n)) if n > 1 else None
    return {'hands': n, 'chips': sum(values), 'bb_per_hand': center,
            'ci95_bb_per_hand': [center - margin, center + margin] if margin is not None else None}


def describe(path, model, prefix_bytes=None):
    # Bound the read once: a live run may append, but cannot change this prefix.
    size = path.stat().st_size if prefix_bytes is None else prefix_bytes
    if size > path.stat().st_size: raise ValueError("Journal prefix exceeds input bytes")
    with path.open('rb') as stream:
        data = stream.read(size)
    complete = data[:data.rfind(b'\n') + 1]
    previous = '0' * 64
    hands = set(); pending = {}; values = {'missing': [], 'no_missing': []}
    decisions = Counter(); missed_hands = Counter(); first = Counter(); failures = Counter()
    identity = None
    for sequence, line in enumerate(complete.splitlines()):
        row = json.loads(line); expected = row.pop('sha256')
        if (row['sequence'] != sequence or row['previous'] != previous
                or sha256(canonical(row).encode()).hexdigest() != expected):
            raise ValueError('Journal prefix chain differs')
        payload = row['payload']; previous = expected
        if sequence == 0:
            if set(payload) != {'identity', 'scope'}: raise ValueError('Missing journal identity')
            identity = payload['identity']
            if identity.get('model_sha256') != model.description['weights_sha256']:
                raise ValueError('Journal model differs from coverage model')
            if identity.get('missing_key_fallback', 'uniform') != model.missing_key_fallback:
                raise ValueError('Journal fallback differs from analysis option')
        elif set(payload) == {'decision'}:
            record = payload['decision']; view = verify_record(record)
            menu, _, known = model.distribution(view)
            key = model.information_key(view, menu)
            lookup = 'missing' if not known else 'zero_mass' if key in model.zero_mass else 'positive_mass'
            entry = pending.setdefault(record['hand_id'], [])
            entry.append((view.street.value, lookup))
        elif set(payload) == {'terminal'}:
            terminal = payload['terminal']; verify_terminal(terminal)
            hand_id = terminal['hand_id']
            if hand_id in hands: raise ValueError('Duplicate terminal')
            hands.add(hand_id)
            touched = pending.pop(hand_id, [])
            misses = [street for street, lookup in touched if lookup == 'missing']
            values['missing' if misses else 'no_missing'].append(terminal['winnings'])
            decisions.update(touched); missed_hands.update(set(misses))
            if misses: first[misses[0]] += 1
        elif set(payload) == {'failure'} and payload['failure'] in FAILURES:
            failures[payload['failure']] += 1
        else: raise ValueError('Foreign journal payload')
    return {'path': str(path), 'prefix_bytes': len(complete), 'prefix_sha256': sha256(complete).hexdigest(),
            'tail_sha256': previous, 'identity': identity, 'groups': {k: group(v) for k, v in values.items()},
            'missing_hands_by_street': {s: missed_hands[s] for s in STREETS},
            'first_missing_street': {s: first[s] for s in STREETS},
            'coverage_completed_hands': {s: {k: decisions[s, k] for k in ('missing', 'zero_mass', 'positive_mass')} for s in STREETS},
            'decisions_without_terminal': sum(map(len, pending.values())), 'failures': dict(failures)}, values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('journals', type=Path, nargs='+')
    p.add_argument('--model', type=Path, required=True); p.add_argument('--model-sha256', required=True)
    p.add_argument('--card-buckets', type=Path, required=True); p.add_argument('--policy-cache', type=Path, required=True)
    p.add_argument('--missing-key-fallback', choices=('uniform', 'check-call'), default='uniform')
    p.add_argument('--prefix-bytes', type=int, help='recheck an already anchored single-journal prefix')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.prefix_bytes is not None and (len(a.journals) != 1 or a.prefix_bytes < 1):
        p.error('--prefix-bytes requires one journal and a positive byte count')
    model = hu200_average(a.model, a.model_sha256, a.card_buckets, a.policy_cache,
                          missing_key_fallback=a.missing_key_fallback, cache_read_only=True)
    parts = []; pooled = {'missing': [], 'no_missing': []}
    for path in a.journals:
        part, values = describe(path, model, a.prefix_bytes); parts.append(part)
        for k in pooled: pooled[k].extend(values[k])
    result = {'selection_biased': True, 'interpretation': 'Missing keys arise in unusual, big-pot lines. This split motivates a test; it does not measure a fallback.',
              'interval': 'nominal Student t 95%, independent verified completed hands within each selected class',
              'missing_key_fallback': a.missing_key_fallback, 'model': model.description,
              'parts': parts, 'groups': {k: group(v) for k, v in pooled.items()}}
    with a.out.open('x') as stream: stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__': main()
