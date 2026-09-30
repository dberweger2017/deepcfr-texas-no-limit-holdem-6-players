"""Verify packaged hashes, native replay and tails from the published CSV."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from math import isclose
from time import time

from src.arena.schedule import digest
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.stackoff_report import summarize
from src.diagnostics.stackoff_tails import hand_tails
from scripts.play_robustness import replay_row


def boolean(value):
    if value not in ('True', 'False'):
        raise ValueError('Invalid CSV boolean')
    return value == 'True'


def checked_rows(evidence, stats):
    with gzip.open(evidence / 'generated-hands.jsonl.gz', 'rt') as hands, gzip.open(evidence / 'decisions.csv.gz', 'rt') as decisions:
        reader = csv.DictReader(decisions)
        for line in hands:
            row = json.loads(line)
            replay_row(row)
            for action in row['actions']:
                decision = next(reader)
                coordinates = {'model': row['policy'], **{k: row[k] for k in ('panel', 'block', 'rotation')},
                               **{k: action[k] for k in ('index', 'seat', 'logical_player', 'street', 'kind')}}
                if any(decision[k] != str(value) for k, value in coordinates.items()):
                    raise ValueError('Raw decision and native replay coordinates differ')
                if (int(decision['raise_to']) if decision['raise_to'] else None) != action['raise_to']:
                    raise ValueError('Raw decision raise-to differs')
                action['observation'] = {
                    'street': decision['street'], 'call_amount': int(decision['call_amount']),
                    'trained': boolean(decision['trained']) if action['logical_player'] == 0 else None,
                    'large_raise_opportunity': boolean(decision['large_raise_opportunity']),
                    'jam_opportunity': boolean(decision['jam_opportunity']), 'menu': json.loads(decision['menu'])}
                stats['raw_decisions'] += 1
            if hand_tails(row) != row['tails']:
                raise ValueError('Tail arithmetic differs from raw decisions')
            stats['native_replays'] += 1
            yield row
        if next(reader, None) is not None:
            raise ValueError('Unmatched trailing decisions')



def same_summary(actual, recorded):
    # Counts/structure stay exact; quantile-library roundoff is not a changed result.
    if type(actual) is not type(recorded):
        return False
    if isinstance(actual, float):
        return isclose(actual, recorded, rel_tol=1e-12, abs_tol=1e-9)
    if isinstance(actual, dict):
        return actual.keys() == recorded.keys() and all(same_summary(v, recorded[k]) for k, v in actual.items())
    if isinstance(actual, list):
        return len(actual) == len(recorded) and all(same_summary(a, b) for a, b in zip(actual, recorded))
    return actual == recorded

def check(evidence):
    started = time()
    manifest = json.loads((evidence / 'evidence-manifest.json').read_text())
    for name, entry in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Evidence manifest must contain simple filenames')
        path = evidence / name
        if path.stat().st_size != entry['bytes'] or file_hash(path) != entry['sha256']:
            raise ValueError('Packaged file hash or size differs')
    plan = json.loads((evidence / 'plan.json').read_text())
    if digest(plan) != json.loads((evidence / 'manifest.json').read_text())['plan_sha256']:
        raise ValueError('Frozen plan digest differs')
    stats = {'native_replays': 0, 'raw_decisions': 0, 'verified_files': len(manifest['files'])}
    computed = summarize(plan, checked_rows(evidence, stats))
    if not same_summary(computed, json.loads((evidence / 'summary.json').read_text())):
        raise ValueError('Paired summary differs from raw chips and decisions')
    stats.update(status=computed['status'], seconds=time() - started)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.evidence), sort_keys=True))


if __name__ == '__main__':
    main()
