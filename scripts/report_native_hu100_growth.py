"""Paired primary statistics and visit/exposure summaries for the fixed growth test."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from scripts.report_native_hu100_learning_curves import summarize
from src.arena.artifacts import write_json
from src.policies.files import file_hash

PRIMARY = ('tight_aggressive', 'loose_aggressive')

def report(run, *, output=None, verified_source_fingerprint=None):
    output = output or run
    settings = json.loads((run / 'settings.json').read_text())
    result = summarize(run, settings, output / 'paired-summary.json', formal_opponents=PRIMARY,
                       verified_source_fingerprint=verified_source_fingerprint)
    visits, exposures = [], []
    for spec in settings['models']:
        target = run / 'final' / str(spec['actual_nodes'])
        selected = set(); decisions = {}
        for opponent in settings['opponents']:
            with gzip.open(target / opponent / 'decisions.jsonl.gz', 'rt') as f:
                rows = [r for line in f if (r := json.loads(line))['arm'] == 'candidate' and r['logical_player'] == 0]
            decisions[opponent] = rows
            selected.update(r['key'] for r in rows)
        table = {}
        with gzip.open(spec['path'], 'rt') as f:
            next(f)
            for line in f:
                row = json.loads(line)
                if row[0] in selected:
                    table[row[0]] = (row[3], row[4])
        if file_hash(Path(spec['path'])) != spec['sha256']:
            raise ValueError('Visit source changed')
        for opponent, rows in decisions.items():
            groups = defaultdict(Counter); hand_lookup = defaultdict(list)
            for r in rows:
                known = table.get(r['key']); lookup = 'missing-key' if known is None else 'positive-mass-known-key' if known[0] > 0 else 'zero-mass'
                if lookup != r['lookup']:
                    raise ValueError('Independent lookup/visit recount mismatch')
                n = known[1] if known else None
                band = 'missing' if n is None else '0' if n == 0 else '1' if n == 1 else '2-9' if n < 10 else '10-99' if n < 100 else '100+'
                groups[r['street']][band] += 1
                hand_lookup[r['block'], r['rotation']].append(lookup)
            for street, counts in groups.items():
                visits.append({'nodes': spec['actual_nodes'], 'opponent': opponent, 'street': street,
                               'decisions': sum(counts.values()), 'visit_bands': dict(counts)})
            counts = Counter(); chips = Counter(); all_hands = 0
            with (target / opponent / 'hands.jsonl').open() as f:
                for line in f:
                    h = json.loads(line)
                    if h['arm'] != 'candidate': continue
                    lookups = hand_lookup[h['block'], h['rotation']]
                    category = ('ever-missing' if 'missing-key' in lookups else 'any-zero-no-missing'
                                if 'zero-mass' in lookups else 'all-positive' if lookups else 'no-target-decision')
                    counts[category] += 1; chips[category] += h['candidate_chips']; all_hands += 1
            exposures.append({'nodes': spec['actual_nodes'], 'opponent': opponent, 'hands': all_hands,
                'categories': {k: {'hands': counts[k], 'bb_per_100_contribution': chips[k] / all_hands}
                               for k in ('ever-missing', 'any-zero-no-missing', 'all-positive', 'no-target-decision')}})
    result.update(primary_opponents=list(PRIMARY), visits=visits, hand_exposures=exposures,
                  association_scope='policy-dependent descriptive exposure; no causal branch gain')
    write_json(output / 'result.json', result)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    a = p.parse_args(); r = report(a.run)
    print(json.dumps({'status': r['status'], 'hands': r['unique_final_hands']}))
