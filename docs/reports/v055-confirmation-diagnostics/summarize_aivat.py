"""Summarize saved card corrections with the unchanged analyzer, without recomputing equity.

Pooled confirmation weights are refitted on all 30,000 hands (A, B, C order), not averaged
from account fits. Earlier corrections come from the hash-verified #316 archive: only
its first four card columns are retained; own-action columns are never used here.
"""
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import math
import sys

import src.arena.external.analysis as analysis
from src.arena.external.aivat import cross_fitted

sys.path.insert(0, str(Path(__file__).parent))
from public_tables import load_runs

OUT = Path('results/outputs')
NAMES = ('deal', 'flop', 'turn', 'river')


@contextmanager
def cached_corrections(corrections):
    original = analysis.ProcessPoolExecutor

    class Cached:
        def __init__(self, *_):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def map(self, *_args, **_kwargs):
            return iter(corrections)

    analysis.ProcessPoolExecutor = Cached
    try:
        yield
    finally:
        analysis.ProcessPoolExecutor = original


def summarize(rows, hands):
    assert len(rows) == len(hands)
    for row, terminal in zip(rows, hands):
        assert (row['hand_id'], row['client_pos'], row['raw']) == (
            terminal['hand_id'], terminal['client_pos'], terminal['winnings'])
    raw = [r['raw'] for r in rows]
    corrections = [r['corrections'][:4] + [0.0] * 4 for r in rows]
    with cached_corrections(corrections):
        aivat = analysis.aivat_summary(raw, hands, [[] for _ in rows], 1, False)
    return {'hands': len(rows), 'bb_per_100': analysis.interval(raw),
            'by_seat': {name: analysis.interval([r['raw'] for r in rows if r['client_pos'] == seat])
                        for seat, name in ((0, 'big_blind'), (1, 'small_blind_button'))},
            'aivat': aivat}


def csv_file(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_close(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_close(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_close(a, b)
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def main():
    hands = load_runs(json.loads((OUT / 'runs.json').read_text()),
                      json.loads((OUT / 'input-integrity.json').read_text()))
    hands['seeds-pooled'] = sum((hands[s] for s in ('2026101302', '2026101421', '2026101422')), [])
    records, summaries = {}, {}
    for name in hands:
        if name == 'confirmation-pooled':
            records[name] = sum((records[f'confirmation-{c}'] for c in 'abc'), [])
        elif name == 'seeds-pooled':
            records[name] = sum((records[s] for s in ('2026101302', '2026101421', '2026101422')), [])
        else:
            source = OUT / f'perhand-{name}.jsonl' if name.startswith('confirmation-') else (
                Path('results/inputs/pr316') / f'perhand-{"blend" if name == "exploratory" else name}.jsonl')
            records[name] = [json.loads(line) for line in source.open()]
        summary = summarize(records[name], hands[name])
        if name.startswith('confirmation-') and name != 'confirmation-pooled':
            assert all(not any(r['corrections'][4:]) for r in records[name])
            analyzer = json.loads((OUT / f'aivat-{name}.json').read_text())
            assert_close(summary['aivat'], analyzer['aivat'])
            assert_close(summary['bb_per_100'], analyzer['bb_per_100'])
        if name in ('2026101302', '2026101421', '2026101422'):
            previous = json.loads(Path(f'docs/reports/slumbot-aivat-real-runs/cards-{name}.json').read_text())
            assert_close(summary['aivat'], previous['aivat'])
        summaries[name] = summary
        (OUT / f'cards-{name}.json').write_text(json.dumps(summary, indent=2) + '\n')
    # The pooled row is an actual concatenated data set for all six charts.
    with (OUT / 'perhand-confirmation-pooled.jsonl').open('x') as stream:
        for row in records['confirmation-pooled']:
            stream.write(json.dumps(row) + '\n')
    rows, seat_rows, weights, split_seats = [], [], [], []
    for name, summary in summaries.items():
        a = summary['aivat']
        luck = a['card_luck_bb_per_100']
        row = {'run': name, 'hands': summary['hands']}
        for key, value in (('raw', summary['bb_per_100']), ('plain', a['bb_per_100']),
                           ('weighted', a['weighted_bb_per_100']), ('card_luck', luck)):
            row[key] = value['mean']
            row[key + '_half95'] = value['half_width_95']
        row.update({'plain_worth': a['plain_hands_worth'], 'weighted_worth': a['weighted_hands_worth'],
                    'card_luck_z': luck['mean'] / (luck['half_width_95'] / 1.96)})
        raw = [r['raw'] for r in records[name]]
        weighted, _ = cross_fitted(raw, [r['corrections'][:4] + [0.0] * 4 for r in records[name]])
        weighted_luck = analysis.interval([r - w for r, w in zip(raw, weighted)])
        row['weighted_card_luck'] = weighted_luck['mean']
        row['weighted_card_luck_half95'] = weighted_luck['half_width_95']
        rows.append(row)
        for seat in summary['by_seat']:
            entry = {'run': name, 'seat': seat, 'hands': summary['by_seat'][seat]['n']}
            for key, values in (('raw', summary['by_seat']), ('plain', a['by_seat']),
                                ('weighted', a['weighted_by_seat'])):
                entry[key] = values[seat]['mean']
                entry[key + '_half95'] = values[seat]['half_width_95']
            seat_rows.append(entry)
        for half, w in enumerate(a['weights_by_half']):
            weights.append({'run': name, 'evaluation_parity': half, **{n: w[n] for n in NAMES}})
            selected = records[name][half::2]
            split_seats.append({'run': name, 'evaluation_parity': half, 'hands': len(selected),
                                'big_blind': sum(r['client_pos'] == 0 for r in selected),
                                'button': sum(r['client_pos'] == 1 for r in selected)})
    csv_file(OUT / 'aivat-estimates.csv', rows)
    csv_file(OUT / 'aivat-seats.csv', seat_rows)
    csv_file(OUT / 'aivat-weights.csv', weights)
    csv_file(OUT / 'aivat-crossfit-seats.csv', split_seats)
    all_luck = [sum(r['corrections'][:4]) for name, rs in records.items()
                if not name.endswith('-pooled') for r in rs]
    assert len(all_luck) == 70000
    (OUT / 'card-luck-70000.json').write_text(json.dumps(analysis.interval(all_luck), indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
