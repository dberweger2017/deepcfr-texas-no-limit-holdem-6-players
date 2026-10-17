"""Public-action diagnosis; no policy or card-strength lookups.

Run after prepare_inputs.py: --runs results/outputs/runs.json
--integrity results/outputs/input-integrity.json --out-dir results/outputs/public

Response outcomes are descriptive final-hand winnings, not action EVs. Repeated flop
responses can refer to the same hand; only the terminal-fold and BB partitions add up.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from src.arena.external.style import replay, size_label

KINDS = ('fold', 'call', 'raise')


def load_runs(spec, integrity):
    """Reuse a complete journal verification only while its exact bytes still match."""
    loaded, result = {}, {}
    for name, paths in spec.items():
        hands = []
        for folder in paths:
            path = str(Path(folder) / 'journal.jsonl')
            if path not in loaded:
                data = Path(path).read_bytes()
                assert hashlib.sha256(data).hexdigest() == integrity['journals'][path]['sha256'], path
                loaded[path] = [p['terminal'] for line in data.splitlines()
                                if 'terminal' in (p := json.loads(line)['payload'])]
                assert len(loaded[path]) == integrity['journals'][path]['hands']
            hands.extend(loaded[path])
        assert len(hands) == (30000 if name == 'confirmation-pooled' else 10000), (name, len(hands))
        result[name] = hands
    return result


def outcome(values, total):
    n, chips = len(values), sum(values)
    mean = chips / n if n else None
    variance = (sum(v * v for v in values) - chips * chips / total) / (total - 1) if total > 1 else 0
    return {'hands': n, 'chips': chips, 'mean_chips_per_hand': mean,
            'half_width_95': 1.96 * statistics.stdev(values) / math.sqrt(n) if n > 1 else None,
            'adds_bb_per_100': chips / total,
            'adds_half_width_95': 1.96 * math.sqrt(max(0, variance) / total)}


def pot_class(events):
    n = sum(e['kind'] == 'raise' for e in events if e['street'] == 0)
    return 'limped' if not n else 'single-raised' if n == 1 else '3bet+' 


def spot(bet, preflop):
    if bet['kind'] == 'raise':
        return 'raise'
    raisers = [e['side'] for e in preflop if e['kind'] == 'raise']
    if not raisers:
        return 'limped bet'
    return 'cbet' if bet['side'] == raisers[-1] else 'non-aggressor bet'


def write_csv(path, rows):
    assert rows, path
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tables(name, hands):
    total = len(hands)
    bb_total = sum(t['client_pos'] == 0 for t in hands)
    responses = defaultdict(lambda: {'counts': Counter(), 'fractions': [], 'terminal_fold_chips': 0})
    folds = defaultdict(list)
    fold_paths = defaultdict(list)
    openings = defaultdict(list)
    bb_responses = defaultdict(list)
    bb_endings = defaultdict(list)
    bb_postflop = defaultdict(list)
    flop_totals = Counter()
    for t in hands:
        events, paid = replay(t)
        preflop = [e for e in events if e['street'] == 0]
        pot = pot_class(events)
        # Event actor is determined by client_pos; seat 0 acts first postflop.
        for bet, answer in zip(events, events[1:]):
            if bet['street'] != 1 or answer['street'] != 1 or bet['kind'] not in ('bet', 'raise'):
                continue
            responder = answer['side']
            seat = t['client_pos'] if responder == 'player' else 1 - t['client_pos']
            position = 'BB (OOP)' if seat == 0 else 'BTN (IP)'
            size = size_label(bet['size'], bet['all_in'])
            role = spot(bet, preflop)
            kind = 'raise' if answer['kind'] in ('bet', 'raise') else answer['kind']
            assert kind in KINDS and answer['facing'] > 0
            for key in ((responder, position, 'all', 'all', size),
                        (responder, position, pot, role, size)):
                group = responses[key]
                group['counts'][kind] += 1
                group['fractions'].append(bet['size'])
                if kind == 'fold':
                    group['terminal_fold_chips'] += t['winnings']
            if kind == 'fold':
                assert answer is events[-1]
                folds[position, pot, role, size, responder].append(t['winnings'])
                flop_totals[responder] += t['winnings']
                fold_paths[t['action'], t['client_pos'], responder].append(t['winnings'])
        if t['client_pos'] != 0:
            continue
        first = preflop[0]
        assert first['side'] == 'slumbot'
        if first['kind'] == 'raise':
            # Initial pot is 150; style.replay stores increment/(pot + call).
            # Opening raise-to = 100 + fraction * 200, in chips.
            open_to = round(100 + first['size'] * 200)
            label = f'{open_to / 100:g} BB' + (' all-in' if first['all_in'] else '')
            response = preflop[1]['kind']
            assert response in KINDS
        else:
            label = 'limp' if first['kind'] == 'call' else 'fold'
            response = preflop[1]['kind'] if len(preflop) > 1 else 'no response'
        openings[label].append(t['winnings'])
        bb_responses[label, response].append(t['winnings'])
        last = events[-1]
        ending = ('preflop: ' if last['street'] == 0 else 'later: ') + (
            f'{last["side"]} folded' if last['kind'] == 'fold' else 'showdown')
        bb_endings[ending].append(t['winnings'])
        if last['street'] != 0:
            if last['kind'] == 'fold':
                street = ('preflop', 'flop', 'turn', 'river')[last['street']]
                end = f'{last["side"]} folded {street}'
            else:
                end = 'showdown'
            bb_postflop[label, response, end].append(t['winnings'])

    response_rows = []
    for key, group in sorted(responses.items()):
        responder, position, pot, role, size = key
        counts, fractions = group['counts'], group['fractions']
        n = sum(counts.values())
        response_rows.append({'run': name, 'responder': responder, 'position': position, 'pot': pot,
                              'spot': role, 'size': size, 'opportunities': n,
                              **{k: counts[k] for k in KINDS},
                              **{f'{k}_rate': counts[k] / n for k in KINDS},
                              'fraction_min': min(fractions), 'fraction_median': statistics.median(fractions),
                              'fraction_max': max(fractions), 'terminal_fold_chips': group['terminal_fold_chips'],
                              'terminal_fold_adds_bb_per_100': group['terminal_fold_chips'] / total})
    fold_rows = [{'run': name, **dict(zip(('position', 'pot', 'spot', 'size', 'folder'), key)),
                  **outcome(values, total)} for key, values in sorted(folds.items())]
    path_rows = [{'run': name, 'action': action, 'client_pos': seat, 'folder': folder,
                  **outcome(values, total)} for (action, seat, folder), values in sorted(fold_paths.items())]
    net_folds = defaultdict(list)
    for key, values in folds.items():
        net_folds[key[:-1]].extend(values)
    net_rows = [{'run': name, **dict(zip(('position', 'pot', 'spot', 'size'), key)),
                 **outcome(values, total)} for key, values in sorted(net_folds.items())]
    open_rows = [{'run': name, 'open': label, 'share_bb_hands': len(v) / bb_total,
                 **outcome(v, bb_total)} for label, v in sorted(openings.items())]
    bb_rows = [{'run': name, 'open': label, 'response': response,
               'response_rate': len(v) / len(openings[label]), **outcome(v, bb_total),
               'adds_over_all_hands': sum(v) / total}
              for (label, response), v in sorted(bb_responses.items())]
    ending_rows = [{'run': name, 'ending': label, **outcome(v, bb_total),
                   'adds_over_all_hands': sum(v) / total} for label, v in sorted(bb_endings.items())]
    postflop_rows = [{'run': name, 'open': label, 'response': response, 'ending': end,
                     **outcome(v, bb_total)} for (label, response, end), v in sorted(bb_postflop.items())]
    all_bb = [t['winnings'] for t in hands if t['client_pos'] == 0]
    assert sum(r['chips'] for r in bb_rows) == sum(all_bb)
    assert sum(r['hands'] for r in bb_rows) == bb_total
    assert sum(r['chips'] for r in ending_rows) == sum(all_bb)
    expected = sum(t['winnings'] for t in hands if t['action'].endswith('f') and t['action'].count('/') == 1)
    assert sum(r['chips'] for r in fold_rows) == expected
    assert sum(r['chips'] for r in net_rows) == expected
    return {'flop-responses': response_rows, 'flop-folds': fold_rows, 'flop-fold-net': net_rows, 'flop-fold-paths': path_rows,
            'bb-opens': open_rows, 'bb-responses': bb_rows, 'bb-endings': ending_rows,
            'bb-later-endings': postflop_rows,
            'reconciliation': [{'run': name, 'hands': total, 'bb_hands': bb_total, 'bb_chips': sum(all_bb),
                                'bb_rate': sum(all_bb) / bb_total, 'flop_fold_chips': expected,
                                'flop_fold_adds': expected / total,
                                'our_flop_fold_chips': flop_totals['player'],
                                'slumbot_flop_fold_chips': flop_totals['slumbot']}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--integrity', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(json.loads(args.runs.read_text()), json.loads(args.integrity.read_text()))
    combined = defaultdict(list)
    for name, hands in runs.items():
        for table, rows in tables(name, hands).items():
            combined[table].extend(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for table, rows in combined.items():
        write_csv(args.out_dir / f'{table}.csv', rows)
    print(json.dumps(combined['reconciliation'], indent=2))


if __name__ == '__main__':
    main()
