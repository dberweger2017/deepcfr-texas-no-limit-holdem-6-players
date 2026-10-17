"""Render the public-action tables and consistency checks into a readable research appendix."""
import csv
from pathlib import Path

ROOT = Path('results/outputs/public-final')
DEST = Path('docs/reports/v055-confirmation-diagnostics')
ORDER = ['confirmation-a', 'confirmation-b', 'confirmation-c', 'confirmation-pooled',
         '2026101302', '2026101421', '2026101422', 'exploratory']
SIZES = ['<0.4', '~½', '~⅔–¾', '~pot', '1.2–2.5×', '>2.5×', 'all-in']


def load(name):
    return list(csv.DictReader((ROOT / f'{name}.csv').open()))


def md(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def rate(row, key):
    return f'{100 * float(row[key + "_rate"]):.1f}%'


def value(row, key='adds_bb_per_100'):
    return f'{float(row[key]):+.2f}'


def main():
    responses, folds, paths = load('flop-responses'), load('flop-folds'), load('flop-fold-paths')
    bb, endings, later, opens = (load(n) for n in ('bb-responses', 'bb-endings', 'bb-later-endings', 'bb-opens'))
    overall = {r['run']: r for r in load('reconciliation')}
    appendix = ['# Public-action tables', '',
                'Every table uses completed, verified hands. `player` is our policy; position is the responder’s seat. '
                'Bet size is the raise beyond a call divided by the pot after calling; all-ins have their own bucket. '
                'Buckets use upper bounds 0.4, 0.6, 0.85, 1.2 and 2.5. Bet and raise responses include all-in raises. '
                'A repeated flop exchange is another opportunity, so response counts are not a hand partition. '
                'Terminal-fold chips and the BB partitions count each hand once. Chips are final net winnings, including blinds.', '',
                'Each BB contribution is normalized by that run’s BB hands (approximately half the sample). '
                'Flop-fold contributions use all hands. At a 100-chip big blind, mean chips per hand equals BB/100. '
                'A losing response group does not establish that the response is wrong: ranges differ and the blind is already posted.', '']
    for run in ORDER:
        appendix += [f'## {run}', '', '### Flop responses', '']
        rows = [r for r in responses if r['run'] == run and r['pot'] == 'all']
        rows.sort(key=lambda r: (r['responder'], r['position'], SIZES.index(r['size'])))
        appendix += [md(['Responder', 'Position', 'Size / pot', 'Chances', 'Fold', 'Call', 'Raise'], [
            [r['responder'], r['position'], r['size'], r['opportunities'], *[rate(r, k) for k in ('fold', 'call', 'raise')]] for r in rows]), '',
            '### Terminal flop folds: additive contributions over all hands', '']
        by_size = []
        for size in SIZES:
            selected = [r for r in folds if r['run'] == run and r['size'] == size]
            our = sum(float(r['adds_bb_per_100']) for r in selected if r['folder'] == 'player')
            their = sum(float(r['adds_bb_per_100']) for r in selected if r['folder'] == 'slumbot')
            by_size.append([size, f'{our:+.2f}', f'{their:+.2f}', f'{our + their:+.2f}'])
        appendix += [md(['Size', 'Our folds', 'Slumbot folds', 'Net BB/100'], by_size), '',
                     '### Slumbot’s first button action', '']
        appendix += [md(['Action', 'Hands', 'Share of BB hands'], [[r['open'], r['hands'],
                     f'{100 * float(r["share_bb_hands"]):.2f}%'] for r in opens if r['run'] == run]), '',
                     '### Our BB response and final-hand outcomes', '']
        appendix += [md(['Open', 'Response', 'Hands', 'Response share', 'Net chips', 'Chips / selected hand',
                         'Adds BB/100 (BB hands)'], [[r['open'], r['response'], r['hands'],
                         f'{100 * float(r["response_rate"]):.1f}%', r['chips'], value(r, 'mean_chips_per_hand'), value(r)]
                         for r in bb if r['run'] == run]), '', '### BB hands by terminal ending', '']
        appendix += [md(['Ending', 'Hands', 'Net chips', 'Adds BB/100 (BB hands)'], [
                     [r['ending'], r['hands'], r['chips'], value(r)] for r in endings if r['run'] == run]), '',
                     '### BB called a 2 BB open: later endings', '']
        appendix += [md(['Ending', 'Hands', 'Net chips', 'Adds BB/100 (BB hands)'], [
                     [r['ending'], r['hands'], r['chips'], value(r)] for r in later
                     if r['run'] == run and r['open'] == '2 BB' and r['response'] == 'call']), '']
    appendix += ['## Consistency of the largest specific flop-fold paths', '',
                 'All losses below use all hands as the denominator. These are gross losses from our own folds; '
                 'Slumbot’s folds elsewhere offset them. Different rows are disjoint terminal outcomes. '
                 'They overlap the BB first-response partition and must not be added to it.', '']
    patterns = [('BB: call 2 BB, check-fold half pot', 'b200c/kb200f', '0'),
                ('BTN: limp-call 3 BB, fold half pot', 'cb300c/b300f', '1')]
    rows = []
    for run in ORDER:
        row = [run, value(overall[run], 'flop_fold_adds')]
        for _, action, pos in patterns:
            selected = [r for r in paths if r['run'] == run and r['action'] == action and r['client_pos'] == pos]
            assert len(selected) == 1
            r = selected[0]
            row.append(f'{r["hands"]}; {value(r)}')
        selected = [r for r in responses if r['run'] == run and (r['responder'], r['position'], r['pot'], r['spot'], r['size']) ==
                    ('player', 'BTN (IP)', '3bet+', 'cbet', '~½')]
        assert len(selected) == 1
        r = selected[0]
        row.append(f'{r["fold"]}/{r["opportunities"]}; {value(r, "terminal_fold_adds_bb_per_100")}')
        b = next(r for r in bb if r['run'] == run and r['open'] == '2 BB' and r['response'] == 'call')
        row.append(value(b))
        rows.append(row)
    appendix += [md(['Run', 'All flop folds net', 'BB 2 BB call / check-fold: n; loss',
                    'BTN limp-call / fold: n; loss', 'BTN 3bet+ / half-pot: folds/chances; loss',
                    'BB call group (BB denominator)'], rows), '',
                 'The first three flop patterns occur with similar frequency in every run. '
                 'The BB call group loses in every run, but BB raise outcomes and showdown results vary in sign; '
                 'there is no consistent claim that raising more would improve the policy.', '']
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'public-tables.md').write_text('\n'.join(appendix))
    for path in ROOT.glob('*.csv'):
        target = DEST / 'tables' / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())


if __name__ == '__main__':
    main()
