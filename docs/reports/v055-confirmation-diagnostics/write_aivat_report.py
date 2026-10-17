"""Insert completed card-only results into the report, and retain compact estimator receipts."""
import csv
import json
import math
from pathlib import Path

OUT = Path('results/outputs')
DEST = Path('docs/reports/v055-confirmation-diagnostics')
LABELS = {'confirmation-a': 'Account A', 'confirmation-b': 'Account B', 'confirmation-c': 'Account C',
          'confirmation-pooled': '**Confirmation pooled**', '2026101302': 'Seed 1302',
          '2026101421': 'Seed 1421', '2026101422': 'Seed 1422', 'seeds-pooled': 'Seeds pooled',
          'exploratory': 'Blend exploratory'}
ORDER = ['confirmation-a', 'confirmation-b', 'confirmation-c', 'confirmation-pooled',
         '2026101302', '2026101421', '2026101422', 'seeds-pooled', 'exploratory']


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def estimate(row, key):
    mean, half = float(row[key]), float(row[key + '_half95'])
    return f'{mean:+.2f} [{mean-half:+.2f}, {mean+half:+.2f}]'


def main():
    estimates = {r['run']: r for r in csv.DictReader((OUT / 'aivat-estimates.csv').open())}
    seats = list(csv.DictReader((OUT / 'aivat-seats.csv').open()))
    weights = list(csv.DictReader((OUT / 'aivat-weights.csv').open()))
    splits = list(csv.DictReader((OUT / 'aivat-crossfit-seats.csv').open()))
    result = ['## Card-only AIVAT', '',
              'Each account uses #316’s unchanged per-hand wrapper and the existing analyzer, without a model. '
              'The pool concatenates A, B, C and fits its own weights; it does not average account fits. '
              'Earlier runs use only #316’s four saved card columns. Every per-hand row matches its verified journal’s '
              'ordered hand ID, seat and raw winnings; seed summaries reproduce #316 to numerical precision.', '',
              'BB/100 with nominal 95% intervals over hands. Worth is raw variance divided by adjusted variance '
              '(raw hands per analyzed hand). Cross-fitting alternates hands, fitting each evaluation half’s weights on the '
              'other half. Weighted intervals omit the dependence introduced by shared fitted weights. '
              'All estimates are retained; weighting is not selected after seeing which interval looks better.', '']
    result += [table(['Run', 'Hands', 'Raw', 'Plain cards', 'Weighted cards', 'Worth plain / weighted'], [
        [LABELS[name], estimates[name]['hands'], *[estimate(estimates[name], k) for k in ('raw', 'plain', 'weighted')],
         f'{float(estimates[name]["plain_worth"]):.2f} / {float(estimates[name]["weighted_worth"]):.2f}'] for name in ORDER]), '',
        'The pooled plain and weighted estimates are worth **1.53 and 1.50 raw hands per hand**, respectively. '
        'Weighting is less precise than plain cards on account C (1.25 versus 1.49), and slightly less precise '
        'in the pool. Both results are retained without changing the estimator.', '',
        '![Raw, plain and weighted card-only intervals](v055-confirmation-diagnostics/charts/aivat-intervals.png)', '',
        'The four confirmation rows, including the actual pooled fit, appear in every chart. '
        'The −50 reference is shown for context only: protocol v2 judges raw BB/100 and its verdict stands.', '',
        '![Cumulative card-only results](v055-confirmation-diagnostics/charts/aivat-cumulative.png)', '',
        'Cumulative curves show raw, plain cards and weighted cards. The pooled curve uses A, B, C concatenation, '
        'not a global chronological order across simultaneous accounts. The weighted curve is retrospective, '
        'using the full-sample cross-fitted weights.', '',
        '### Removed card luck and the zero-mean check', '',
        'Positive luck means favorable cards increased raw winnings; the correction is subtracted. '
        'The exact chance corrections have mean zero in expectation. These finite-sample intervals check compatibility '
        'with that expectation; they cannot prove absence of bias. Plain corrections, weighted corrections and '
        'their nominal intervals are reported separately.', '']
    result += [table(['Run', 'Plain removed luck ±95%', 'Plain z', 'Weighted removed luck ±95%'], [
        [LABELS[name], f'{float(estimates[name]["card_luck"]):+.2f} ± {float(estimates[name]["card_luck_half95"]):.2f}',
         f'{float(estimates[name]["card_luck_z"]):+.2f}',
         f'{float(estimates[name]["weighted_card_luck"]):+.2f} ± {float(estimates[name]["weighted_card_luck_half95"]):.2f}']
        for name in ORDER]), '',
        'Every individual and pooled removed-luck interval above includes zero. Confirmation’s plain pooled '
        'z-score is +0.30; there is no detectable nonzero average correction in this sample.', '']
    all_luck = json.loads((OUT / 'card-luck-70000.json').read_text())
    result += [f'Across all **70,000 distinct hands**, plain removed card luck is '
               f'**{all_luck["mean"]:+.2f} ± {all_luck["half_width_95"]:.2f} BB/100**. '
               'The two pooled rows above repeat their constituent hands and are excluded from this check.', '',
               '![Card luck by confirmation account and pooled](v055-confirmation-diagnostics/charts/aivat-luck.png)', '',
               '![Raw hands per card-only AIVAT hand](v055-confirmation-diagnostics/charts/aivat-worth.png)', '',
               '### Fitted street weights', '',
               'Both evaluation halves’ weights are below. Half 0 evaluates even-indexed hands with a fit on odd hands; '
               'half 1 reverses them. The chart displays the mean of those two fits, not a new fit. '
               'All earlier-run weights are also retained in the CSV.', '']
    result += [table(['Run', 'Evaluation half', 'Deal', 'Flop', 'Turn', 'River'], [
        [LABELS[r['run']], r['evaluation_parity'], *[f'{float(r[k]):.3f}' for k in ('deal', 'flop', 'turn', 'river')]]
        for r in weights if r['run'].startswith('confirmation-')]), '',
        '![Cross-fitted card weights](v055-confirmation-diagnostics/charts/aivat-weights.png)', '',
        'The [seat counts within each cross-fit half](v055-confirmation-diagnostics/tables/aivat-crossfit-seats.csv) '
        'also matter when comparing runs. The unchanged analyzer alternates hands between halves; '
        'uninterrupted sessions also alternate seats. The table below records the resulting confirmation splits. '
        'Each half still uses only the other half’s fitted weights; no alternative splitter was run. '
        'In confirmation the halves coincide with seats, so a fit learned on the button evaluates BB hands, '
        'and vice versa. These coefficients describe that unchanged split rather than an optimal fit for each seat.', '',
        table(['Run', 'Evaluation half', 'BB hands', 'Button hands'], [
            [LABELS[r['run']], r['evaluation_parity'], r['big_blind'], r['button']]
            for r in splits if r['run'].startswith('confirmation-')]), '',
        '### Results by seat', '',
        'These rates use hands in that seat as the denominator, unlike flop-fold contributions over all hands.', '',
        table(['Run', 'Seat', 'Hands', 'Raw', 'Plain cards', 'Weighted cards'], [
            [LABELS[r['run']], 'BB' if r['seat'] == 'big_blind' else 'Button', r['hands'],
             *[estimate(r, k) for k in ('raw', 'plain', 'weighted')]]
            for r in seats if r['run'].startswith('confirmation-')]), '',
        '![Raw and weighted card-only results by seat](v055-confirmation-diagnostics/charts/aivat-seats.png)', '',
        '### Comparison with #316', '']
    new = estimates['confirmation-pooled']
    for name in ('seeds-pooled', 'exploratory'):
        old = estimates[name]
        diff = float(new['weighted']) - float(old['weighted'])
        half = math.hypot(float(new['weighted_half95']), float(old['weighted_half95']))
        result += [f'Weighted confirmation minus {LABELS[name].lower()}: **{diff:+.2f} ± {half:.2f} BB/100** '
                   '(nominal independent-hand difference interval).', '']
    result += ['Both difference intervals include zero; neither comparison establishes a change in playing strength.', '',
               '#316’s headline worth of 1.3–1.8 included own-action corrections. The comparison above is '
               '**cards only throughout**, so it addresses the same estimator on every run. The exploratory '
               'weighted estimate is less precise than its plain estimate, illustrating that fitted weighting '
               'need not improve every run. No acceptance verdict or future action is chosen from these comparisons.', '']
    report = Path('docs/reports/v055-confirmation-diagnostics.md')
    text = report.read_text()
    start, stop = text.index('## Card-only AIVAT'), text.index('## What accounts for the flop-fold loss')
    report.write_text(text[:start] + '\n'.join(result) + '\n' + text[stop:])
    for name in ORDER:
        source = OUT / f'cards-{name}.json'
        (DEST / source.name).write_bytes(source.read_bytes())
    for name in ('aivat-estimates.csv', 'aivat-seats.csv', 'aivat-weights.csv', 'aivat-crossfit-seats.csv',
                 'decision-reconciliation.csv'):
        (DEST / 'tables' / name).write_bytes((OUT / name).read_bytes())


if __name__ == '__main__':
    main()
