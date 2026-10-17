"""Charts of AIVAT luck removal for Slumbot runs.

    python -m scripts.plot_slumbot_aivat --run "LABEL=PERHAND.jsonl,SUMMARY.json" [--run ...] --out-dir DIR

PERHAND.jsonl has one line per hand: `raw` winnings in chips and the eight AIVAT `corrections` in chips
(deal, flop, turn, river, then own actions preflop to river), in play order. SUMMARY.json is the
`scripts.analyze_slumbot_run --aivat` output for the same hands. With a 100-chip big blind, chips per
hand equal BB/100. Writes six PNG charts. Needs matplotlib, which is not a runtime dependency.
"""

import argparse
import json
import math
from pathlib import Path

CARD, ACTION = slice(0, 4), slice(4, 8)
STREETS = ('preflop', 'flop', 'turn', 'river')


def interval(values):
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, 1.96 * sd / math.sqrt(n)


def load(spec):
    label, _, files = spec.partition('=')
    perhand, summary = files.split(',')
    rows = [json.loads(line) for line in Path(perhand).open()]
    return {'label': label, 'rows': rows, 'summary': json.loads(Path(summary).read_text())}


def series(run):
    raw = [r['raw'] for r in run['rows']]
    card = [sum(r['corrections'][CARD]) for r in run['rows']]
    action = [sum(r['corrections'][ACTION]) for r in run['rows']]
    return raw, card, action


def cumulative(values):
    total, out = 0.0, []
    for v in values:
        total += v
        out.append(total / 100)
    return out


def pooled(runs, label):
    """Equal-size runs pooled: per-hand raw and card-only values; weighted estimates averaged."""
    rows = [r for run in runs for r in run['rows']]
    best = [run['summary']['aivat']['weighted_bb_per_100'] for run in runs]
    mean, half = interval([r['raw'] for r in rows])
    weighted = {'mean': sum(b['mean'] for b in best) / len(best),
                'half_width_95': math.sqrt(sum(b['half_width_95'] ** 2 for b in best)) / len(best)}
    return {'label': label, 'rows': rows,
            'summary': {'bb_per_100': {'mean': mean, 'half_width_95': half}, 'aivat': {'weighted_bb_per_100': weighted}}}


def forest(runs, plt, out, pools=(), cards_only=False):
    """Raw, cards-only and cards+actions weighted intervals per run, plus any pooled groups."""
    runs = list(runs) + list(pools)
    figure, axis = plt.subplots(figsize=(10, 0.9 * (len(runs) + 2) + 1))
    labels, y = [], 0
    styles = (('raw', 'Raw', '#888888'), ('cards', 'AIVAT, cards', '#4c72b0'), ('best', 'AIVAT, cards + own actions, weighted', '#dd8452'))
    if cards_only:
        styles = styles[:2] + (('best', 'AIVAT, cards, weighted', '#dd8452'),)
    for run in runs:
        s = run['summary']
        values = {'raw': s['bb_per_100'], 'cards': None, 'best': s['aivat']['weighted_bb_per_100']}
        raw, card, _ = series(run)
        mean, half = interval([r - c for r, c in zip(raw, card)])
        values['cards'] = {'mean': mean, 'half_width_95': half}
        for offset, (key, name, color) in zip((0.25, 0, -0.25), styles):
            v = values[key]
            axis.errorbar(v['mean'], y + offset, xerr=v['half_width_95'], fmt='o', color=color, capsize=3,
                          label=name if y == 0 else None)
        labels.append(run['label'])
        y -= 1
    axis.axvline(-50, color='#c44e52', linestyle='--', linewidth=1)
    axis.text(-50, 0.01, ' −50 bar', color='#c44e52', va='bottom', fontsize=9, transform=axis.get_xaxis_transform())
    axis.axvline(0, color='black', linewidth=0.6)
    axis.set_yticks(range(0, y, -1))
    axis.set_yticklabels(labels)
    axis.set_xlabel('BB/100 against Slumbot, 95% interval')
    axis.set_title('Raw and AIVAT estimates on the same hands')
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9, frameon=False)
    axis.grid(axis='x', alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / 'aivat-intervals.png', dpi=140)
    plt.close(figure)


def running(runs, plt, out, cards_only=False):
    """Cumulative winnings: raw, with card luck removed, with all measured luck removed."""
    columns = min(2, len(runs))
    rows_n = math.ceil(len(runs) / columns)
    figure, axes = plt.subplots(rows_n, columns, figsize=(6.2 * columns, 4.2 * rows_n), squeeze=False)
    for axis, run in zip(axes.flat, runs):
        raw, card, action = series(run)
        x = range(1, len(raw) + 1)
        axis.plot(x, cumulative(raw), color='#888888', linewidth=1.2, label='raw')
        axis.plot(x, cumulative([r - c for r, c in zip(raw, card)]), color='#4c72b0', linewidth=1.2, label='card luck removed')
        if cards_only:
            from src.arena.external.aivat import cross_fitted
            weighted, _ = cross_fitted(raw, [r['corrections'][:4] for r in run['rows']])
            axis.plot(x, cumulative(weighted), color='#dd8452', linewidth=1.2, label='card luck removed, weighted')
        else:
            axis.plot(x, cumulative([r - c - a for r, c, a in zip(raw, card, action)]), color='#dd8452', linewidth=1.2,
                      label='card + own-action luck removed')
        axis.axhline(0, color='black', linewidth=0.6)
        axis.set_title(run['label'])
        axis.set_xlabel('hands')
        axis.set_ylabel('cumulative big blinds')
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, loc='lower left')
    for axis in list(axes.flat)[len(runs):]:
        axis.set_visible(False)
    figure.suptitle('The same hands, with the luck taken out')
    figure.tight_layout()
    figure.savefig(out / 'aivat-cumulative.png', dpi=140)
    plt.close(figure)


def worth(runs, plt, out, cards_only=False):
    """How many raw hands one hand is worth, for each estimate."""
    def ratio(run, part):
        raw, card, action = series(run)
        adjusted = [r - c - (a if part == 'actions' else 0) for r, c, a in zip(raw, card, action)]
        return (interval(raw)[1] / interval(adjusted)[1]) ** 2
    keys = (('cards', lambda run: ratio(run, 'cards')),
            ('cards, weighted', lambda run: run.get('cards_summary', {}).get('weighted_hands_worth')),
            ('cards + own actions', lambda run: ratio(run, 'actions')),
            ('cards + own actions, weighted', lambda run: run['summary']['aivat']['weighted_hands_worth']))
    if cards_only:
        keys = (keys[0], ('cards, weighted', lambda run: run['summary']['aivat']['weighted_hands_worth']))
    figure, axis = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(keys)
    for i, (name, value) in enumerate(keys):
        points = [(j + (i - (len(keys) - 1) / 2) * width, value(run)) for j, run in enumerate(runs)]
        points = [(x, v) for x, v in points if v]
        axis.bar([x for x, _ in points], [v for _, v in points], width, label=name)
    axis.axhline(1, color='black', linewidth=0.8)
    axis.set_xticks(range(len(runs)))
    axis.set_xticklabels([r['label'] for r in runs], fontsize=9)
    axis.set_ylabel('raw hands per hand')
    axis.set_title('How many raw hands one AIVAT hand is worth')
    axis.legend(fontsize=8)
    axis.grid(axis='y', alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / 'aivat-worth.png', dpi=140)
    plt.close(figure)


def luck(runs, plt, out, cards_only=False):
    """The card luck and own-action luck each run had, with 95% intervals."""
    figure, axis = plt.subplots(figsize=(10, 4.5))
    styles = ((CARD, 'card luck', '#4c72b0'), (ACTION, 'own-action luck', '#dd8452'))
    if cards_only:
        styles = styles[:1]
    for offset, (part, name, color) in zip((0,) if cards_only else (-0.15, 0.15), styles):
        for j, run in enumerate(runs):
            mean, half = interval([sum(r['corrections'][part]) for r in run['rows']])
            axis.errorbar(j + offset, mean, yerr=half, fmt='o', color=color, capsize=4, label=name if j == 0 else None)
    axis.axhline(0, color='black', linewidth=0.8)
    axis.set_xticks(range(len(runs)))
    axis.set_xticklabels([r['label'] for r in runs], fontsize=9)
    axis.set_ylabel('BB/100 of luck removed')
    axis.set_title('How lucky each run was')
    axis.legend(fontsize=9)
    axis.grid(axis='y', alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / 'aivat-luck.png', dpi=140)
    plt.close(figure)


def weights(runs, plt, out, cards_only=False):
    """The cross-fitted weights: how much more each correction is worth than the check-down value says."""
    names = ('deal', 'flop', 'turn', 'river', 'preflop_actions', 'flop_actions', 'turn_actions', 'river_actions')
    if cards_only:
        names = names[:4]
    figure, axis = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / len(runs)
    for i, run in enumerate(runs):
        halves = run['summary']['aivat']['weights_by_half']
        values = [sum(h[n] for h in halves) / len(halves) for n in names]
        axis.bar([j + (i - (len(runs) - 1) / 2) * width for j in range(len(names))], values, width, label=run['label'])
    axis.axhline(1, color='black', linewidth=0.8)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels([n.replace('_', ' ') for n in names], fontsize=9)
    axis.set_ylabel('fitted weight (1 = check-down value)')
    axis.set_title('Cross-fitted correction weights by street')
    axis.legend(fontsize=8)
    axis.grid(axis='y', alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / 'aivat-weights.png', dpi=140)
    plt.close(figure)


def seats(runs, plt, out, cards_only=False):
    """Big blind against button, raw and with all measured luck removed (weighted)."""
    figure, axis = plt.subplots(figsize=(10, 4.5))
    seats = (('big_blind', 'big blind'), ('small_blind_button', 'button'))
    styles = (('raw', '#888888', 'o'), ('aivat', '#dd8452', 's'))
    for j, run in enumerate(runs):
        s = run['summary']
        for k, (seat, seat_name) in enumerate(seats):
            for m, (kind, color, marker) in enumerate(styles):
                v = s['by_seat'][seat] if kind == 'raw' else s['aivat']['weighted_by_seat'][seat]
                x = j + (k - 0.5) * 0.4 + (m - 0.5) * 0.12
                axis.errorbar(x, v['mean'], yerr=v['half_width_95'], fmt=marker, color=color, capsize=3,
                              label=f'{"AIVAT cards, weighted" if cards_only and kind == "aivat" else kind}, {seat_name}' if j == 0 else None,
                              alpha=1 if k == 0 else 0.55)
    axis.axhline(0, color='black', linewidth=0.8)
    axis.set_xticks(range(len(runs)))
    axis.set_xticklabels([r['label'] for r in runs], fontsize=9)
    axis.set_ylabel('BB/100, 95% interval')
    axis.set_title('Big blind (left of each pair) against button (right)')
    axis.legend(fontsize=8, ncol=2)
    axis.grid(axis='y', alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / 'aivat-seats.png', dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', required=True, help='LABEL=PERHAND.jsonl,SUMMARY.json')
    parser.add_argument('--cards-summary', action='append', default=[],
                        help='LABEL=SUMMARY.json of a cards-only run, for the worth chart')
    parser.add_argument('--pool', action='append', default=[],
                        help='LABEL=RUN_LABEL,RUN_LABEL,... equal-size runs pooled in the interval chart')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--cards-only', action='store_true',
                        help='show card-only plain/weighted estimates; omit uncomputed own-action series')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = [load(spec) for spec in args.run]
    if args.cards_only and any(run['summary']['aivat'].get('own_actions_corrected', True) for run in runs):
        parser.error('--cards-only requires card-only summaries')
    cards = dict(spec.split('=', 1) for spec in args.cards_summary)
    for run in runs:
        if run['label'] in cards:
            run['cards_summary'] = json.loads(Path(cards[run['label']]).read_text())['aivat']
    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_label = {run['label']: run for run in runs}
    pools = [pooled([by_label[name] for name in members.split(',')], label)
             for label, _, members in (spec.partition('=') for spec in args.pool)]
    forest(runs, plt, args.out_dir, pools, cards_only=args.cards_only)
    for chart in (running, worth, luck, weights, seats):
        chart(runs, plt, args.out_dir, cards_only=args.cards_only)


if __name__ == '__main__':
    main()
