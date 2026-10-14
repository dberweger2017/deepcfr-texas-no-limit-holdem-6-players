"""Compare how several players played against Slumbot: hand length, pot sizes, statistics, bet sizes.

    python -m scripts.profile_slumbot_runs --player "K50 752M=RUN1,RUN2" --player "Random=RUN" \\
        --hands 5000 --out profile.json [--markdown profile.md] [--chart profile.png]

Each player's run directories are pooled in the order given and every journal is verified first.
--hands N keeps each player's first N hands in play order. Works from terminal records alone, so
players without decision records are profiled the same way (src/arena/external/style.py).
The chart needs matplotlib, which is not a runtime dependency.
"""

import argparse
import json
from pathlib import Path

from src.arena.external.style import POTS, SIZE_ORDER, STREETS, profile, terminals


def percent(entry) -> str:
    return '—' if entry['rate'] is None else f"{100 * entry['rate']:.0f}%"


def signed(value, half=None) -> str:
    if value is None:
        return '—'
    return f'{value:+.0f}' + (f' ± {half:.0f}' if half is not None else '')


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(header) - 1)) + '|']
    return '\n'.join(lines + ['| ' + ' | '.join(row) + ' |' for row in rows])


def markdown(profiles: dict[str, dict]) -> str:
    names = list(profiles)
    out = []

    out.append('### Where the result comes from: streets reached\n')
    out.append('Each cell: share of hands, then what those hands add to the overall BB/100 (95%).\n')
    keys = [f'ended {s}' for s in STREETS] + ['showdown']
    out.append(table(['Hand ended'] + names, [[k] + [
        f"{100 * p['streets_reached'][k]['share']:.0f}%, {signed(p['streets_reached'][k]['adds_bb_per_100'], p['streets_reached'][k]['adds_half_width_95'])}"
        if k in p['streets_reached'] else '—' for p in profiles.values()] for k in keys]
        + [['**Total**'] + [f"**{signed(p['overall']['bb_per_100'], p['overall']['half_width_95'])}**" for p in profiles.values()]]))

    out.append('\n### By how big the pot got\n')
    out.append('Matched pot: twice the smaller contribution. Same cells as above.\n')
    out.append(table(['Matched pot'] + names, [[name] + [
        f"{100 * p['matched_pot'][name]['share']:.0f}%, {signed(p['matched_pot'][name]['adds_bb_per_100'], p['matched_pot'][name]['adds_half_width_95'])}"
        if name in p['matched_pot'] else '—' for p in profiles.values()] for _, name in POTS]))

    out.append('\n### How each side played\n')
    rows = []
    def stat(label, get):
        rows.append([label] + [f"{get(p['player'])} / {get(p['slumbot'])}" for p in profiles.values()])
    stat('VPIP', lambda s: percent(s['vpip']))
    stat('PFR', lambda s: percent(s['pfr']))
    stat('Small blind: fold / limp / raise', lambda s: '·'.join(percent(s['small_blind_first_action'][k]) for k in ('fold', 'call', 'raise')))
    stat('3-bet', lambda s: percent(s['3bet']))
    stat('Fold to 3-bet', lambda s: percent(s['fold_to_3bet']))
    stat('Flop c-bet', lambda s: percent(s['flop_cbet']))
    stat('Fold to flop c-bet', lambda s: percent(s['against_flop_cbet']['fold']))
    stat('Raise flop c-bet', lambda s: percent(s['against_flop_cbet']['raise']))
    stat('Saw flop', lambda s: percent(s['saw_flop']))
    stat('Showdown after flop', lambda s: percent(s['showdown_after_flop']))
    stat('Won at showdown', lambda s: percent(s['won_at_showdown']))
    stat('Won after seeing flop', lambda s: percent(s['won_after_flop']))
    for street in STREETS[1:]:
        stat(f'Aggression frequency, {street}', lambda s, street=street: '—' if s['by_street'][street]['aggression_frequency'] is None
             else f"{100 * s['by_street'][street]['aggression_frequency']:.0f}%")
    out.append('Each cell: the player / Slumbot against that player. A 3-bet is any re-raise of the first preflop raise, '
               'including after a limp; a c-bet is the last preflop raiser betting first on the flop.\n')
    out.append(table(['Statistic'] + names, rows))

    out.append('\n### Bet and raise sizes, postflop\n')
    out.append('Share of each side\'s postflop bets and raises by size (fraction of the pot after calling).\n')
    rows = []
    for name, p in profiles.items():
        for who in ('player', 'slumbot'):
            counts = {k: sum(p[who]['bet_sizes'].get(s, {}).get(k, 0) for s in STREETS[1:]) for k in SIZE_ORDER}
            n = sum(counts.values())
            rows.append([f'{name}' if who == 'player' else f'Slumbot vs {name}']
                        + [f'{100 * counts[k] / n:.0f}%' if n else '—' for k in SIZE_ORDER] + [str(n)])
    out.append(table(['Side'] + SIZE_ORDER + ['bets'], rows))

    out.append("\n### Answers to Slumbot's half-pot bets\n")
    out.append('Fold / call / raise when Slumbot bets or raises about half the pot.\n')
    rows = []
    for street in STREETS[1:]:
        cells = []
        for p in profiles.values():
            a = p['player']['answers_to_bets'].get(f'{street} ~½', {})
            n = sum(a.values())
            cells.append('—' if not n else '·'.join(f"{100 * a.get(k, 0) / n:.0f}%" for k in ('fold', 'call', 'raise')) + f' of {n}')
        rows.append([street] + cells)
    out.append(table(['Street'] + names, rows))

    out.append('\n### Hand length\n')
    out.append(table(['', *names], [['Actions per hand, mean'] + [f"{p['actions_per_hand']['mean']:.1f}" for p in profiles.values()],
                                    ['Actions per hand, max'] + [str(p['actions_per_hand']['max']) for p in profiles.values()]]))
    return '\n'.join(out) + '\n'


def chart(profiles: dict[str, dict], path: Path, colors: dict[str, str] | None = None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    panels = ((axes[0], 'streets_reached', [f'ended {s}' for s in STREETS] + ['showdown'], 'By street the hand ended'),
              (axes[1], 'matched_pot', [name for _, name in POTS], 'By matched pot'))
    width = 0.8 / len(profiles)
    for axis, field, keys, title in panels:
        for i, (name, p) in enumerate(profiles.items()):
            values = [p[field].get(k, {}).get('adds_bb_per_100', 0) for k in keys]
            errors = [p[field].get(k, {}).get('adds_half_width_95') or 0 for k in keys]
            axis.bar([j + (i - (len(profiles) - 1) / 2) * width for j in range(len(keys))], values, width, yerr=errors,
                     capsize=2, label=name, color=(colors or {}).get(name), error_kw={'linewidth': 0.8})
        axis.axhline(0, color='0.5', linewidth=0.8)
        axis.set_xticks(range(len(keys)), [k.replace('ended ', '') for k in keys], fontsize=9)
        axis.set_ylabel('Adds to BB/100, with 95% interval')
        axis.set_title(title, fontsize=11)
        axis.grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--player', action='append', required=True, help='LABEL=RUN_DIR[,RUN_DIR...]')
    parser.add_argument('--hands', type=int, help="each player's first N hands")
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    parser.add_argument('--chart', type=Path)
    parser.add_argument('--color', action='append', default=[], help='LABEL=COLOR, to keep colors fixed across charts')
    args = parser.parse_args()
    profiles = {}
    for spec in args.player:
        label, _, runs = spec.partition('=')
        profiles[label] = profile(terminals([Path(r) for r in runs.split(',')], args.hands))
        profiles[label]['runs'] = runs.split(',')
    with args.out.open('x') as stream:
        stream.write(json.dumps(profiles, indent=2, ensure_ascii=False) + '\n')
    if args.markdown:
        args.markdown.write_text(markdown(profiles))
    if args.chart:
        chart(profiles, args.chart, dict(c.split('=', 1) for c in args.color))
    print(json.dumps({k: p['overall'] for k, p in profiles.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
