"""Chart several players' verified Slumbot runs side by side.

    python -m scripts.plot_slumbot_runs --player "K50 752M=RUN1,RUN2" --player "Random=RUN" --out chart.png

Each player's run directories are pooled in the order given; every journal is verified first.
Left: cumulative winnings in big blinds. Right: BB/100 so far with its 95% band.
Needs matplotlib, which is not a runtime dependency.
"""

import argparse
import json
import math
from pathlib import Path

from src.arena.external.journal import verify_journal


def winnings(runs: list[Path]) -> list[int]:
    """Chips won per hand, in play order, from verified journals."""
    hands = []
    for run in runs:
        verify_journal(run / 'journal.jsonl')
        for line in (run / 'journal.jsonl').open():
            payload = json.loads(line)['payload']
            if 'terminal' in payload:
                hands.append(payload['terminal']['winnings'])
    return hands


def running(hands: list[int]):
    """Per hand: cumulative BB, BB/100 so far, and its 95% half-width (None before two hands)."""
    total = square = 0.0
    rows = []
    for n, chips in enumerate(hands, 1):
        total += chips; square += chips * chips
        mean = total / n
        half = 1.96 * math.sqrt(max(0.0, (square - n * mean * mean) / (n - 1)) / n) if n > 1 else None
        rows.append((total / 100, mean, half))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--player', action='append', required=True, help='LABEL=RUN_DIR[,RUN_DIR...]')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--title', default='Against Slumbot')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))
    for spec in args.player:
        label, _, runs = spec.partition('=')
        rows = running(winnings([Path(r) for r in runs.split(',')]))
        x = range(1, len(rows) + 1)
        final = rows[-1]
        line, = left.plot(x, [r[0] for r in rows], linewidth=1.4,
                          label=f'{label}: {final[1]:+.0f} ± {final[2]:.0f} BB/100, {len(rows):,} hands')
        start = min(200, len(rows) - 1)          # The first hands' intervals are too wide to draw.
        right.plot(x[start:], [r[1] for r in rows[start:]], color=line.get_color(), linewidth=1.4)
        right.fill_between(x[start:], [r[1] - r[2] for r in rows[start:]], [r[1] + r[2] for r in rows[start:]],
                           color=line.get_color(), alpha=0.15, linewidth=0)
    for axis in (left, right):
        axis.axhline(0, color='0.5', linewidth=0.8)
        axis.set_xlabel('Hands played')
        axis.grid(alpha=0.3)
    left.set_ylabel('Cumulative winnings (big blinds)')
    right.set_ylabel('BB/100 so far, with 95% band')
    left.legend(loc='upper right', fontsize=9)
    figure.suptitle(args.title)
    figure.tight_layout()
    figure.savefig(args.out, dpi=130)


if __name__ == '__main__':
    main()
