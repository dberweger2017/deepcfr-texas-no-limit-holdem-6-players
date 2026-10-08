"""Export the complete audited descriptive curves, contrasts and coverage tables."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.evaluate_native_hu100_baseline import OPPONENTS

STREETS = ('preflop', 'flop', 'turn', 'river')
LOOKUPS = ('positive-mass-known-key', 'zero-mass', 'missing-key')


def export(summary, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    nodes = sorted({r['actual_nodes'] for r in summary['curves']})
    labels = ['100.691k', '1.001382M', '5.001210M', '10.001922M', '11.042440M']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for opponent, ax in zip(OPPONENTS, axes.flat):
        rows = sorted((r for r in summary['curves'] if r['opponent'] == opponent), key=lambda r: r['actual_nodes'])
        centers = [r['policy']['bb_per_100'] for r in rows]
        errors = np.array([[c - r['policy']['interval'][0], r['policy']['interval'][1] - c]
                           for c, r in zip(centers, rows)]).T
        ax.errorbar(nodes, centers, yerr=errors, color='#1864ab', marker='o', capsize=3, label='Average · block 95% CI')
        reference = rows[0]['uniform']
        ax.axhspan(*reference['interval'], color='#868e96', alpha=.16)
        ax.axhline(reference['bb_per_100'], color='#495057', linestyle='--', label='Uniform · block 95% CI')
        ax.axhline(0, color='#ced4da', linewidth=.8)
        ax.set_xscale('log'); ax.set_xticks([nodes[0], nodes[1], nodes[2], nodes[-1]], ['100.691k', '1.001M', '5.001M', '11.042M'])
        ax.set_title(opponent); ax.set_ylabel('BB/100'); ax.set_xlabel('Actual completed training nodes · log scale')
        ax.grid(axis='y', alpha=.15)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, .9, 'Five verified averages · one training seed\n\n'
                          f"{summary['blocks_per_opponent']:,} duplicate blocks/opponent/policy\n"
                          f"{summary['unique_final_hands']:,} distinct final hands\n\n"
                          'Uniform evaluated once/opponent and reused.\n'
                          'Seats swapped; private streams paired.\n'
                          'Intervals describe this scripted panel.\n\n'
                          'Final checkpoint: 11,042,440 actual nodes.\n'
                          'Its requested-1B filename is not its budget.', va='top', linespacing=1.5)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('HU100 checkpoint learning curves · all checkpoints retained', fontsize=16)
    for suffix in ('svg', 'png'):
        fig.savefig(out / f'learning-curves.{suffix}', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(16, 5), layout='constrained')
    for opponent, ax in zip(OPPONENTS, axes):
        rows = sorted((r for r in summary['final_minus_earlier'] if r['opponent'] == opponent),
                      key=lambda r: r['earlier_nodes'])
        for y, row in enumerate(rows):
            c = row['descriptive_95']['bb_per_100']
            ax.plot(row['bonferroni_20']['interval'], [y, y], color='#adb5bd', linewidth=6)
            ax.plot(row['descriptive_95']['interval'], [y, y], color='#1864ab', linewidth=2)
            ax.plot(c, y, 'o', color='#1864ab', markersize=5)
        ax.axvline(0, color='#868e96', linestyle='--'); ax.set_yticks(range(4), labels[:4])
        ax.set_title(opponent); ax.set_xlabel('Final − earlier · BB/100'); ax.invert_yaxis()
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('Paired final-minus-earlier differences\nBlue: descriptive 95% · gray: Bonferroni 99.75% (20 comparisons)', fontsize=13)
    for suffix in ('svg', 'png'):
        fig.savefig(out / f'paired-differences.{suffix}', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(16, 11), layout='constrained')
    for col, opponent in enumerate(OPPONENTS):
        for row, street in enumerate(STREETS):
            ax = axes[row, col]
            data = sorted((r for r in summary['coverage'] if r['opponent'] == opponent and r['street'] == street),
                          key=lambda r: r['actual_nodes'])
            ax.stackplot(nodes, *[[100 * r['lookup_rates'][k] for r in data] for k in LOOKUPS],
                         colors=['#2f9e44', '#fab005', '#e03131'], alpha=.8, labels=LOOKUPS)
            ax.set_ylim(0, 100); ax.set_xscale('log'); ax.set_xticks([nodes[0], nodes[1], nodes[-1]], ['100k', '1M', '11.04M'])
            if row == 0: ax.set_title(opponent)
            if col == 0: ax.set_ylabel(street + '\nDecision %')
            if row == 3: ax.set_xlabel('Actual nodes · log scale')
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='outside lower center', ncol=3)
    fig.suptitle('Decision-weighted coverage on each policy’s own trajectories\nGreen: positive mass · yellow: zero mass · red: missing key', fontsize=14)
    for suffix in ('svg', 'png'):
        fig.savefig(out / f'coverage.{suffix}', dpi=150)
    plt.close(fig)

    with (out / 'learning-curves.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['actual_nodes', 'opponent', 'policy_bb100', 'ci95_low', 'ci95_high',
                                       'uniform_bb100', 'uniform_ci95_low', 'uniform_ci95_high'])
        for r in summary['curves']:
            w.writerow([r['actual_nodes'], r['opponent'], r['policy']['bb_per_100'], *r['policy']['interval'],
                        r['uniform']['bb_per_100'], *r['uniform']['interval']])
    with (out / 'paired-differences.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['final_nodes', 'earlier_nodes', 'opponent', 'difference_bb100',
                                       'ci95_low', 'ci95_high', 'bonferroni_low', 'bonferroni_high', 'formal_label'])
        for r in summary['final_minus_earlier']:
            w.writerow([r['final_nodes'], r['earlier_nodes'], r['opponent'], r['descriptive_95']['bb_per_100'],
                        *r['descriptive_95']['interval'], *r['bonferroni_20']['interval'], r['formal_label']])
    with (out / 'coverage.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['actual_nodes', 'opponent', 'street', 'decisions',
                                       *[k + '_count' for k in LOOKUPS], *[k + '_rate' for k in LOOKUPS]])
        for r in summary['coverage']:
            w.writerow([r['actual_nodes'], r['opponent'], r['street'], r['decisions'],
                        *[r['lookup'].get(k, 0) for k in LOOKUPS], *[r['lookup_rates'][k] for k in LOOKUPS]])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); args = p.parse_args()
    export(json.loads(args.summary.read_text()), args.out)
