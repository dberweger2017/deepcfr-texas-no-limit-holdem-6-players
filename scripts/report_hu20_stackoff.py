"""Stream generated regression records into paired estimates and tail tables."""
import argparse
import gzip
import json
from pathlib import Path

from scripts.evaluate_hu20 import write_json
from src.diagnostics.stackoff_report import summarize


def records(folder):
    for path in sorted(folder.glob('*.hands.jsonl.gz')):
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                yield json.loads(line)



def formatted(value):
    if value['bb_per_100'] is None:
        return 'unavailable'
    point = f"{value['bb_per_100']:+.2f}"
    interval = value['ci95']
    return point + (f" [{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else f" (CI: {value['reason']})")


def markdown(result):
    lines = ['# HU20 post-Luna regression dashboard', '',
             f"Schedule status: **{result['status']}**; {result['attempted_hands']:,}/{result['requested_hands']:,} attempted hands; "
             f"{result['failed_hands']} failed; {result['unattempted_hands']} unattempted.", '',
             'This is a post-Luna stress test, not independent confirmation, exact exploitability or a strength claim.', '',
             'Intervals are exploratory unadjusted 95%, conditional on the three fixed lineages. I average shared lineages within '
             'each paired deal block before calculating aggregate intervals; rotations and lineages are not independent samples.', '',
             '## Three-lineage checkpoint panel (BB/100)', '',
             '| Opponent | Work milestone | Paired blocks | Overall [95% CI] | Button [95% CI] | BB [95% CI] |',
             '| --- | ---: | ---: | --- | --- | --- |']
    for row in result['three_lineage_aggregate']:
        lines.append(f"| {row['panel']} | {row['milestone']/1e6:g}M | {row['overall']['blocks']} | {formatted(row['overall'])} | "
                     f"{formatted(row['positions']['button'])} | {formatted(row['positions']['big_blind'])} |")
    lines += ['', '## Stress checkpoint changes', '',
              '| Baseline → candidate | Paired blocks | Difference in BB/100 [95% CI] |', '| --- | ---: | --- |']
    for row in result['checkpoint_changes']:
        if row['panel'] == 'Selective-stackoff-v1':
            lines.append(f"| {row['baseline']/1e6:g}M → {row['candidate']/1e6:g}M | {row['paired_difference']['blocks']} | "
                         f"{formatted(row['paired_difference'])} |")
    lines += ['', '## Stress test by saved lineage', '',
              '| Seed | Work | Overall [95% CI] | Button [95% CI] | BB [95% CI] |', '| --- | ---: | --- | --- | --- |']
    stress = [row for row in result['per_seed'] if row['panel'] == 'Selective-stackoff-v1']
    for row in stress:
        lines.append(f"| {row['seed']} | {row['milestone']/1e6:g}M | {formatted(row['overall'])} | "
                     f"{formatted(row['positions']['button'])} | {formatted(row['positions']['big_blind'])} |")
    lines += ['', '## Stress tails: counts with denominators', '',
              'Large means at least 800 additional chips for the rival to call. Opportunities count decisions, not menu options. '
              'Jam means an exact all-in raise; all-in calls are excluded. Counts are descriptive and shared deals remain correlated.', '',
              '| Seed | Work | Hands | Large raises / opportunities | Rival folds / responses | Continuations / responses | Jams / opportunities | '
              '+20BB wins / hands | −20BB losses / hands | Fallback / target decisions | Large fallback / large raises |',
              '| --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in stress:
        c = row['counts']; responses = c['large_actions'] - c['no_response']
        lines.append(f"| {row['seed']} | {row['milestone']/1e6:g}M | {c['hands']} | {c['large_actions']}/{c['large_opportunities']} | "
                     f"{c['rival_folds']}/{responses} | {c['rival_continuations']}/{responses} | {c['jam_actions']}/{c['jam_opportunities']} | "
                     f"{c['full_stack_wins']}/{c['hands']} | {c['full_stack_losses']}/{c['hands']} | {c['fallback']}/{c['target_decisions']} | "
                     f"{c.get('large_fallback',0)}/{c['large_actions']} |")
    lines += ['', '## Whole-hand return partitions (stress)', '',
              '**These are whole-hand returns, not individual-bet EV.** I partition each hand using only its first large target raise; '
              'later raises do not duplicate its profit. No-response cases are explicit. JSON retains individual-seed and positional partitions.', '',
              '| Work | First large raise response | Hands | Total target BB | Mean target BB/hand |', '| ---: | --- | ---: | ---: | ---: |']
    for milestone in sorted({row['milestone'] for row in stress}):
        for label in ('folded','continued','no_response','no_large_raise'):
            partitions = [row['whole_hand_partitions'].get(label, {'hands':0,'target_chips':0})
                          for row in stress if row['milestone']==milestone]
            hands = sum(p['hands'] for p in partitions); chips = sum(p['target_chips'] for p in partitions)
            average = f'{chips/100/hands:+.4f}' if hands else 'unavailable'
            lines.append(f"| {milestone/1e6:g}M | {label} | {hands} | {chips/100:+.2f} | {average} |")
    lines += ['', '## Stress late-street lookup exposure', '',
              '| Seed | Work | Preflop fallback / decisions | Flop fallback / decisions | Turn fallback / decisions | River fallback / decisions |',
              '| --- | ---: | --- | --- | --- | --- |']
    for row in stress:
        cells = []
        for street in ('preflop','flop','turn','river'):
            c=row['counts'];fallback=c.get(street+'_fallback',0);trained=c.get(street+'_trained',0)
            cells.append(f'{fallback}/{fallback+trained}')
        lines.append(f"| {row['seed']} | {row['milestone']/1e6:g}M | " + ' | '.join(cells) + ' |')
    lines += ['', '## Bounded LBR execution', '',
              '| Seed | Work | Decisions | Completed / requested batches | Partial decisions | Over soft budget | Zero-likelihood events |',
              '| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for row in result['per_seed']:
        if row['panel'] == 'LBR-original-cap2':
            c=row['counts']
            lines.append(f"| {row['seed']} | {row['milestone']/1e6:g}M | {c['lbr_decisions']} | "
                         f"{c['lbr_completed_batches']}/{c['lbr_requested_batches']} | {c['lbr_partial_decisions']} | "
                         f"{c['lbr_over_soft_budget']} | {c['lbr_zero_likelihood_events']} |")
    lines += ['', '## Interpretation limits', '',
              'I retain the complete scripted/LBR panel, individual-seed differences and positional tails in `summary.json`. '
              'This fresh, smaller schedule does not replace or pool #116/#117. Positive scripted-opponent returns do not establish general wins. '
              'Training-node work is not a count of poker hands. Similar aggression across card buckets alone does not prove an error.', '']
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads((args.run / 'plan.json').read_text()), records(args.run))
    write_json(args.run / 'summary.json', result)
    (args.run / 'dashboard.md').write_text(markdown(result))
    print(json.dumps({k: result[k] for k in ('status', 'requested_hands', 'attempted_hands', 'failed_hands')}))


if __name__ == '__main__':
    main()
