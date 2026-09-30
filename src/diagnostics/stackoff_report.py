"""Paired block uncertainty and integer-chip tail arithmetic."""

from collections import Counter, defaultdict
from statistics import mean

from src.arena.report import estimate


def summarize(plan, rows):
    models = {m['name']: m for m in plan['models']}
    panels = {p['name']: p for p in plan['panels']}
    expected = {(m, p, b, r) for m in models for p, panel in panels.items()
                for b in range(panel['blocks']) for r in (0, 1)}
    observed, values, tails, partitions = set(), {}, defaultdict(Counter), defaultdict(dict)
    failures = []
    for row in rows:
        key = (row['policy'], row['panel'], row['block'], row['rotation'])
        if key not in expected or key in observed:
            raise ValueError('Unexpected or duplicate scheduled hand')
        observed.add(key)
        if row['status'] != 'complete':
            failures.append(key)
            continue
        if not row.get('native_replay_verified'):
            raise ValueError('Completed hand lacks native replay verification')
        if sum(row['net_chips_by_seat']) != 0 or row['target_chips'] != row['net_chips_by_seat'][row['rotation']]:
            raise ValueError('Invalid target payoff')
        position = 'button' if row['rotation'] == row['button'] else 'big_blind'
        values[key[:3] + (position,)] = row['target_chips']
        group = key[:2]
        partition = row['tails']['first_large_raise_response']
        for cell in (group, (*group, position)):
            tails[cell].update(row['tails']['counts'])
            item = partitions[cell].setdefault(partition, {'hands': 0, 'target_chips': 0})
            item['hands'] += 1
            item['target_chips'] += row['target_chips']
    complete = observed == expected and not failures
    summaries, series = [], {}
    for name, spec in models.items():
        for panel, definition in panels.items():
            group = (name, panel)
            blocks = [b for b in range(definition['blocks'])
                      if all((name, panel, b, pos) in values for pos in ('button', 'big_blind'))]
            positions = {pos: [values[(name, panel, b, pos)] for b in blocks]
                         for pos in ('button', 'big_blind')}
            # At 100 chips/BB, mean chips/hand numerically equals BB/100.
            overall = [mean(pair) for pair in zip(*positions.values(), strict=True)]
            series[group] = (blocks, overall, positions)
            summaries.append({'model': name, 'seed': spec['seed'], 'milestone': spec['milestone'],
                              'panel': panel, 'overall': estimate(overall),
                              'positions': {p: estimate(v) for p, v in positions.items()},
                              'counts': dict(tails[group]), 'whole_hand_partitions': partitions[group],
                              'positional_tails': {p: {'counts': dict(tails[(*group, p)]),
                                  'whole_hand_partitions': partitions[(*group, p)]}
                                  for p in ('button', 'big_blind')},
                              'completed_paired_blocks': len(blocks)})
    aggregates, changes, seed_changes, seed_differences = [], [], [], []
    milestones = sorted({m['milestone'] for m in models.values()})
    for panel, definition in panels.items():
        aggregated = {}
        for milestone in milestones:
            names = [n for n, m in models.items() if m['milestone'] == milestone]
            blocks = sorted(set.intersection(*(set(series[(n, panel)][0]) for n in names)))
            position_values = {pos: [mean(values[(n, panel, b, pos)] for n in names) for b in blocks]
                               for pos in ('button', 'big_blind')}
            overall = [mean(pair) for pair in zip(*position_values.values(), strict=True)]
            aggregated[milestone] = dict(zip(blocks, overall, strict=True))
            aggregates.append({'panel': panel, 'milestone': milestone, 'lineages': len(names),
                               'overall': estimate(overall),
                               'positions': {p: estimate(v) for p, v in position_values.items()}})
        for baseline, candidate in sorted(set([(milestones[0], m) for m in milestones[1:]] +
                                               list(zip(milestones, milestones[1:])))):
            shared = sorted(aggregated[baseline].keys() & aggregated[candidate].keys())
            changes.append({'panel': panel, 'baseline': baseline, 'candidate': candidate,
                            'paired_difference': estimate([aggregated[candidate][b] - aggregated[baseline][b]
                                                           for b in shared])})
    for panel in panels:
        for seed in sorted({m['seed'] for m in models.values()}):
            names = {m['milestone']: n for n, m in models.items() if m['seed'] == seed}
            for baseline, milestone in sorted(set([(milestones[0], m) for m in milestones[1:]] +
                                                  list(zip(milestones, milestones[1:])))):
                base, candidate = names[baseline], names[milestone]
                shared = sorted(set(series[(base, panel)][0]) & set(series[(candidate, panel)][0]))
                seed_changes.append({'panel': panel, 'seed': seed, 'baseline': baseline,
                                     'candidate': milestone, 'paired_difference': estimate([
                    mean(values[(candidate, panel, b, p)] - values[(base, panel, b, p)]
                         for p in ('button', 'big_blind')) for b in shared])})
        for milestone in milestones:
            names = sorted((m['seed'], n) for n, m in models.items() if m['milestone'] == milestone)
            for index, (baseline_seed, base) in enumerate(names):
                for candidate_seed, candidate in names[index + 1:]:
                    shared = sorted(set(series[(base, panel)][0]) & set(series[(candidate, panel)][0]))
                    seed_differences.append({'panel': panel, 'milestone': milestone,
                        'baseline_seed': baseline_seed, 'candidate_seed': candidate_seed,
                        'paired_difference': estimate([mean(
                            values[(candidate, panel, b, p)] - values[(base, panel, b, p)]
                            for p in ('button', 'big_blind')) for b in shared])})
    return {'status': 'complete' if complete else 'incomplete', 'requested_hands': len(expected),
            'attempted_hands': len(observed), 'failed_hands': len(failures),
            'unattempted_hands': len(expected - observed), 'per_seed': summaries,
            'three_lineage_aggregate': aggregates, 'checkpoint_changes': changes, 'per_seed_checkpoint_changes': seed_changes,
            'between_seed_differences': seed_differences,
            'uncertainty': plan['interval'],
            'warning': 'Whole-hand subgroup returns are not individual-bet EV.'}
