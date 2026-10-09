"""Strict per-seed paired inference after full HU100 replay and reproduction."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from math import isclose
from pathlib import Path
from statistics import stdev

from scripts.evaluate_native_hu100_baseline import OPPONENTS
from scripts.hu100_qualification_guard import read, put
from scripts.report_native_hu100_learning_curves import frozen_schedule, interval
from scripts.run_hu100_seed_qualification import OUT, SEEDS, config
from src.arena.schedule import canonical
from src.policies.files import file_hash


def traces(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def effect(values, alpha):
    result = {'descriptive_95': interval(values), 'adjusted': interval(values, alpha)}
    low, high = result['adjusted']['interval']
    result.update(label='improvement' if low > 0 else 'decline' if high < 0 else 'inconclusive',
        practically_supported=low > 10)
    return result


def report(out=OUT, *, settings_factory=config):
    freeze = read(out / 'frozen-final.json')
    comparisons = read(out / 'frozen-comparisons.json')
    models = read(out / 'models.json')
    if file_hash(out / 'models.json') != freeze['models_sha256'] or file_hash(out / 'frozen-comparisons.json') != freeze['comparisons_sha256']:
        raise ValueError('Frozen models/comparisons changed')
    if comparisons['seeds'] != list(SEEDS) or comparisons['growth_alpha'] != .05/6 or comparisons['translation_alpha'] != .05/3:
        raise ValueError('Frozen comparison families changed')
    if comparisons['practical_lower_bb100'] != 10 or comparisons['outcomes_inspected']:
        raise ValueError('Frozen practical rule changed')
    blocks, root = freeze['blocks_per_opponent'], freeze['final_root']
    if blocks not in (4096, 8192) or blocks != comparisons['blocks']:
        raise ValueError('Final count differs')
    expected = frozen_schedule({'models': [models[str(SEEDS[0])]['early']]}, blocks, root)
    if file_hash(out / 'frozen-schedule.json') != freeze['schedule_sha256'] or canonical(read(out / 'frozen-schedule.json')) != canonical(expected):
        raise ValueError('Frozen paired schedule changed')
    source = freeze['source']
    baseline = {}
    values, hand_digests, coverage, translations, visits = {}, {}, [], [], []
    hands_replayed = actions_replayed = 0
    visit_tables = {}
    for seed in SEEDS:
        for label in ('early', 'terminal', 'terminal-on'):
            name = f'{seed}-{label}'
            model = models[str(seed)]['terminal' if label == 'terminal-on' else label]
            run = out / 'final' / name
            reproduction = out / 'final-reproduction' / name
            audit = read(out / 'final' / (name + '-audit.json'))
            repeat = read(reproduction / 'complete.json')
            inputs = read(run / 'inputs.json')
            if (audit['status'] != 'verified' or not audit['all_settlements_replayed'] or not repeat['reproduced_all_hands_and_decisions']
                    or inputs['source'] != source or inputs['root'] != root or inputs['blocks_per_opponent'] != blocks
                    or inputs['config'] != settings_factory(model, label == 'terminal-on')):
                raise ValueError('Unverified arm identity/replay/reproduction: ' + name)
            hands_replayed += audit['hands_replayed']
            actions_replayed += audit['actions_replayed']
            # Both primary and reproduced artifacts must retain their bound bytes.
            for artifact in (run, reproduction):
                for rel, pin in read(artifact / 'output-files.json').items():
                    path = artifact / rel
                    if path.stat().st_size != pin['bytes'] or file_hash(path) != pin['sha256']:
                        raise ValueError('Changed bound arm artifact: ' + str(path))
            selected_keys = set()
            for opponent in OPPONENTS:
                panel = run / opponent
                manifest = read(panel / 'manifest.json')
                if manifest['revision'] != source or manifest['dirty'] or manifest['plan']['models'][0]['sha256'] != model['sha256']:
                    raise ValueError('Source/model manifest differs')
                if canonical(read(panel / 'schedule.json')['blocks']) != canonical(expected['panels'][opponent]['blocks']):
                    raise ValueError('Physical schedule differs')
                coords = {}
                reference_hash, candidate_hash = hashlib.sha256(), hashlib.sha256()
                with (panel / 'hands.jsonl').open() as rows:
                    for line in rows:
                        row = json.loads(line)
                        if row['arm'] == 'baseline':
                            reference_hash.update(canonical(row).encode() + b'\n')
                            continue
                        coord = row['block'], row['rotation']
                        if row['status'] != 'completed' or coord in coords:
                            raise ValueError('Failed/duplicate final hand')
                        coords[coord] = row['candidate_chips']
                        candidate_hash.update(canonical(row).encode() + b'\n')
                reference = reference_hash.hexdigest()
                if opponent not in baseline:
                    baseline[opponent] = reference
                elif reference != baseline[opponent]:
                    raise ValueError('Uniform reference not reused exactly')
                if set(coords) != {(b, r) for b in range(blocks) for r in (0, 1)}:
                    raise ValueError('Incomplete final paired coordinates')
                series = [(coords[b, 0] + coords[b, 1])/2 for b in range(blocks)]
                recorded = read(panel / 'report.json')
                if not isclose(interval(series)['bb_per_100'], recorded['scenarios'][opponent]['comparison']['candidate']['bb_per_100'], abs_tol=1e-8):
                    raise ValueError('Independent chip mean differs')
                values[seed, label, opponent] = series
                hand_digests[seed, label, opponent] = candidate_hash.hexdigest()
                for row in recorded['diagnostics']:
                    if row['arm'] == 'candidate' and row['logical_player'] == 0:
                        coverage.append({'seed': seed, 'option': label, 'opponent': opponent, **row})
                groups = {}
                for d in traces(panel / 'decisions.jsonl.gz'):
                    if d['arm'] != 'candidate' or d['logical_player'] != 0:
                        continue
                    selected_keys.add(d['key'])
                    g = groups.setdefault(d['street'], {'count': 0, 'modes': Counter(), 'distance_sum': 0,
                        'distance_count': 0, 'distance_max': None, 'states_max': 0, 'bounds': 0, 'lookup_sum': 0})
                    t = d['translation']
                    g['count'] += 1
                    g['modes'][t['mode']] += 1
                    g['states_max'] = max(g['states_max'], t['states'])
                    g['bounds'] += t['bound_reached']
                    g['lookup_sum'] += t['lookup_seconds']
                    if t['mode'] == 'translated':
                        g['distance_count'] += 1
                        g['distance_sum'] += t['distance']
                        g['distance_max'] = max(g['distance_max'] or 0, t['distance'])
                for street, g in groups.items():
                    counts = g['modes']
                    translations.append({'seed': seed, 'option': label, 'opponent': opponent, 'street': street,
                        'decisions': g['count'], 'counts': dict(counts),
                        'rates': {mode: counts[mode]/g['count'] for mode in ('exact', 'translated', 'uniform')},
                        'distance_mean': g['distance_sum']/g['distance_count'] if g['distance_count'] else None,
                        'distance_max': g['distance_max'], 'states_max': g['states_max'],
                        'bounds_reached': g['bounds'], 'lookup_mean_ms': 1000*g['lookup_sum']/g['count']})
            if model['sha256'] not in visit_tables:
                related = [label] if label == 'early' else ['terminal', 'terminal-on']
                union_keys = set(selected_keys)
                for option in related:
                    for opponent in OPPONENTS:
                        for d in traces(out / 'final' / f'{seed}-{option}' / opponent / 'decisions.jsonl.gz'):
                            if d['arm'] == 'candidate' and d['logical_player'] == 0:
                                union_keys.add(d['key'])
                table = {}
                with gzip.open(model['path'], 'rt') as f:
                    next(f)
                    for line in f:
                        row = json.loads(line)
                        if row[0] in union_keys:
                            table[row[0]] = (row[3], row[4])
                if file_hash(Path(model['path'])) != model['sha256']:
                    raise ValueError('Visit-band model changed')
                visit_tables[model['sha256']] = table
            table = visit_tables[model['sha256']]
            for opponent in OPPONENTS:
                groups = defaultdict(Counter)
                for d in traces(run / opponent / 'decisions.jsonl.gz'):
                    if d['arm'] != 'candidate' or d['logical_player'] != 0:
                        continue
                    known = table.get(d['key'])
                    lookup = 'missing-key' if known is None else 'positive-mass-known-key' if known[0] > 0 else 'zero-mass'
                    if lookup != d['lookup']:
                        raise ValueError('Visit lookup recount differs')
                    n = known[1] if known else None
                    band = 'missing' if n is None else '0' if n == 0 else '1' if n == 1 else '2-9' if n < 10 else '10-99' if n < 100 else '100+'
                    groups[d['street']][band] += 1
                for street, bands in groups.items():
                    visits.append({'seed': seed, 'option': label, 'opponent': opponent,
                        'street': street, 'decisions': sum(bands.values()), 'bands': dict(bands)})
    per_seed = []
    for seed in SEEDS:
        growth, translation, absolute, controls = {}, {}, {}, {}
        for opponent in OPPONENTS:
            early, terminal, on = [values[seed, label, opponent] for label in ('early', 'terminal', 'terminal-on')]
            growth[opponent] = effect([a-b for a, b in zip(terminal, early, strict=True)], .05/6 if opponent in ('tight_aggressive', 'loose_aggressive') else .05)
            translation[opponent] = effect([a-b for a, b in zip(on, terminal, strict=True)], .05/3 if opponent == 'pot_pressure' else .05)
            absolute[opponent] = {label: interval(values[seed, label, opponent]) for label in ('early', 'terminal', 'terminal-on')}
            controls[opponent] = hand_digests[seed, 'terminal', opponent] == hand_digests[seed, 'terminal-on', opponent]
            if opponent in ('check_call', 'tight_aggressive', 'loose_aggressive') and not controls[opponent]:
                raise ValueError('On-menu action/event/settlement control changed')
        flag = translation['random']['descriptive_95']['interval'][1] < -20
        per_seed.append({'seed': seed, 'growth': growth, 'translation': translation, 'absolute': absolute,
            'identical_translation_candidate_hands': controls, 'random_severe_regression_flag': flag,
            'terminal_actual_nodes': models[str(seed)]['terminal']['actual_nodes']})
    repeated = {}
    for family, opponent in (('growth', 'tight_aggressive'), ('growth', 'loose_aggressive'), ('translation', 'pot_pressure')):
        effects = [s[family][opponent] for s in per_seed]
        estimates = [e['adjusted']['bb_per_100'] for e in effects]
        repeated[family + '-' + opponent] = {'repeat_improvement': all(e['label'] == 'improvement' for e in effects),
            'repeat_practical': all(e['practically_supported'] for e in effects),
            'seed_estimates': estimates, 'descriptive_seed_range': [min(estimates), max(estimates)],
            'descriptive_seed_sample_sd': stdev(estimates)}
    complete = all(s['terminal_actual_nodes'] >= 1_000_000_000 for s in per_seed)
    scientific = complete and all(e['repeat_improvement'] for e in repeated.values()) and not any(s['random_severe_regression_flag'] for s in per_seed)
    result = {'status': 'science-verified', 'source': source, 'blocks_per_opponent': blocks,
        'all_final_hands_replayed_and_reproduced': True, 'all_fixed_1b_endpoints_completed': complete,
        'unique_final_hands': blocks*100, 'replayed_rows_including_reference_copies': hands_replayed,
        'replayed_actions_including_reference_copies': actions_replayed, 'per_seed': per_seed,
        'repetition': repeated, 'coverage': coverage, 'translation_telemetry': translations, 'visits': visits,
        'deal_uncertainty': 'paired independent-block Student-t conditional on each policy',
        'seed_variation': 'three descriptive estimates/range/sample SD; common deals correlate estimates; no population or pooled-hand claim',
        'statistical_families': 'growth six Bonferroni contrasts; translation three separately adjusted contrasts, not union FWER .05',
        'science_supports_benchmark_preparation': scientific,
        'recipe_qualification': 'pending archive acceptance and independent evidence review even if scientific gates pass',
        'uniform_reference_baseline_sha256': baseline}
    put(out / 'result.json', result)
    return result


if __name__ == '__main__':
    report()
