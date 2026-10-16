"""Fixed HU200 capacity matches; all final means wait for a global byte-verified barrier."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from statistics import mean, stdev
import subprocess
from time import perf_counter

from scripts import evaluate_hu100_direct as direct
from scripts.run_hu200_capacity_training import SEEDS, pin, put
from scripts.run_hu200_frozen_evaluation import series, estimate
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy
from src.blueprint import policy_cache
from src.policies.files import file_hash


def read(path):
    return json.loads(Path(path).read_text())


def cases(seed, kind, pilot=False):
    """Panels share one root across named scenarios; every other stream is disjoint."""
    index = SEEDS.index(seed)
    root = 202610160300 + 10 * index
    if kind == '5B':
        if pilot:
            raise ValueError('No extra 5B pilot root was reserved')
        return [dict(name='direct-5B', opponent=None, blocks=131072, root=root + 4)]
    if kind != '20B':
        raise ValueError('Only fixed 5B and 20B models admitted')
    return [dict(name='direct-20B', opponent=None, blocks=1024 if pilot else 131072,
                 root=root + (1 if pilot else 3)), *[
        dict(name='panel-' + opponent, opponent=opponent, blocks=64 if pilot else 16384,
             root=root + (2 if pilot else 5)) for opponent in direct.PANEL_OPPONENTS]]


def checked_config(path):
    config = read(path)
    specs = config['specs']
    if (config['seed'] not in SEEDS or config['kind'] not in ('5B', '20B')
            or len(specs) != 2 or specs[0]['name'] == specs[1]['name']):
        raise ValueError('Foreign capacity evaluation configuration')
    options = asdict(TranslationOptions(max_states=512, max_events=128))
    for spec in specs:
        if (spec['abstraction'] != HU200_EQUITY_SCHEMA or spec['translation'] != options
                or spec.get('missing_key_fallback', 'uniform') != 'uniform'):
            raise ValueError('Capacity play recipe differs')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if source != config['source'] or subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Clean pinned evaluation source required')
    return config


def build_cache(config, index):
    spec = config['specs'][index]
    cache = Path(config['cache'])
    target = policy_cache.folder(cache, spec['sha256'], spec['abstraction'])
    if target.exists():
        raise ValueError('Cache already exists: never build a policy cache twice')
    tick = perf_counter()
    model = AveragePolicy(Path(spec['path']), spec['sha256'], expected_schema=spec['abstraction'],
        card_buckets=spec['card_buckets'], sorted_rows=True, cache=cache)
    if (model.description['entries'] != spec['entries'] or model.description['iteration'] != spec['iteration']
            or model.description['source_checkpoint_sha256'] != spec['source_checkpoint_sha256']):
        raise ValueError('Loaded cache identity differs')
    put(Path(config['receipts']) / ('cache-' + spec['name'] + '.json'),
        dict(status='verified', model=spec, seconds=perf_counter()-tick,
             manifest=pin(target / 'manifest.json'), files=read(target / 'manifest.json')['files']))


def run_phase(config, phase, pilot):
    root = Path(config['output']) / ('pilot' if pilot else 'final')
    # A final admission is published after cost-only pilots and root reservation.
    if not pilot:
        admission = read(config['final_admission'])
        if (admission['status'] != 'admitted' or admission['config_sha256'] != config['_file_sha256']
                or not admission['roots_fresh'] or not admission['pr_description_readback_verified']):
            raise ValueError('Missing pre-final-hand admission')
    registry = None
    timing = {}
    for case in cases(config['seed'], config['kind'], pilot):
        out = root / case['name']
        args = dict(stack_bb=200, opponent=case['opponent'], cache=Path(config['cache']), cache_read_only=True)
        if phase in ('play', 'reproduce'):
            registry, done = direct.execute(config['specs'], case['blocks'], case['root'], case['name'],
                out / ('play' if phase == 'play' else 'reproduction'), config['source'], registry,
                reproduce=out / 'play' if phase == 'reproduce' else None, **args)
            timing[case['name']] = done
        elif phase == 'replay':
            timing[case['name']] = direct.audit(out / 'play', out / 'replay.json', report=False, cost_only=pilot)
        else:
            raise ValueError('Foreign match phase')
    put(root / (phase + '-complete.json'), dict(status='complete', cases=timing,
        outcomes_inspected=False, registry_reused_across_compatible_matches=phase != 'replay'))


def verify_group(config, pilot=False):
    root = Path(config['output']) / ('pilot' if pilot else 'final')
    checked = []
    for case in cases(config['seed'], config['kind'], pilot):
        folder = root / case['name']
        expected_hands = case['blocks'] * (4 if case['opponent'] else 2)
        replay = read(folder / 'replay.json')
        if replay['status'] != 'verified' or replay['hands_replayed'] != expected_hands:
            raise ValueError('Every-hand independent replay is incomplete')
        pins = []
        for arm in ('play', 'reproduction'):
            inputs = read(folder / arm / 'inputs.json')
            done = read(folder / arm / 'complete.json')
            plan = direct.make_plan(config['specs'], case['blocks'], case['root'], case['name'],
                stack_bb=200, opponent=case['opponent'])
            if (inputs['plan'] != json.loads(direct.canonical(asdict(plan))) or inputs['models'] != config['specs']
                    or inputs['source'] != config['source'] or done['status'] != 'complete'
                    or done['hands'] != expected_hands
                    or (arm == 'reproduction' and not done['reproduced_all_hands_and_decisions'])):
                raise ValueError('Foreign or incomplete exact reproduction')
            for name, expected in done['files'].items():
                value = pin(folder / arm / name)
                if value['bytes'] != expected['bytes'] or value['sha256'] != expected['sha256']:
                    raise ValueError('Verified final raw bytes changed')
                pins.append(value)
            pins += [pin(folder / arm / name) for name in ('inputs.json', 'complete.json')]
        pins.append(pin(folder / 'replay.json'))
        checked.append(dict(case=case, raw_and_receipts=pins))
    result = dict(status='verified', seed=config['seed'], kind=config['kind'], pilot=pilot,
        config_sha256=config['_file_sha256'], all_hands_replayed_and_reproduced=True, cases=checked)
    put(root / 'verified.json', result)
    return result


def cost_quote(config):
    """Unpooled pilot work, including exact reproduction and independent replay."""
    root = Path(config['output']) / 'pilot'
    verified = read(root / 'verified.json')
    if verified['status'] != 'verified' or not verified['pilot'] or config['kind'] != '20B':
        raise ValueError('Verified cost-only 20B pilots required')
    primary, repeated, replayed = (read(root / (name + '-complete.json'))['cases']
                                  for name in ('play', 'reproduce', 'replay'))
    quotes = {}
    for case in cases(config['seed'], '20B', True):
        name = case['name']
        final_count = 16384 if case['opponent'] else 131072
        fixed = (primary[name]['load_or_validation_seconds']
                 + repeated[name]['load_or_validation_seconds'] + replayed[name]['fixed_setup_seconds'])
        scaled = (primary[name]['play_seconds'] + repeated[name]['play_seconds']
                  + replayed[name]['block_work_seconds']) * final_count / case['blocks']
        trace_bytes = sum(p['bytes'] for run in (primary[name], repeated[name]) for p in run['files'].values())
        quotes[name] = dict(fixed_seconds=fixed, scalable_seconds=scaled,
            projected_trace_bytes_with_1_25_margin=int(1.25 * trace_bytes * final_count / case['blocks']),
            projected_seconds_with_1_5_margin=1.5 * (fixed + scaled),
            pilot_blocks=case['blocks'], final_blocks=final_count,
            paired_sd_bb_per_100=replayed[name]['paired_sd_bb_per_100'])
    # The separate 5B direct has the same fixed count and selected table capacity.
    # Use the 20B direct cost without a speed credit; admit again before its hands.
    projected = sum(q['projected_seconds_with_1_5_margin'] for q in quotes.values())
    projected += quotes['direct-20B']['projected_seconds_with_1_5_margin']
    return dict(seed=config['seed'], cases=quotes, projected_all_nine_cases_seconds=projected,
                five_B_direct_cost_proxy='20B direct; no assumed speed credit',
                projected_trace_bytes_with_1_25_margin=sum(q['projected_trace_bytes_with_1_25_margin'] for q in quotes.values())
                    + quotes['direct-20B']['projected_trace_bytes_with_1_25_margin'],
                includes_play_replay_reproduction=True, excludes_cold_cache_construction=True,
                outcomes_inspected=False, pooled_with_final=False)


def global_barrier(config_paths, destination):
    groups = []
    for path in config_paths:
        config = checked_config(path)
        checked = read(Path(config['output']) / 'final/verified.json')
        if (checked['status'] != 'verified' or checked['pilot']
                or checked['config_sha256'] != file_hash(path)
                or not checked['all_hands_replayed_and_reproduced']):
            raise ValueError('Global every-hand barrier incomplete')
        if [item['case'] for item in checked['cases']] != cases(config['seed'], config['kind']):
            raise ValueError('Global final cases differ')
        for item in checked['cases']:
            folder = Path(config['output']) / 'final' / item['case']['name']
            expected_paths = {str(folder / arm / name) for arm in ('play', 'reproduction')
                for name in ('hands.jsonl.gz', 'decisions.jsonl.gz', 'inputs.json', 'complete.json')}
            expected_paths.add(str(folder / 'replay.json'))
            actual_paths = [old['path'] for old in item['raw_and_receipts']]
            if len(actual_paths) != 9 or set(actual_paths) != expected_paths:
                raise ValueError('Global raw and replay receipt coverage incomplete')
            for old in item['raw_and_receipts']:
                current = pin(Path(old['path']))
                if current['bytes'] != old['bytes'] or current['sha256'] != old['sha256']:
                    raise ValueError('Global final bytes changed')
        groups.append(dict(config=str(path), config_sha256=file_hash(path), receipt=checked))
    if sorted((g['receipt']['seed'], g['receipt']['kind']) for g in groups) != sorted(
            (seed, kind) for seed in SEEDS for kind in ('5B', '20B')):
        raise ValueError('All six seed/model groups must verify before outcomes are read')
    put(destination, dict(status='verified', groups=groups, primary_hands=2949120,
        independently_replayed=True, exactly_reproduced=True, outcomes_inspected=False))


def readout(barrier, destination):
    sealed = read(barrier)
    if sealed['status'] != 'verified' or len(sealed['groups']) != 6 or not sealed['exactly_reproduced']:
        raise ValueError('Missing global final barrier')
    if sorted((g['receipt']['seed'], g['receipt']['kind']) for g in sealed['groups']) != sorted(
            (seed, kind) for seed in SEEDS for kind in ('5B', '20B')):
        raise ValueError('Missing seed/model group at readout')
    results = []
    # Validate every retained raw file again before computing any mean.
    for group in sealed['groups']:
        if file_hash(group['config']) != group['config_sha256']:
            raise ValueError('Readout configuration changed')
        for item in group['receipt']['cases']:
            for old in item['raw_and_receipts']:
                if Path(old['path']).stat().st_size != old['bytes'] or file_hash(old['path']) != old['sha256']:
                    raise ValueError('Readout raw bytes changed')
    for group in sealed['groups']:
        config = read(group['config'])
        for case in cases(config['seed'], config['kind']):
            folder = Path(config['output']) / 'final' / case['name']
            values = series(folder, case['blocks'], panel=case['opponent'] is not None)
            scores = dict(candidate=estimate(values['candidate']))
            if case['opponent']:
                scores.update(baseline=estimate(values['baseline']), paired_difference=estimate([
                    a-b for a,b in zip(values['candidate'], values['baseline'])]))
            coverage = read(folder / 'replay.json')['coverage']
            for cell in coverage.values():
                cell['missing_share'] = cell['lookup'].get('missing-key', 0) / cell['decisions']
                cell['zero_mass_share'] = cell['lookup'].get('zero-mass', 0) / cell['decisions']
            selected = {}
            with gzip.open(folder / 'play/decisions.jsonl.gz', 'rt') as stream:
                for row in map(json.loads, stream):
                    if 'translation' not in row:
                        continue
                    info = row['translation']
                    support = ('translated-positive-mass' if info['mode'] == 'translated' else
                        'missing-fallback' if info['selected_key'] is None else
                        'zero-mass' if row['lookup'] == 'zero-mass' else 'exact-positive-mass')
                    cell = selected.setdefault(row['policy'] + '/' + row['street'], {})
                    cell[support] = cell.get(support, 0) + 1
            for cell in selected.values():
                total = sum(cell.values())
                cell['decisions'] = total
                cell['missing_fallback_share'] = cell.get('missing-fallback', 0) / total
                cell['zero_mass_share'] = cell.get('zero-mass', 0) / total
            results.append(dict(seed=config['seed'], kind=config['kind'], case=case,
                scores=scores, played_exact_key_coverage=coverage, played_selected_support=selected))
    dispersion = {}
    for kind in ('5B', '20B'):
        points = [r['scores']['candidate']['bb_per_100'] for r in results
                  if r['kind'] == kind and r['case']['opponent'] is None]
        dispersion[kind] = dict(seed_points=points, mean=mean(points), sample_sd=stdev(points),
                                range=[min(points), max(points)], training_seeds=3)
    put(destination, dict(status='readout', results=results, direct_seed_dispersion=dispersion,
        interval_scope='Conditional paired Student-t95% over independent duplicate blocks; no strength gate',
        global_barrier_sha256=file_hash(barrier)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('cache', 'play', 'replay', 'reproduce', 'verify', 'quote', 'barrier', 'readout'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--configs', nargs='+', type=Path)
    parser.add_argument('--index', type=int, choices=(0, 1))
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--barrier', type=Path)
    args = parser.parse_args()
    if args.phase == 'barrier':
        global_barrier(args.configs, args.out)
    elif args.phase == 'readout':
        readout(args.barrier, args.out)
    else:
        config = checked_config(args.config)
        config['_file_sha256'] = file_hash(args.config)
        if args.phase == 'cache':
            build_cache(config, args.index)
        elif args.phase == 'verify':
            verify_group(config, args.pilot)
        elif args.phase == 'quote':
            put(args.out, cost_quote(config))
        else:
            run_phase(config, args.phase, args.pilot)


if __name__ == '__main__':
    main()
