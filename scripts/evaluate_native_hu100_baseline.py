"""Small HU100 adapter around the existing registry, paired runner and reporter."""

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from statistics import mean
import subprocess
from time import perf_counter, time

import numpy as np

from src.arena.artifacts import manifest, write_json
from src.arena.catalog import Checkpoint
from src.arena.registry import PolicyRegistry
from src.arena.report import performance, summarize
from src.arena.runner import run_schedule
from src.arena.schedule import Plan, Scenario, build_schedule, canonical, schedule_document, stream_seed
from src.blueprint.abstraction import HU100_SCHEMA, information_key
from src.blueprint.action_translation import TranslationOptions
from src.policies.files import file_hash

OPPONENTS = ('random', 'check_call', 'tight_aggressive', 'loose_aggressive', 'pot_pressure')
REFERENCE = 'native_hu100_uniform'


def action_seed(root, block, rotation, arm, logical_player):
    return stream_seed(root, 'test', 'action', 'pr197-hu100-v1',
                       block.scenario, block.index, rotation, arm, logical_player)


def make_plan(settings, opponent, blocks, root):
    model = Checkpoint(**{k: settings['model'][k] for k in ('name', 'path', 'sha256', 'format')})
    return Plan((Scenario(opponent, (10000, 10000)),), candidate=model.name,
                baseline=REFERENCE, opponents=(opponent,), blocks=blocks,
                root_seed=root, split='test', max_decisions=1000, models=(model,))


class Probe:
    """Observe after the original sampler; telemetry never consumes an action draw."""

    def __init__(self, player, model, context, emit):
        self.player, self.model, self.context, self.emit = player, model, context, emit

    def choose_action(self, view):
        started = perf_counter()
        action = self.player.choose_action(view)
        elapsed = perf_counter() - started
        row = {**self.context, 'hand_id': view.hand_id, 'seat': view.seat,
               'street': view.street.value, 'action': asdict(action), 'seconds': elapsed}
        if self.context['logical_player'] == 0:
            if getattr(self.model, 'record_translation', False) and hasattr(self.player, 'last_decision'):
                menu, probabilities, known, translation = self.player.last_decision
                row['translation'] = json.loads(canonical(translation))
            else:
                menu, probabilities, known = self.model.distribution(view)
            key = information_key(view, menu, schema=HU100_SCHEMA)
            row.update(key=key, lookup='missing-key' if not known else
                       'zero-mass' if key in self.model.zero_mass else 'positive-mass-known-key',
                       menu=[asdict(item) for item in menu],
                       probabilities=list(probabilities) if self.context['arm'] == 'candidate'
                       else [1 / len(menu)] * len(menu))
        self.emit(row)
        return action


def diagnostics(rows):
    groups = {}
    for row in rows:
        key = (row['arm'], row['logical_player'], row['street'])
        group = groups.setdefault(key, {'decisions': 0, 'lookup': {}, 'actions': {}, 'latency': []})
        group['decisions'] += 1
        action = row['action']['kind']
        group['actions'][action] = group['actions'].get(action, 0) + 1
        if 'lookup' in row:
            group['lookup'][row['lookup']] = group['lookup'].get(row['lookup'], 0) + 1
        group['latency'].append(row['seconds'])
    result = []
    for (arm, role, street), g in sorted(groups.items()):
        durations = g.pop('latency')
        result.append({'arm': arm, 'logical_player': role, 'street': street, **g,
                       'lookup_rates': {k: g['lookup'].get(k, 0) / g['decisions']
                                        for k in ('positive-mass-known-key', 'zero-mass', 'missing-key')}
                       if role == 0 else None,
                       'action_rates': {k: n / g['decisions'] for k, n in g['actions'].items()},
                       'inference_latency_ms': {'mean': 1000 * mean(durations),
                           'p50': 1000 * float(np.quantile(durations, .5)),
                           'p95': 1000 * float(np.quantile(durations, .95)),
                           'max': 1000 * max(durations)}})
    return result


def execute(config, out, blocks, root, source, *, reproduce=None, reference_run=None):
    settings = json.loads(config.read_text())
    reference_inputs = None
    if reference_run:
        reference_inputs = json.loads((reference_run / 'inputs.json').read_text())
        if (reference_inputs['root'] != root or reference_inputs['blocks_per_opponent'] != blocks
                or reference_inputs['source'] != source
                or reference_inputs['config']['opponents'] != settings['opponents']
                or not (reference_run / 'complete.json').exists()):
            raise ValueError('Reused reference must have identical schedule/source')
    if (settings['opponents'] != list(OPPONENTS) or settings['reference'] != REFERENCE
            or settings['stacks'] != [10000, 10000] or settings['small_blind'] != 50
            or settings['big_blind'] != 100 or settings['rake'] != 0
            or not 1 <= blocks <= settings['proposed_final_blocks_per_opponent']):
        raise ValueError('Unknown HU100 baseline recipe/budget')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != source:
        raise ValueError('Evaluation source differs from frozen source')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Evaluation requires committed clean tracked source')
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    first = make_plan(settings, OPPONENTS[0], blocks, root)
    model_path = Path(settings['model']['path'])
    if model_path.stat().st_size != settings['model']['bytes']:
        raise ValueError('Pinned model byte size differs')
    load_started = perf_counter()
    registry = PolicyRegistry(first, artifact_dir=reproduce / 'models' if reproduce else None)
    model = registry.models[first.candidate]
    if (model.description['entries'] != settings['model']['entries']
            or model.description['iteration'] != settings['model']['iteration']
            or model.description['source_checkpoint_sha256'] != settings['model']['source_checkpoint_sha256']):
        raise ValueError('Pinned model training identity differs')
    if 'action_translation' in settings:
        option = settings['action_translation']
        if option is not None and not isinstance(option, dict):
            raise ValueError('Invalid action translation config')
        model.configure_translation(TranslationOptions(**option) if option is not None else None)
        model.record_translation = True
        model.description['action_translation'] = option
    load_seconds = perf_counter() - load_started
    registry.snapshot(out)
    write_json(out / 'inputs.json', {'config': settings, 'config_sha256': file_hash(config),
                                   'blocks_per_opponent': blocks, 'root': root, 'source': source,
                                   'reference_run': str(reference_run) if reference_run else None,
                                   'reference_inputs_sha256': file_hash(reference_run / 'inputs.json')
                                   if reference_run else None})
    totals = {'hands': 0, 'decisions': 0}
    costs = []
    all_seeds = set()
    for opponent in OPPONENTS:
        plan = make_plan(settings, opponent, blocks, root)
        registry.plan = plan  # Same validated table/model; only the unchanged rival differs.
        panel = out / opponent
        panel.mkdir()
        write_json(panel / 'manifest.json', manifest(plan, registry))
        write_json(panel / 'schedule.json', schedule_document(plan))
        schedule = build_schedule(plan)
        contexts = {}
        for block in schedule:
            for rotation in (0, 1):
                for arm in ('candidate', 'baseline'):
                    for role in (0, 1):
                        seed = action_seed(root, block, rotation, arm, role)
                        if seed in all_seeds:
                            raise ValueError('Private action seed collision')
                        all_seeds.add(seed)
                        contexts[seed] = {'opponent': opponent, 'block': block.index,
                                         'rotation': rotation, 'arm': arm,
                                         'logical_player': role, 'seed': seed}
        write_json(panel / 'private-action-seeds.json', contexts)
        rows, timings, decisions = [], [], []
        wall = perf_counter()
        with ((panel / 'hands.jsonl').open('x') as hands,
              (panel / 'timings.jsonl').open('x') as latency,
              gzip.open(panel / 'decisions.jsonl.gz', 'wt') as traces):
            def record_decision(row):
                traces.write(canonical(row) + '\n')
                decisions.append(row)

            def factory(name, seed):
                return Probe(registry.make_policy(name, seed), model,
                             {**contexts[seed], 'policy': name}, record_decision)

            def emit(row, timing):
                hands.write(canonical(row) + '\n'); hands.flush()
                latency.write(canonical(timing) + '\n')
                rows.append(row); timings.append(timing)
                write_json(out / 'status.json', {'status': 'playing', 'opponent': opponent,
                                               'completed_hands': totals['hands'] + len(rows),
                                               'source': source, 'at': time()})

            ok = run_schedule(plan, schedule, emit, factory=factory,
                              seed_factory=lambda b, r, a, i: action_seed(root, b, r, a, i),
                              arms=('candidate',) if reference_run else ('candidate', 'baseline'))
            if reference_run:
                # Reuse exact uniform actions and settlements, never sample them again.
                original = reference_run / opponent
                with (original / 'hands.jsonl').open() as f:
                    for line in f:
                        row = json.loads(line)
                        if row['arm'] == 'baseline':
                            hands.write(line); rows.append(row)
                with (original / 'timings.jsonl').open() as f:
                    for line in f:
                        row = json.loads(line)
                        if row['arm'] == 'baseline':
                            latency.write(line); timings.append(row)
                with gzip.open(original / 'decisions.jsonl.gz', 'rt') as f:
                    for line in f:
                        row = json.loads(line)
                        if row['arm'] == 'baseline':
                            traces.write(line); decisions.append(row)
        report = summarize(plan, rows)
        report['performance'] = performance(timings, perf_counter() - wall)
        report['diagnostics'] = diagnostics(decisions)
        write_json(panel / 'report.json', report)
        if not ok or report['status'] != 'valid':
            raise ValueError('Incomplete or illegal panel; stop without retry')
        if reproduce:
            original = reproduce / opponent
            if file_hash(original / 'hands.jsonl') != file_hash(panel / 'hands.jsonl'):
                raise ValueError('Deterministic hand reproduction differs')
            with gzip.open(original / 'decisions.jsonl.gz', 'rt') as old:
                for actual, line in zip(decisions, old, strict=True):
                    expected = json.loads(line)
                    if {k: v for k, v in actual.items() if k not in ('seconds', 'translation')} != {
                            k: v for k, v in expected.items() if k not in ('seconds', 'translation')}:
                        raise ValueError('Deterministic policy/telemetry reproduction differs')
                    if 'translation' in actual:
                        strip = lambda d: {k: v for k, v in d.items() if k != 'lookup_seconds'}
                        if strip(actual['translation']) != strip(expected['translation']):
                            raise ValueError('Deterministic translation reproduction differs')
        totals['hands'] += len(rows); totals['decisions'] += len(decisions)
        costs.append({'opponent': opponent, 'seconds': perf_counter() - wall,
                      'hands': len(rows), 'decisions': len(decisions)})
    receipt = {'status': 'complete', **totals, 'blocks_per_opponent': blocks, 'root': root,
               'hands_actually_played': totals['hands'] // 2 if reference_run else totals['hands'],
               'reference_coverage_model': reference_inputs['config']['model']['sha256']
               if reference_inputs else settings['model']['sha256'],
               'model_load_seconds': load_seconds, 'wall_seconds': perf_counter() - started,
               'panel_costs': costs, 'source': source, 'reproduced_all_hands_and_decisions': bool(reproduce)}
    write_json(out / 'complete.json', receipt)
    write_json(out / 'status.json', receipt)
    write_json(out / 'output-files.json', {str(p.relative_to(out)): {
        'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in out.rglob('*') if p.is_file()})
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--blocks', type=int, required=True); p.add_argument('--root', type=int, required=True)
    p.add_argument('--source', required=True); p.add_argument('--reproduce', type=Path)
    p.add_argument('--reference-run', type=Path)
    a = p.parse_args()
    print(json.dumps(execute(a.config, a.out, a.blocks, a.root, a.source,
                            reproduce=a.reproduce, reference_run=a.reference_run)))


if __name__ == '__main__':
    main()
