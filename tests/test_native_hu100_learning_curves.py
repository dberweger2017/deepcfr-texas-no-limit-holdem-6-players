"""Pairing, uniform reuse, multiplicity and outcome-blind budgeting regressions."""

import gzip
import json
from pathlib import Path
import subprocess

import pytest

from scripts.audit_native_hu100_baseline import audit
from scripts.evaluate_native_hu100_baseline import execute
from scripts.report_native_hu100_learning_curves import frozen_schedule, interval
from scripts.run_native_hu100_learning_curves import quote
from src.policies.files import file_hash
from tests.test_native_hu100_baseline import model_fixture


def test_reuse_reference_plays_candidate_only_and_reproduces(tmp_path, monkeypatch):
    _, model, path = model_fixture(tmp_path)
    config = json.loads(Path('configs/arena/hu100-playing-baseline-v1.json').read_text())
    config['model'].update(name='fixture', path=str(path), sha256=file_hash(path),
                           bytes=path.stat().st_size, entries=1, iteration=2,
                           source_checkpoint_sha256=model.description['source_checkpoint_sha256'])
    config_path = tmp_path / 'config.json'; config_path.write_text(json.dumps(config))
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    first = tmp_path / 'first'
    execute(config_path, first, 2, 4321, source)
    import scripts.evaluate_native_hu100_baseline as evaluator
    original = evaluator.run_schedule
    actual_arms = []
    def run(*args, **kwargs):
        actual_arms.append(kwargs['arms'])
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluator, 'run_schedule', run)
    second = tmp_path / 'second'
    r = execute(config_path, second, 2, 4321, source, reference_run=first)
    assert actual_arms == [('candidate',)] * 5
    assert r['hands'] == 40 and r['hands_actually_played'] == 20
    assert audit(second, tmp_path / 'audit.json')['hands_replayed'] == 40
    assert execute(config_path, tmp_path / 'repeat', 2, 4321, source,
                   reference_run=first, reproduce=second)['reproduced_all_hands_and_decisions']
    with pytest.raises(ValueError, match='identical schedule'):
        execute(config_path, tmp_path / 'wrong', 2, 999, source, reference_run=first)
    # Changing a cached trace cannot pass by recomputing the outer file inventory.
    trace = second / 'random/decisions.jsonl.gz'
    with gzip.open(trace, 'rt') as f:
        rows = [json.loads(line) for line in f]
    next(r for r in rows if r['arm'] == 'baseline')['seconds'] += .1
    with gzip.open(trace, 'wt') as f:
        f.write(''.join(json.dumps(r) + '\n' for r in rows))
    inventory = json.loads((second / 'output-files.json').read_text())
    inventory['random/decisions.jsonl.gz'].update(bytes=trace.stat().st_size, sha256=file_hash(trace))
    (second / 'output-files.json').write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match='uniform evidence differs'):
        audit(second, tmp_path / 'bad.json')


def test_schedule_is_model_independent_and_fresh():
    config = json.loads(Path('configs/arena/hu100-learning-curves-v1.json').read_text())
    first = frozen_schedule(config, 32, config['final_root'])
    config['models'].reverse()
    assert frozen_schedule(config, 32, config['final_root']) == first
    pilot = frozen_schedule(config, 16, config['pilot_root'])
    deals = lambda s: {b['deal_seeds'][0] for p in s['panels'].values() for b in p['blocks']}
    assert not deals(first) & deals(pilot)


def test_bonferroni_family_and_uniform_budget():
    values = list(range(64))
    normal, adjusted = interval(values), interval(values, .05 / 20)
    assert normal['bb_per_100'] == adjusted['bb_per_100']
    assert adjusted['interval'][0] < normal['interval'][0]
    assert adjusted['interval'][1] > normal['interval'][1]
    pilot = {'panel_costs': [{'seconds': 10}] * 25, 'wall_seconds': 270,
             'blocks_per_opponent': 16, 'profit': -1e9}
    repeat = {**pilot}
    replay = {'seconds': 80, 'profit': -1e9}
    result = quote(pilot, replay, repeat, 1700)
    assert result['final_hands'] == result['blocks_per_opponent'] * 60
    assert result['predicted_seconds'] + result['closeout_reserve_seconds'] <= 1700
    pilot['profit'] = 1e9; replay['profit'] = 1e9
    assert quote(pilot, replay, repeat, 1700) == result
