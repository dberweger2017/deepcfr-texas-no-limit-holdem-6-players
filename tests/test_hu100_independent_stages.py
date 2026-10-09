"""Regression checks for shared policy state and independent stage costs."""
import json
import subprocess
from pathlib import Path

import pytest

from scripts import run_hu100_independent_stages as stages
from scripts.evaluate_native_hu100_baseline import execute, make_plan
from scripts.audit_native_hu100_baseline import audit
from tests.test_native_hu100_baseline import model_fixture
from src.policies.files import file_hash


def test_reused_registry_reproduces_and_resets_translation(tmp_path):
    _, model, path = model_fixture(tmp_path)
    settings = json.loads(Path('configs/arena/hu100-playing-baseline-v1.json').read_text())
    settings['model'].update(name='fixture', path=str(path), sha256=file_hash(path),
        bytes=path.stat().st_size, entries=1, iteration=2,
        source_checkpoint_sha256=model.description['source_checkpoint_sha256'])
    settings['action_translation'] = None
    cfg = tmp_path/'config.json'; cfg.write_text(json.dumps(settings))
    revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    registry = stages.LinkedRegistry(make_plan(settings,'random',2,12345))
    first, repeat, fresh = [tmp_path/x for x in ('first','repeat','fresh')]
    execute(cfg, first, 2, 12345, revision, registry=registry)
    audit(first,tmp_path/'audit.json')
    settings['action_translation'] = {'max_states':512,'max_events':128}
    cfg.write_text(json.dumps(settings))
    execute(cfg,tmp_path/'on',2,12345,revision,registry=registry)
    settings['action_translation'] = None
    cfg.write_text(json.dumps(settings))
    execute(cfg,repeat,2,12345,revision,registry=registry,reproduce=first)
    execute(cfg,fresh,2,12345,revision)
    for opponent in stages.previous.config(settings['model'])['opponents']:
        assert (first/opponent/'hands.jsonl').read_bytes() == (fresh/opponent/'hands.jsonl').read_bytes()
    snapshot = next((repeat/'models').iterdir())
    assert snapshot.stat().st_ino == path.stat().st_ino
    assert json.loads((repeat/'costs.json').read_text())['registry_reused']
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        execute(cfg,tmp_path/'bad',2,12345,revision,registry=registry)


def test_quote_separates_fixed_cost_and_never_uses_winnings():
    measurements = [{'endpoint':p,'load_seconds':10,'arms':[
        {'blocks':32,'fixed_seconds':5,'variable_seconds':32},
        {'blocks':512,'fixed_seconds':5,'variable_seconds':512}], 'winnings':-9999}
        for p in ('early','terminal')]
    models = {str(s):{'early':{'entries':7643261},'terminal':{'entries':41010014}} for s in stages.SEEDS}
    a,b = [stages.evaluation_quote(n,measurements,models) for n in (4096,8192)]
    assert a['load_seconds'] == b['load_seconds'] == 60
    assert a['fixed_arm_seconds'] == b['fixed_arm_seconds'] == 45
    assert b['play_replay_reproduction_seconds']-a['play_replay_reproduction_seconds'] == 3*9*4096
    for m in measurements: m['winnings']=99999
    assert stages.evaluation_quote(4096,measurements,models) == a
