"""Final capacity readout refuses incomplete verification and changed raw bytes."""
from dataclasses import asdict
import json
from pathlib import Path
import pytest
from scripts import evaluate_hu200_capacity as capacity
from scripts.run_hu200_capacity_training import pin
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.average import HU200_FORMAT


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def config(tmp_path):
    specs = [dict(name=name, path=str(tmp_path / name), sha256='a'*64,
                  format=HU200_FORMAT, abstraction=HU200_EQUITY_SCHEMA,
                  card_buckets=str(tmp_path / 'tables')) for name in ('large', 'twin')]
    return dict(seed=2026101302, kind='5B', specs=specs, output=str(tmp_path / 'out'),
                source='source', _file_sha256='config')


def trusted_replay_fixture(config):
    case, = capacity.cases(config['seed'], config['kind'])
    folder = Path(config['output']) / 'final' / case['name']
    plan = capacity.direct.make_plan(config['specs'], case['blocks'], case['root'], case['name'], stack_bb=200)
    for arm in ('play', 'reproduction'):
        raw = folder / arm / 'hands.jsonl.gz'
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b'previously replayed raw hands')
        decisions = raw.with_name('decisions.jsonl.gz')
        decisions.write_bytes(b'previously replayed decisions')
        put(folder / arm / 'complete.json', dict(status='complete', hands=262144,
            reproduced_all_hands_and_decisions=arm == 'reproduction', files={
                p.name:{'bytes':p.stat().st_size, 'sha256':pin(p)['sha256']} for p in (raw, decisions)}))
        put(folder / arm / 'inputs.json', dict(plan=asdict(plan), models=config['specs'], source='source'))
    put(folder / 'replay.json', dict(status='verified', hands_replayed=262144))
    return folder


def test_changed_raw_cannot_cross_verification_barrier(tmp_path):
    cfg = config(tmp_path)
    folder = trusted_replay_fixture(cfg)
    (folder / 'reproduction/decisions.jsonl.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='raw bytes changed'):
        capacity.verify_group(cfg)
    assert not (Path(cfg['output']) / 'final/verified.json').exists()


def test_one_complete_seed_group_cannot_release_outcomes(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    trusted_replay_fixture(cfg)
    capacity.verify_group(cfg)
    path = tmp_path / 'config.json'
    put(path, {k:v for k,v in cfg.items() if not k.startswith('_')})
    checked = Path(cfg['output']) / 'final/verified.json'
    receipt = json.loads(checked.read_text())
    receipt['config_sha256'] = capacity.file_hash(path)
    put(checked, receipt)
    monkeypatch.setattr(capacity, 'checked_config', lambda path: json.loads(path.read_text()))
    with pytest.raises(ValueError, match='All six'):
        capacity.global_barrier([path], tmp_path / 'barrier.json')
    assert not (tmp_path / 'barrier.json').exists()


def test_pilot_or_missing_seed_barrier_cannot_read_outcomes(tmp_path):
    barrier = tmp_path / 'barrier.json'
    put(barrier, dict(status='verified', groups=[], exactly_reproduced=True))
    with pytest.raises(ValueError, match='global final barrier'):
        capacity.readout(barrier, tmp_path / 'scores.json')
    assert not (tmp_path / 'scores.json').exists()
