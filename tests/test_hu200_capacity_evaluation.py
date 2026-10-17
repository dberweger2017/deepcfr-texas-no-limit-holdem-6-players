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


def test_cache_once_and_reused_match_phases_replay_every_fixture_hand(tmp_path, monkeypatch):
    from tests.test_hu200_evaluation import fixture_spec, synthetic_cards
    from tests.test_v2_consumers import dev_source
    from src.blueprint.action_translation import TranslationOptions
    from src.policies.files import file_hash
    import gzip

    source = dev_source(monkeypatch)
    root, blocks = 202610160399, 2
    initial = [fixture_spec(tmp_path / ('initial-' + name), name=name) for name in ('large', 'twin')]
    cards = synthetic_cards(tmp_path / 'cards', initial, root, blocks, monkeypatch)
    specs = [fixture_spec(tmp_path / name, HU200_EQUITY_SCHEMA, name, cards) for name in ('large', 'twin')]
    # Distinct source pins exercise two cache builds, even for equal toy policies.
    other = Path(specs[1]['path'])
    with gzip.open(other, 'rt') as stream:
        metadata, row = map(json.loads, stream)
    metadata['source_checkpoint_sha256'] = 'b'*64
    with gzip.open(other, 'wt') as stream:
        stream.write(json.dumps(metadata) + '\n' + json.dumps(row) + '\n')
    specs[1].update(sha256=file_hash(other), bytes=other.stat().st_size, source_checkpoint_sha256='b'*64)
    for spec in specs:
        spec['translation'] = asdict(TranslationOptions(max_states=512, max_events=128))
    cfg = dict(seed=2026101302, kind='20B', specs=specs, source=source, _file_sha256='toy-fixture',
        cache=str(tmp_path / 'cache'), receipts=str(tmp_path / 'receipts'), output=str(tmp_path / 'out'))
    fixture_cases = [dict(name='fixture', opponent=None, blocks=blocks, root=root),
                     dict(name='panel-random', opponent='random', blocks=blocks, root=root)]
    monkeypatch.setattr(capacity, 'cases', lambda seed, kind, pilot=False: fixture_cases)
    for index in (0, 1):
        capacity.build_cache(cfg, index)
    with pytest.raises(ValueError, match='never build'):
        capacity.build_cache(cfg, 0)
    for phase in ('play', 'replay', 'reproduce'):
        capacity.run_phase(cfg, phase, pilot=True)
    checked = capacity.verify_group(cfg, pilot=True)
    assert checked['all_hands_replayed_and_reproduced']
    assert len(checked['cases']) == 2
    for case in fixture_cases:
        replay = json.loads((Path(cfg['output']) / 'pilot' / case['name'] / 'replay.json').read_text())
        assert replay['hands_replayed'] == blocks * (4 if case['opponent'] else 2)
        assert 'bb_per_100' not in replay
