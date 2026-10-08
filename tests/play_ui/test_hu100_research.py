"""Complete fixed-100BB hands, deterministic recovery and research rejection."""

from copy import deepcopy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from src.blueprint.action_translation import TranslationOptions, VERSION
from src.blueprint.solver import HU100_GAME
from src.game.types import ActionKind
from src.play_api.configuration import PlayTable
from src.play_api.play_audit import audit_state
from src.play_api.service import PlayError, PlayService, _model_info
from src.play_api.spectator import SpectatorService
from src.play_api.spectator_audit import audit_states
from tests.test_native_hu100_baseline import model_fixture
from tests.test_hu100_action_translation import model, offmenu, stored, witness_key


def fixture_policy(tmp_path, enabled=False):
    path = tmp_path / 'average.gz'
    if path.exists():
        from src.blueprint.average import AveragePolicy
        from src.blueprint.abstraction import HU100_SCHEMA
        from src.policies.files import file_hash
        policy = AveragePolicy(path, file_hash(path), expected_schema=HU100_SCHEMA)
    else:
        _, policy, path = model_fixture(tmp_path)
    from src.policies.files import file_hash
    policy.spec = SimpleNamespace(sha256=file_hash(path))
    policy.name = 'HU100 fixture'
    policy.configure_translation(TranslationOptions() if enabled else None)
    policy.adapter_id = VERSION if enabled else 'direct-v1'
    return policy


def request(state):
    return {'handId': state['hand']['id'], 'revision': state['revision']}


def finish(service, state, prefix):
    for step in range(100):
        if state['phase'] in ('finished', 'complete'):
            return state
        body = request(state)
        if state['hand']['actor'] == 1:
            state = service.advance(state['sessionId'], f'{prefix}-bot-{step:016}', body)
        else:
            kinds = state['hand']['legal']['kinds']
            state = service.act(state['sessionId'], f'{prefix}-human-{step:014}',
                                {**body, 'kind': 'check' if 'check' in kinds else 'call', 'raiseTo': None})
    pytest.fail('Unsettled HU100 hand')


@pytest.mark.parametrize('enabled', [False, True])
def test_complete_hands_accounting_reset_replay_restart(tmp_path, enabled):
    policy = fixture_policy(tmp_path, enabled)
    db = tmp_path / 'play.sqlite'
    service = PlayService(db, policy)
    try:
        state = service.create('research-create-0001', {'sessionType': 'benchmark', 'playMode': 'free', 'targetHands': 4})
        assert state['table']['stack'] == 10000
        assert bool(state['model']['inference']['translation']) == enabled
        for number in range(4):
            state = service.new_hand(state['sessionId'], f'research-hand-{number:016}', {'revision': state['revision']})
            assert state['hand']['button'] == number % 2
            assert sorted(p['stack'] for p in state['hand']['players']) == [9900, 9950]
            if number == 0:
                state = service.act(state['sessionId'], 'research-offmenu-001',
                                    {**request(state), 'kind': 'raise', 'raiseTo': 550})
                # Reopen before bot action and reproduce from the same private RNG.
                clone = tmp_path / 'clone.sqlite'
                target = sqlite3.connect(clone); service.db.backup(target); target.close()
                restarted = PlayService(clone, fixture_policy(tmp_path, enabled))
                try:
                    body = request(state)
                    original = service.advance(state['sessionId'], 'research-lostreply-001', body)
                    assert restarted.advance(state['sessionId'], 'research-lostreply-001', body) == original
                    assert restarted.advance(state['sessionId'], 'research-lostreply-001', body) == original
                    state = original
                finally:
                    restarted.close()
            state = finish(service, state, f'hand{number}')
        assert state['phase'] == 'complete'
        assert service.verify_replay(state['sessionId']) == 4
        report = service.benchmark_result(state['sessionId'])
        assert report['netChips'] == state['sessionChips']
        assert report['table']['stack'] == 10000
        assert report['inference'] == state['model']['inference']
        raw = service._load(state['sessionId'])
        assert all(row['table']['stack'] == 10000 for row in raw['history'])
        assert any(row['botDecisions'] for row in raw['history'])
        assert all('telemetry' in decision for row in raw['history'] for decision in row['botDecisions'])
        public = json.dumps({'state': state, 'history': service.history(state['sessionId'])})
        assert not any(key in public for key in ('dealSeed', 'botRng', 'deck', 'botDecisions'))
        for field in ('menu', 'observation', 'telemetry', 'selectedIndex'):
            evidence = deepcopy(raw)
            decision = next(r for r in evidence['history'] if r['botDecisions'])['botDecisions'][0]
            decision[field] = None
            evidence['current'] = deepcopy(evidence['history'][-1])
            with pytest.raises((AssertionError, TypeError)):
                audit_state(evidence, policy)
    finally:
        service.close()


def test_translation_witness_is_recorded_without_changing_real_wager(tmp_path):
    view = offmenu(); menu, key = witness_key(view)
    policy = model([(key, stored(menu))])
    policy.game = HU100_GAME; policy.spec = SimpleNamespace(sha256='a' * 64)
    policy.description['strategy'] = 'fixture'; policy.adapter_id = VERSION
    service = PlayService(tmp_path / 'translate.sqlite', policy)
    try:
        state = service.create('translate-create-0001', {'playMode': 'free', 'visibility': 'developer'})
        state = service.new_hand(state['sessionId'], 'translate-hand-00001', {'revision': 0})
        raw = service._load(state['sessionId']); raw['current']['dealSeed'] = 12
        # Fixture uses the exact same legal own cards; player IDs are key-neutral.
        service.db.execute('UPDATE sessions SET state=? WHERE id=?', (json.dumps(raw), state['sessionId'])); service.db.commit()
        state = service.act(state['sessionId'], 'translate-raise-0001', {**request(state), 'kind': 'raise', 'raiseTo': 550})
        state = service.advance(state['sessionId'], 'translate-bot-000001', request(state))
        raw = service._load(state['sessionId'])
        receipt = raw['current']['botDecisions'][0]['telemetry']
        assert receipt['mode'] == 'translated' and receipt['selected_key'] == key
        assert raw['current']['actions'][0]['raiseTo'] == 550
        audit_state(raw, policy)
    finally:
        service.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_spectator_100bb_audit_and_resume(tmp_path, enabled):
    policy = fixture_policy(tmp_path, enabled)
    identity = {'version': 'fixture', **_model_info(policy)}
    policies, identities = {'fixture': policy}, {'fixture': identity}
    path = tmp_path / 'spectator.sqlite'
    service = SpectatorService(path, policies, identities)
    try:
        state = service.create('spectator100-create', {'sessionType': 'spectator', 'modelVersions': ['fixture', 'fixture']})
        for number in range(6):
            state = service.new_hand(state['sessionId'], f'spectator100-hand-{number:016}', {'revision': state['revision']})
            assert sorted(p['stack'] for p in state['hand']['perspectives'][0]['players']) == [9900, 9950]
            state = service.advance(state['sessionId'], f'spectator100-first-{number:016}', request(state))
            service.close(); service = SpectatorService(path, policies, identities)
            assert service.state(state['sessionId']) == state
            for step in range(100):
                if state['phase'] == 'finished': break
                state = service.advance(state['sessionId'], f'spectator100-{number}-{step:016}', request(state))
            assert state['phase'] == 'finished'
        assert service.verify_replay(state['sessionId']) == 6
        raw = service._load(state['sessionId'])
        assert audit_states([raw], policies, identities)['hands'] == 6
        assert sum(state['sessionChips']) == 0
    finally:
        service.close()


def test_incompatible_tables_models_and_restart_options(tmp_path):
    policy = fixture_policy(tmp_path)
    with pytest.raises(ValueError, match='incompatible'):
        PlayService(tmp_path / 'wrong.sqlite', policy, table=PlayTable())
    for kwargs in ({'stack': 20000}, {'stack': True}, {'small_blind': 100}, {'chip_unit': '1'}):
        with pytest.raises(ValueError): PlayTable(**kwargs)
    path = tmp_path / 'identity.sqlite'
    service = PlayService(path, policy)
    state = service.create('identity-create-0001', {'playMode': 'free', 'visibility': 'developer'})
    service.close()
    different = fixture_policy(tmp_path, True)
    reopened = PlayService(path, different)
    try:
        with pytest.raises(PlayError, match='identity'):
            reopened.state(state['sessionId'])
        with pytest.raises(PlayError, match='another model'):
            reopened.create('identity-create-0001', {'playMode': 'free', 'visibility': 'developer'})
    finally:
        reopened.close()


def test_research_pin_rejects_wrong_bytes_and_fixture_loader(tmp_path, monkeypatch):
    from src.policies import hu100_research as pin
    _, fixture, path = model_fixture(tmp_path)
    with pytest.raises(ValueError): pin.load_policy(path)
    monkeypatch.setattr(pin, 'MODEL_BYTES', path.stat().st_size)
    monkeypatch.setattr(pin, 'MODEL_SHA256', fixture.description['weights_sha256'])
    monkeypatch.setattr(pin, 'CHECKPOINT_SHA256', fixture.description['source_checkpoint_sha256'])
    monkeypatch.setattr(pin, 'SEED', fixture.description['training_seed'])
    monkeypatch.setattr(pin, 'ITERATION', fixture.description['iteration'])
    monkeypatch.setattr(pin, 'ENTRIES', fixture.description['entries'])
    # Existing fixture uses traverser averaging; reject that lineage, too.
    with pytest.raises(ValueError, match='lineage'): pin.load_policy(path)


def test_legacy_hu20_in_progress_and_spectator_identity(tmp_path):
    from tests.play_ui.test_service import FixturePolicy
    from tests.play_ui.test_spectator import setup, create, deal, finish as spectator_finish
    service = PlayService(tmp_path / 'legacy.sqlite', FixturePolicy())
    try:
        state = service.create('legacy-create-key-01', {'playMode': 'free', 'visibility': 'developer'})
        state = service.new_hand(state['sessionId'], 'legacy-hand-key-001', {'revision': 0})
        state = service.act(state['sessionId'], 'legacy-call-key-001', {**request(state), 'kind': 'call', 'raiseTo': None})
        raw = service._load(state['sessionId'])
        for key in ('table', 'modelIdentity'): raw.pop(key)
        for key in ('table', 'game', 'modelIdentity', 'botDecisions', 'samplingStart'): raw['current'].pop(key)
        service.db.execute('UPDATE sessions SET state=? WHERE id=?', (json.dumps(raw), state['sessionId'])); service.db.commit()
        state = finish(service, state, 'legacy')
        assert service.verify_replay(state['sessionId']) == 1
    finally: service.close()
    tables = setup(tmp_path / 'tables')
    try:
        state = spectator_finish(tables, deal(tables, create(tables)))
        raw = tables.spectator._load(state['sessionId'])
        for model in raw['models']:
            for key in ('table', 'inference', 'research'): model.pop(key)
        for row in [*raw['history'], raw['current']]:
            row.pop('table', None); row.pop('game', None)
            row['models'] = raw['models']
            for record in row['decisions']:
                record['model'] = raw['models'][record['seat']]
        raw.pop('table')
        tables.spectator.db.execute('UPDATE sessions SET state=? WHERE id=?', (json.dumps(raw), state['sessionId'])); tables.spectator.db.commit()
        assert tables.spectator.verify_replay(state['sessionId']) == 1
    finally: tables.close()


def test_exact_100bb_all_in_showdown_and_restricted_rejection(tmp_path):
    from tests.play_ui.test_service import FixturePolicy
    from src.blueprint.abstraction import HU100_SCHEMA, choices
    class CallingPolicy(FixturePolicy):
        def __init__(self):
            super().__init__(); self.game = HU100_GAME; self.abstraction = HU100_SCHEMA
        def distribution(self, view):
            menu = choices(view, raise_cap=None, free_fold=False)
            selected = next(i for i,c in enumerate(menu) if c.action.kind in (ActionKind.CALL, ActionKind.CHECK))
            return menu, tuple(float(i == selected) for i in range(len(menu))), True
    policy = CallingPolicy(); service = PlayService(tmp_path / 'allin.sqlite', policy)
    try:
        state = service.create('allin-create-key001', {'playMode': 'free', 'visibility': 'developer'})
        state = service.new_hand(state['sessionId'], 'allin-hand-key0001', {'revision': 0})
        assert state['hand']['legal']['maxRaiseTo'] == 10000
        state = service.act(state['sessionId'], 'allin-raise-key001', {**request(state), 'kind': 'raise', 'raiseTo': 10000})
        state = service.advance(state['sessionId'], 'allin-call-key0001', request(state))
        assert state['phase'] == 'finished' and len(state['hand']['board']) == 5
        assert abs(state['hand']['result']['humanChips']) in (0,10000)
        assert sum(p['stack'] for p in state['hand']['players']) == 20000
        assert service.verify_replay(state['sessionId']) == 1
        restricted = service.create('restricted100-key01', {'playMode': 'restricted', 'visibility': 'developer'})
        restricted = service.new_hand(restricted['sessionId'], 'restricted100-hand01', {'revision': 0})
        with pytest.raises(PlayError, match='restricted menu'):
            service.act(restricted['sessionId'], 'restricted100-raise1', {**request(restricted), 'kind': 'raise', 'raiseTo': 550})
        assert service.state(restricted['sessionId']) == restricted
    finally: service.close()


def test_research_loader_matches_arena_reader_with_explicit_options(tmp_path, monkeypatch):
    import gzip
    from src.blueprint.average import AveragePolicy, EXTRACTIONS
    from src.blueprint.abstraction import HU100_SCHEMA
    from src.policies.files import file_hash
    from src.policies import hu100_research as pin
    _, _, path = model_fixture(tmp_path)
    with gzip.open(path, 'rt') as source:
        rows = [json.loads(line) for line in source]
    rows[0]['checkpoint_header']['average_rule'] = 'opponent-sampled'
    rows[0]['extraction'] = EXTRACTIONS['opponent-sampled']
    updated = tmp_path / 'opponent-average.gz'
    with gzip.open(updated, 'wt') as output:
        for row in rows: output.write(json.dumps(row)+'\n')
    arena = AveragePolicy(updated, file_hash(updated), expected_schema=HU100_SCHEMA)
    for name, value in {'MODEL_BYTES': updated.stat().st_size, 'MODEL_SHA256': file_hash(updated),
        'CHECKPOINT_SHA256': arena.description['source_checkpoint_sha256'],
        'SEED': arena.description['training_seed'], 'ITERATION': arena.description['iteration'],
        'ENTRIES': arena.description['entries']}.items(): monkeypatch.setattr(pin, name, value)
    web = pin.load_policy(updated)
    assert web.translation is None and web.spec.sha256 == arena.description['weights_sha256']
    from src.game.hand import Hand, Table
    view = Hand.start(Table(('a','b'), (10000,10000)), hand_id='fixture', seed=17).observe(0)
    assert web.distribution(view) == arena.distribution(view)
    translated = pin.load_policy(updated, translation=True)
    assert translated.description['action_translation']['version'] == VERSION
    with pytest.raises(ValueError): pin.load_policy(updated, translation='yes')


@pytest.mark.parametrize('arguments', [
    ['--translate-off-menu'], ['--stack-bb', '100'],
    ['--hu100-research', 'absent.gz', '--stack-bb', '20'],
])
def test_cli_rejects_incompatible_assertions_before_loading(monkeypatch, arguments):
    from src.play_api.server import main
    monkeypatch.setattr('sys.argv', ['play-api', *arguments])
    with pytest.raises(SystemExit) as stopped: main()
    assert stopped.value.code == 2
