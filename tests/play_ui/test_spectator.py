"""Real HU20 decisions, recovery, information isolation and independent tamper detection."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from random import Random

import pytest

from src.blueprint.abstraction import choices, information_key
from src.game.observation import Observation
from src.play_api.service import PlayError, PlayService, _model_info
from src.play_api.spectator import SpectatorService
from src.play_api.spectator_audit import audit_states
from src.play_api.versions import VersionedTables
from tests.play_ui.test_service import FixturePolicy


class ObservingPolicy(FixturePolicy):
    def __init__(self, digest, strategy='call'):
        super().__init__()
        self.spec = type('Identity', (), {'sha256': digest * 64})()
        self.strategy = strategy
        self.zero_mass = set()
        self.views = []

    def distribution(self, view):
        assert isinstance(view, Observation)
        assert not hasattr(view, 'deck') and not hasattr(view, 'seed') and not hasattr(view, 'apply')
        assert len(view.hole_cards) == 2
        assert not view.players[1 - view.seat].shown_cards
        with pytest.raises(FrozenInstanceError):
            view.board = ('As',)
        self.views.append(view)
        menu = choices(view, raise_cap=None, free_fold=False)
        key = information_key(view, menu, schema=self.abstraction)
        source = int(key[-2:], 16) % 3
        if source == 0:
            self.zero_mass.add(key)
        if source != 1:
            return menu, (1 / len(menu),) * len(menu), source == 0
        if self.strategy == 'allin':
            selected = max(range(len(menu)), key=lambda i: menu[i].action.raise_to or 0)
        elif self.strategy == 'fold':
            selected = next((i for i, item in enumerate(menu) if item.name == 'fold'), 0)
        else:
            selected = next(i for i, item in enumerate(menu) if item.action.kind.value in ('call', 'check'))
        return menu, tuple(float(i == selected) for i in range(len(menu))), True


def setup(tmp_path, strategies=('call', 'allin')):
    policies = {version: ObservingPolicy(sha, strategy)
                for version, sha, strategy in zip(('v0.4.1', 'v0.4.0'), ('a', 'b'), strategies)}
    identities = {v: {'version': v, **_model_info(p), 'manifestSha256': str(i) * 64,
                      'manifestUrl': f'https://example.test/{v}/release-manifest.json'}
                  for i, (v, p) in enumerate(policies.items())}
    spectator = SpectatorService(tmp_path / 'spectator/private.sqlite', policies, identities,
                                  source_version='test')
    services = {v: PlayService(tmp_path / v / 'private.sqlite', p) for v, p in policies.items()}
    return VersionedTables(services, spectator=spectator, identities=identities)


def create(tables, versions=None, key='spectator-create-key-001'):
    return tables.create(key, {'sessionType': 'spectator',
                               'modelVersions': versions or ['v0.4.1', 'v0.4.0']})


def deal(tables, state):
    return tables.new_hand(state['sessionId'], f"deal-key-{state['revision']:020d}",
                           {'revision': state['revision']})


def step(tables, state, key=None):
    return tables.advance(state['sessionId'], key or f"step-key-{state['revision']:020d}",
                          {'revision': state['revision'], 'handId': state['hand']['id']})


def finish(tables, state):
    for _ in range(100):
        if state['phase'] == 'finished':
            return state
        prior = len(state['hand']['decisions'])
        state = step(tables, state)
        assert len(state['hand']['decisions']) == prior + 1
    pytest.fail('Hand did not settle')


def private(tables, state):
    return tables.spectator._load(state['sessionId'])


def audit(tables, states):
    return audit_states(states, tables.spectator.policies, tables.spectator.identities)


def test_two_models_both_seats_exact_observations_and_lookup_status(tmp_path):
    tables = setup(tmp_path)
    try:
        state = create(tables)
        for _ in range(20):
            state = finish(tables, deal(tables, state))
        result = audit(tables, [private(tables, state)])
        assert result['hands'] == 20
        assert all(result['counts'][status] > 0 for status in ('missing', 'trained', 'zero-mass'))
        assert result['counts']['button_0_hands'] == result['counts']['button_1_hands'] == 10
        assert tables.verify_replay(state['sessionId']) == 20
        assert sum(state['sessionChips']) == 0
        history = tables.history(state['sessionId'])['hands']
        assert len(history) == 20 and history[0]['models'] == state['models']
        assert history[-1]['decisions'] == state['hand']['decisions']
        for decision in state['hand']['decisions']:
            snapshot = decision['observation']
            assert snapshot['seat'] == decision['seat']
            assert len(snapshot['hole_cards']) == 2
            assert snapshot['actor'] == decision['seat']
            assert decision['model'] == state['models'][decision['seat']]
            assert decision['menu'][decision['selectedIndex']]['probability'] > 0
        # Neither projection publishes private reproduction state.
        public = json.dumps(state) + json.dumps(tables.history(state['sessionId']))
        assert all(field not in public for field in ('dealSeed', 'dealRng', 'botRngs', 'samplingStart'))
    finally:
        tables.close()


def test_step_retry_conflict_restart_and_same_model(tmp_path):
    tables = setup(tmp_path)
    state = create(tables, ['v0.4.0', 'v0.4.0'])
    state = deal(tables, state)
    before = deepcopy(state)
    key = 'retried-decision-key-0001'
    state = step(tables, before, key)
    assert step(tables, before, key) == state
    assert state['revision'] == before['revision'] + 1
    assert len(state['hand']['decisions']) == 1
    with pytest.raises(PlayError, match='Stale'):
        step(tables, before, 'different-decision-key-001')
    with pytest.raises(PlayError, match='another model'):
        tables.create(key, {'playMode': 'restricted', 'visibility': 'developer'})
    with pytest.raises(PlayError, match='only bot'):
        tables.act(state['sessionId'], 'human-action-key-0001', {})
    assert audit(tables, [private(tables, state)])['hands'] == 0
    tables.close()
    reopened = setup(tmp_path)
    try:
        assert reopened.state(state['sessionId']) == state
        assert step(reopened, before, key) == state
        assert audit(reopened, [private(reopened, state)])['decisions'] == 1
        state = finish(reopened, state)
        assert reopened.verify_replay(state['sessionId']) == 1
        assert state['models'][0] == state['models'][1]
        next_hand = deal(reopened, state)
        assert next_hand['hand']['button'] == 1
        assert sorted(p['stack'] for p in next_hand['hand']['perspectives'][0]['players']) == [1900, 1950]
    finally:
        reopened.close()


@pytest.mark.parametrize('field', ['probability', 'cards', 'bounds', 'selected', 'status', 'model',
                                  'manifest', 'settlement', 'rotation', 'sampling', 'history'])
def test_independent_audit_rejects_tampered_evidence(tmp_path, field):
    tables = setup(tmp_path)
    try:
        state = finish(tables, deal(tables, create(tables)))
        evidence = private(tables, state)
        row = evidence['history'][0]
        record = row['decisions'][0]
        if field == 'probability': record['menu'][0]['probability'] += .01
        elif field == 'cards': record['observation']['hole_cards'] = ['As', 'As']
        elif field == 'bounds': record['observation']['legal_actions']['call_amount'] += 1
        elif field == 'selected': record['selectedIndex'] = -1
        elif field == 'status': record['lookup'] = 'not-a-status'
        elif field == 'model': record['model']['sha256'] = 'e' * 64
        elif field == 'manifest': evidence['models'][0]['manifestSha256'] = 'e' * 64
        elif field == 'settlement': row['netChips'][0] += 1
        elif field == 'rotation': row['button'] = 1
        elif field == 'sampling': evidence['botRngs'][0] = json.loads(json.dumps(Random(2).getstate()))
        elif field == 'history': evidence['history'] = []
        evidence['current'] = deepcopy(row)
        with pytest.raises(AssertionError):
            audit(tables, [evidence])
    finally:
        tables.close()


def test_invalid_models_and_extra_fields_fail_without_creating_session(tmp_path):
    tables = setup(tmp_path)
    try:
        for versions in (['v0.4.0'], ['unknown', 'v0.4.1'], [[], 'v0.4.1'], 'v0.4.0'):
            with pytest.raises(PlayError, match='two available'):
                create(tables, versions)
        with pytest.raises(PlayError, match='two available'):
            tables.create('invalid-model-key-001', {'sessionType': 'spectator',
                          'modelVersions': ['v0.4.0', 'v0.4.1'], 'dealSeed': 1})
        assert tables.spectator.db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0] == 0
        state = create(tables)
        assert tables.create('spectator-create-key-001', {'sessionType': 'spectator',
                              'modelVersions': ['v0.4.1', 'v0.4.0']}) == state
    finally:
        tables.close()
