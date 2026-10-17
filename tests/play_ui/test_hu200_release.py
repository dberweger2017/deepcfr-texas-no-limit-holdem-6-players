"""HU200 table boundary, restricted/free wagers and durable identity checks."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.blueprint.solver import HU200_GAME
from src.play_api.configuration import PlayTable, spectator_identity_matches
from src.play_api.play_audit import audit_state
from src.play_api.service import PlayService, PlayError
from src.policies.hu200_bundle import INFERENCE
from tests.play_ui.test_hu100_research import fixture_policy, request


def policy_fixture(tmp_path):
    policy = fixture_policy(tmp_path, True)
    policy.game = HU200_GAME; policy.abstraction = HU200_EQUITY_SCHEMA
    policy.identity = {'card_tables': EQUITY_K50_SHA256}
    policy.equity_cards = SimpleNamespace(sha256=EQUITY_K50_SHA256)
    policy.adapter_id = INFERENCE['adapter']
    policy.configure_translation(TranslationOptions())
    return policy


@pytest.mark.parametrize('mode,amount', [('restricted', 300), ('free', 550), ('free', 20000)])
def test_wager_table_reset_and_pending_restart(tmp_path, mode, amount):
    policy = policy_fixture(tmp_path)
    db = tmp_path / 'play.sqlite'; service = PlayService(db, policy)
    state = service.create('hu200-create-session-001', {'playMode': mode, 'visibility': 'developer'})
    state = service.new_hand(state['sessionId'], 'hu200-start-hand-0001', {'revision': 0})
    assert state['table']['stack'] == 20000 and service.api_version == 'hu200-release-play-api-v1'
    assert state['model']['inference'] == INFERENCE
    assert {p['stack'] for p in state['hand']['players']} == {19900, 19950}
    if mode == 'restricted':
        with pytest.raises(PlayError, match='restricted menu'):
            service.act(state['sessionId'], 'hu200-refuse-offmenu-001', {**request(state), 'kind': 'raise', 'raiseTo': 550})
    body = {**request(state), 'kind': 'raise', 'raiseTo': amount}
    pending = service.act(state['sessionId'], 'hu200-wager-00000001', body)
    assert pending['hand']['actor'] == 1
    audit_state(service._load(state['sessionId']), policy)
    service.close()
    restarted = PlayService(db, policy)
    try:
        assert restarted.state(state['sessionId']) == pending
        assert restarted.act(state['sessionId'], 'hu200-wager-00000001', body) == pending
    finally: restarted.close()
    wrong = deepcopy(policy)
    wrong.description['action_translation']['max_states'] = 511
    with pytest.raises(ValueError, match='translation identity'): PlayService(db, wrong)
    for stack in (2000, 10000):
        with pytest.raises(ValueError): PlayService(tmp_path / f'wrong-{stack}.sqlite', policy, table=PlayTable(stack=stack))


def test_hu200_spectator_requires_complete_identity(tmp_path):
    from src.play_api.service import _model_info
    record = _model_info(policy_fixture(tmp_path))
    assert spectator_identity_matches(record, record)
    changed = dict(record); changed.pop('inference')
    assert not spectator_identity_matches(changed, record)
