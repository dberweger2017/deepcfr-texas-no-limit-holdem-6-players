"""Restricted human sizing follows the loaded model, including independent replay."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.blueprint.abstraction import HU100_EQUITY_MENU_V2_SCHEMA
from src.blueprint.action_translation import TranslationOptions, HU100_K50_MENU_V2_VERSION
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.play_api.play_audit import audit_state
from src.play_api.service import PlayError, PlayService
from tests.play_ui.test_hu100_research import fixture_policy, request


@pytest.mark.parametrize('menu_v2', [False, True])
def test_restricted_render_action_and_replay_use_schema_owned_sizes(tmp_path, menu_v2):
    policy = fixture_policy(tmp_path, True)
    if menu_v2:
        policy.abstraction = HU100_EQUITY_MENU_V2_SCHEMA
        policy.description['abstraction'] = policy.abstraction
        policy.identity = {'card_tables': EQUITY_K50_SHA256}
        policy.equity_cards = SimpleNamespace(sha256=EQUITY_K50_SHA256)
        policy.adapter_id = HU100_K50_MENU_V2_VERSION
        policy.configure_translation(TranslationOptions())
    service = PlayService(tmp_path/'restricted.sqlite', policy)
    try:
        state = service.create('restricted-create-0001', {'playMode': 'restricted', 'visibility': 'developer'})
        state = service.new_hand(state['sessionId'], 'restricted-hand-0001', {'revision': 0})
        targets = {item['raiseTo'] for item in state['hand']['menu'] if item['kind'] == 'raise'}
        assert (400 in targets) is menu_v2
        assert (500 in targets) is menu_v2
        assert 800 not in targets
        for amount in ([800] if menu_v2 else [400, 800]):
            with pytest.raises(PlayError, match='outside the restricted menu'):
                service.act(state['sessionId'], f'restricted-reject-{amount}',
                    {**request(state), 'kind': 'raise', 'raiseTo': amount})
        state = service.act(state['sessionId'], 'restricted-accept-0001',
            {**request(state), 'kind': 'raise', 'raiseTo': 400 if menu_v2 else 300})
        raw = service._load(state['sessionId'])
        audit_state(raw, policy)
        changed = deepcopy(raw)
        changed['current']['actions'][0]['raiseTo'] = 800
        with pytest.raises(AssertionError, match='Restricted human menu'):
            audit_state(changed, policy)
    finally:
        service.close()
