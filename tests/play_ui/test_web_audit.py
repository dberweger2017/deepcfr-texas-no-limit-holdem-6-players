"""Recorded decisions detect adapter and settlement drift independently of the service."""

from copy import deepcopy

import pytest

from scripts.audit_v041_web import audit_states, read_states
from src.blueprint.average import AveragePolicy
from src.policies import v041 as candidate
from src.play_api.service import PlayService
from tests.play_ui.test_v041 import model
from tests.play_ui.test_service import create, human_action, bot_action


def test_recorded_both_seats_distributions_and_settlements(tmp_path, monkeypatch):
    path, _ = model(tmp_path, monkeypatch)
    web = candidate.load_policy(path)
    arena = AveragePolicy(path, candidate.MODEL_SHA256)
    database = tmp_path / 'play.sqlite'
    service = PlayService(database, web, source_version='recorded-fixture')
    try:
        state = create(service)
        for number in range(2):
            state = service.new_hand(state['sessionId'], f'new-hand-{number:020d}',
                                     {'revision': state['revision']})
            for step in range(100):
                if state['phase'] == 'finished':
                    break
                key = f'action-{number:03d}-{step:020d}'
                if state['hand']['actor'] == 1:
                    state = bot_action(service, state, key)
                else:
                    kind = 'check' if 'check' in state['hand']['legal']['kinds'] else 'call'
                    state = human_action(service, state, kind, key=key)
            assert state['phase'] == 'finished'
    finally:
        service.close()
    states = read_states(database)
    result = audit_states(states, web, arena, candidate.MODEL_SHA256)
    assert result['counts']['hands'] == 2
    assert result['counts']['human_button'] == result['counts']['human_big_blind'] == 1
    assert result['positions'] and result['unexpected_fallbacks'] == 0
    changed = deepcopy(states)
    changed[0]['history'][0]['humanChips'] += 1
    with pytest.raises(AssertionError, match='Settlement'):
        audit_states(changed, web, arena, candidate.MODEL_SHA256)
    original = web.distribution

    def wrong_probability(view):
        menu, probabilities, trained = original(view)
        return menu, tuple(value + .01 for value in probabilities), trained

    monkeypatch.setattr(web, 'distribution', wrong_probability)
    with pytest.raises(AssertionError, match='distribution mismatch'):
        audit_states(states, web, arena, candidate.MODEL_SHA256)
