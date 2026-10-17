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


@pytest.mark.parametrize('stack', [10000, 20000])
def test_k50_spectator_postflop_and_independent_audit_use_policy_tables(tmp_path, monkeypatch, stack):
    from random import Random
    from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
    from src.blueprint.solver import HU200_GAME
    from src.blueprint.action_translation import version_for_schema
    from src.blueprint import equity_buckets
    from src.blueprint.equity_buckets import EquityCards
    from src.game.hand import Hand, Table
    from src.game.types import Action, ActionKind
    from src.blueprint.abstraction import schema_choices, information_key
    from src.play_api.service import _model_info
    from src.play_api.spectator import SpectatorService
    from src.play_api.spectator_audit import audit_states
    from tests.native_equity_tables import write_tables

    seed = Random(1).randrange(2**63)
    hand = Hand.start(Table(('bot-a', 'bot-b'), (stack, stack)), hand_id='fixture', seed=seed)
    scenes = []; views = []
    while not hand.finished:
        view = hand.observe(hand.actor); views.append(view)
        for seat in (0, 1):
            own = hand.observe(seat)
            if own.board: scenes.append((own.hole_cards, own.board))
        hand = hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL))
    tables = tmp_path/'tables'; tables.mkdir(); write_tables(tables, scenes)
    cards = EquityCards(tables, pinned=False)
    monkeypatch.setattr(equity_buckets, 'EQUITY_K50_SHA256', cards.sha256)
    monkeypatch.setattr(equity_buckets, '_registered', object())
    monkeypatch.setattr('src.play_api.spectator.secrets.randbits', lambda _: 1)
    policy = fixture_policy(tmp_path, True)
    policy.abstraction = HU100_EQUITY_MENU_V2_SCHEMA if stack == 10000 else HU200_EQUITY_SCHEMA
    if stack == 20000: policy.game = HU200_GAME
    policy.description['abstraction'] = policy.abstraction
    policy.identity = {'card_tables': cards.sha256}; policy.equity_cards = cards
    policy.adapter_id = version_for_schema(policy.abstraction)
    policy.configure_translation(TranslationOptions())
    policy.entries = {}; policy.zero_mass = set(); policy.visits = {}
    for view in views:
        menu = schema_choices(view, policy.abstraction)
        key = information_key(view, menu, schema=policy.abstraction, equity_cards=cards)
        probabilities = tuple(float(item.action.kind in (ActionKind.CALL, ActionKind.CHECK)) for item in menu)
        assert sum(probabilities) == 1
        policy.entries[key] = (tuple(item.name for item in menu), probabilities)
        policy.visits[key] = 1
    version = 'v0.5.3' if stack == 10000 else 'v0.5.5'; identity = {'version': version, **_model_info(policy)}
    service = SpectatorService(tmp_path/'spectator.sqlite', {version: policy}, {version: identity})
    try:
        state = service.create('k50-spectator-create-0001', {'sessionType': 'spectator', 'modelVersions': [version, version]})
        state = service.new_hand(state['sessionId'], 'k50-spectator-hand-0001', {'revision': 0})
        streets = set()
        for number in range(20):
            if state['phase'] == 'finished': break
            streets.add(state['hand']['street'])
            state = service.advance(state['sessionId'], f'k50-spectator-step-{number:04}',
                {'revision': state['revision'], 'handId': state['hand']['id']})
        assert state['phase'] == 'finished' and state['handsPlayed'] == 1
        assert streets == {'preflop', 'flop', 'turn', 'river'}
        raw = service._load(state['sessionId'])
        checked = audit_states([raw], {version: policy}, {version: identity})
        assert checked['hands'] == 1 and service.verify_replay(state['sessionId']) == 1
        assert all(row['lookup'] == 'trained' for row in state['hand']['decisions'])
    finally:
        service.close()
