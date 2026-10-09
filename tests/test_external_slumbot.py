"""Official protocol fixtures; no live endpoint or external poker campaign."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from src.arena.external.interface import GameContract
from src.arena.external.slumbot import (CONTRACT, SlumbotAdapter, SlumbotConnection,
    public_response, replay_record, verify_record)
from src.blueprint.abstraction import HU100_SCHEMA
from src.blueprint.solver import HU100_GAME
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


def response(action='b200', board=(), seat=0):
    return {'old_action': '', 'action': action, 'client_pos': seat,
            'hole_cards': ['Ac', '9d'], 'board': list(board), 'token': 'secret-token'}


def test_official_open_raise_position_amount_and_filter():
    raw = response()
    raw.update(opponent_cards=['As','Ad'], deck=['Ks'], seed=9, evaluator={'future': 99})
    view = SlumbotAdapter().observation(raw, 'fixture')
    assert view.seat == 0 and view.button == 1
    assert view.hole_cards == ('Ac','9d') and view.board == ()
    assert [p.stack for p in view.players] == [19900, 19800]
    assert view.legal_actions.call_amount == 100
    assert view.legal_actions.min_raise_to == 300
    assert view.legal_actions.max_raise_to == 20000
    assert not view.players[1].shown_cards
    assert not any(k in json.dumps(asdict(view)) for k in ('secret-token','opponent_cards','deck','evaluator','seed'))
    assert SlumbotAdapter().encode_action(Action(ActionKind.RAISE, 500), view) == 'b500'
    assert SlumbotAdapter().encode_action(Action(ActionKind.CALL), view) == 'c'
    with pytest.raises(ValueError): SlumbotAdapter().encode_action(Action(ActionKind.RAISE, 250), view)


def test_street_local_bets_do_not_become_total_commitment():
    raw = response('b200c/kb400', ('2c','3d','4h'))
    view = SlumbotAdapter().observation(raw, 'fixture')
    assert view.pot == 800 and view.players[1].contributed == 600
    assert view.players[1].street_bet == 400
    assert SlumbotAdapter().encode_action(Action(ActionKind.RAISE, 800), view) == 'b800'
    record = json.loads(json.dumps(replay_record(raw, 'fixture', Action(ActionKind.CALL))))
    assert verify_record(record) == view
    assert 'token' not in json.dumps(record)
    for key, value in [('contract', {}), ('observation', {}), ('selected', 'b700')]:
        changed = deepcopy(record); changed[key] = value
        with pytest.raises(ValueError): verify_record(changed)


@pytest.mark.parametrize('action,board,seat', [
    ('b200c/kk/kk/kb200', ('2c','3d','4h','5s','6c'), 0),
    ('c', (), 0), ('b200c', ('2c','3d','4h'), 0),
    ('ck', ('2c','3d','4h'), 0), ('b200c/k', ('2c','3d','4h'), 1),
    ('b200c/kb400c/kb800', ('2c','3d','4h','5s'), 0),
])
def test_public_prefix_matches_independent_native_engine(action, board, seat):
    # Engine needs a complete deck, but it remains strictly within this test host.
    # Production decoding uses public replay and never invents an opponent hand.
    raw = response(action, board, seat)
    expected = SlumbotAdapter().observation(raw, 'fixture')
    from src.arena.external.slumbot import TOKEN
    hand = Hand.start(Table(('slumbot-seat-0','slumbot-seat-1'), (20000,20000), button=1),
                      hand_id='fixture', seed=1)
    for token in TOKEN.findall(action):
        if token == '/': continue
        chosen = Action(ActionKind.RAISE, int(token[1:])) if token.startswith('b') else Action(
            {'k': ActionKind.CHECK, 'c': ActionKind.CALL, 'f': ActionKind.FOLD}[token])
        hand = hand.apply(chosen)
    actual = hand.observe(seat)
    assert expected.actor == actual.actor and expected.street == actual.street
    assert expected.players == actual.players
    assert expected.legal_actions == actual.legal_actions
    assert expected.pot == actual.pot


@pytest.mark.parametrize('raw', [response('k'), response('b199'), response('b20001'),
    response('b200c//'), response('b200c/k', (), 1), response('b200', (), 1),
    response('b200c/kk/kk/kk', ('2c','3d','4h','5s','6c')),
    response('b20000c///', ('2c','3d','4h','5s','6c')),
    {**response(), 'client_pos': True}, {**response(), 'hole_cards': ['Ac','Ac']},
    {**response(), 'winnings': 100}, {**response(), 'error': 'bad'},
    response('b200<script>'), response('b0200'), response('b200c/kkkk', ('2c','3d','4h'))])
def test_invalid_terminal_or_nonclient_input_is_rejected(raw):
    with pytest.raises(ValueError): SlumbotAdapter().observation(raw, 'fixture')


def test_fixed_hu100_is_rejected_before_transport_or_stack_scaling():
    class Transport:
        def post(self, endpoint, body): pytest.fail('Must reject before network access')
    policy = SimpleNamespace(game=HU100_GAME, abstraction=HU100_SCHEMA, players=2, raise_cap=None)
    with pytest.raises(ValueError, match='incompatible'): SlumbotConnection(Transport(), policy)
    GameContract(stack=10000).admit(policy)
    for contract in [replace(CONTRACT, stack=10000, rake=1), replace(CONTRACT, stack=10000, ante=1),
                     replace(CONTRACT, stack=10000, reset_each_hand=False), replace(CONTRACT, stack=10000, big_blind=50)]:
        with pytest.raises(ValueError): contract.admit(policy)


def test_transport_rotation_and_no_automatic_retries(monkeypatch):
    # Isolate wire behavior from the admission guard, covered independently above.
    monkeypatch.setattr(GameContract, 'admit', lambda self, policy: None)
    calls = []
    class Transport:
        def post(self, endpoint, body):
            calls.append((endpoint, body.copy()))
            if len(calls) == 3: raise TimeoutError('Ambiguous remote mutation')
            return {**response(), 'token': f'rotated-{len(calls)}'}
    connection = SlumbotConnection(Transport(), None)
    connection.request('new_hand'); connection.request('act', 'c')
    assert calls == [('/slumbot/api/new_hand', {}),
                     ('/slumbot/api/act', {'token': 'rotated-1', 'incr': 'c'})]
    with pytest.raises(TimeoutError): connection.request('act', 'k')
    assert len(calls) == 3 and connection.token == 'rotated-2'


def test_append_only_journal_replays_prefixes_and_preserves_failure(tmp_path):
    from src.arena.external.journal import PrefixJournal, verify_journal
    path = tmp_path / 'prefixes.jsonl'
    journal = PrefixJournal(path, {'source_sha256': 'a'*64, 'policy_sha256': 'b'*64,
                                 'translation': None, 'opponent': 'slumbot-public-200bb'})
    journal.append(json.loads(json.dumps(replay_record(response(), 'fixture', Action(ActionKind.CALL)))))
    journal.failure('timeout-ambiguous'); journal.close()
    assert verify_journal(path)['decisions'] == 1
    assert 'secret-token' not in path.read_text()
    with pytest.raises(FileExistsError): PrefixJournal(path, {})
    lines = path.read_text().splitlines(); path.write_text('\n'.join(lines[::-1])+'\n')
    with pytest.raises(ValueError): verify_journal(path)


def test_official_error_msg_and_optional_token_updates(monkeypatch):
    with pytest.raises(ValueError): public_response({**response(), 'error_msg': ''})
    monkeypatch.setattr(GameContract, 'admit', lambda self, policy: None)
    calls = []
    class Transport:
        def post(self, endpoint, body):
            calls.append(body)
            if len(calls) == 1: return response()
            if len(calls) == 2: return {k:v for k,v in response().items() if k != 'token'}
            return {**response(), 'error_msg': 'invalid'}
    connection = SlumbotConnection(Transport(), None)
    connection.request('new_hand'); connection.request('act', 'c')
    assert connection.token == 'secret-token'
    with pytest.raises(ValueError): connection.request('act', 'c')
    with pytest.raises(ValueError, match='stopped'): connection.request('act', 'c')
    assert len(calls) == 3


def test_journal_tail_anchor_detects_truncation(tmp_path):
    from src.arena.external.journal import PrefixJournal, verify_journal
    path = tmp_path / 'records.jsonl'
    journal = PrefixJournal(path, {'opponent':'slumbot'})
    journal.failure('timeout-ambiguous'); journal.close()
    tail = verify_journal(path)['tail_sha256']
    path.write_text(path.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError): verify_journal(path, expected_tail=tail)
    path.write_text('')
    with pytest.raises(ValueError): verify_journal(path)
