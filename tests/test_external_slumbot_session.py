"""Complete Slumbot hands: settlement, the hand runner, the journal and the HTTP transport.

`live-hands-20261013.json` holds 12 hands played against the live API on October 13, 2026 by a
probe bot (tokens removed): each item is a hand's responses, with the increment the client sent
before each one after the first.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from src.arena.external.http import AmbiguousRequest, HttpTransport, ServiceError
from src.arena.external.journal import PrefixJournal, verify_journal
from src.arena.external.session import CheckCall, ending, play_hand, summarize
from src.arena.external.slumbot import CONTRACT, SlumbotConnection, settle, terminal_record, verify_terminal
from src.blueprint.abstraction import HU100_SCHEMA, HU200_SCHEMA
from src.blueprint.solver import HU100_GAME, HU200_GAME
from src.game.types import Action, ActionKind

LIVE = json.loads((Path(__file__).parent / 'fixtures/slumbot/live-hands-20261013.json').read_text())
HU200 = SimpleNamespace(game=HU200_GAME, abstraction=HU200_SCHEMA, players=2, raise_cap=None)


def decode(increment):
    if increment.startswith('b'):
        return Action(ActionKind.RAISE, int(increment[1:]))
    return Action({'k': ActionKind.CHECK, 'c': ActionKind.CALL, 'f': ActionKind.FOLD}[increment])


class Replay:
    """Serves recorded hands as a transport, and plays the recorded client actions."""
    def __init__(self, hands):
        self.responses = [dict(r, token='session-token-7f3a') for hand in hands for r in hand]
        self.sent = [r['sent'] for hand in hands for r in hand[1:]]
        self.bodies = []

    def post(self, endpoint, body):
        self.bodies.append((endpoint, body))
        response = self.responses.pop(0)
        return {k: v for k, v in response.items() if k != 'sent'}

    def choose_action(self, view):
        return decode(self.sent.pop(0))


def test_every_live_hand_settles_to_the_reported_winnings():
    kinds = set()
    for hand in LIVE:
        record = terminal_record(hand[-1], 'live')
        assert verify_terminal(record) == hand[-1]['winnings']
        kinds.add(ending(record))
    assert kinds == {'client_fold', 'slumbot_fold', 'showdown'}
    allin = next(h[-1] for h in LIVE if h[-1]['action'] == 'ck/kk/b100b19900c/')
    assert terminal_record(allin, 'live')['winnings'] == -20000


def test_settlement_rejects_changed_winnings_cards_and_impossible_actions():
    terminal = LIVE[0][-1]
    record = terminal_record(terminal, 'live')
    for key, value in [('winnings', record['winnings'] + 100), ('board', record['board'][:4]),
                       ('bot_hole_cards', record['hole_cards'])]:
        with pytest.raises(ValueError):
            verify_terminal({**record, key: value})
    for bad in [{**terminal, 'bot_hole_cards': ['Zz', '2c']}, {**terminal, 'winnings': 1.5},
                {**terminal, 'action': terminal['action'] + 'x'}]:
        with pytest.raises(ValueError):
            terminal_record(bad, 'live')
    with pytest.raises(ValueError):
        settle('b200fk', 0, ['As', 'Ad'], ['2c', '3d'], [])
    with pytest.raises(ValueError):
        settle('b200b30000c///', 0, ['As', 'Ad'], ['2c', '3d'], ['4h', '5h', '6h', '7h', '9c'])


def test_ties_split_and_uncalled_raises_are_returned():
    board = ['As', 'Ks', 'Qs', 'Js', 'Ts']   # a royal flush on board: every hand ties
    assert settle('b200c/kk/kk/kk', 0, ['2c', '3d'], ['4h', '5c'], board) == 0
    # The client (button) raises to 600 and the big blind folds: the client wins its 100.
    assert settle('b600f', 1, ['2c', '3d'], ['4h', '5c'], []) == 100
    # The big blind folds to a flop bet after calling 200: it loses 200, not the bet.
    assert settle('b200c/kb400f', 0, ['2c', '3d'], ['4h', '5c'], board[:3]) == -200


def test_runner_replays_live_hands_into_a_verified_journal(tmp_path):
    replay = Replay(LIVE)
    connection = SlumbotConnection(replay, HU200)
    journal = PrefixJournal(tmp_path / 'journal.jsonl', {'fixture': 'live-hands-20261013'})
    hands = [play_hand(connection, replay, f'h{i}', journal) for i in range(len(LIVE))]
    journal.close()
    checked = verify_journal(tmp_path / 'journal.jsonl')
    assert checked['hands'] == 12 and checked['decisions'] == sum(len(h) - 1 for h in LIVE)
    assert not replay.responses and not replay.sent
    assert all(body.get('token') == 'session-token-7f3a' for endpoint, body in replay.bodies[1:])
    assert 'session-token-7f3a' not in (tmp_path / 'journal.jsonl').read_text()
    summary = summarize(hands)
    assert summary['net_chips'] == sum(h[-1]['winnings'] for h in LIVE)
    assert summary['bb_per_100']['mean'] == pytest.approx(summary['net_chips'] / 12)
    assert summary['baseline_adjusted_bb_per_100']['n'] == 12
    assert sum(summary['endings'].values()) == 12


def test_runner_rejects_a_terminal_that_does_not_continue_the_decisions(tmp_path):
    hand = [dict(r) for r in LIVE[0]]
    hand[-1]['action'] = hand[-1]['action'].replace('b100c/b100c', 'b100c/kk')
    hand[-1]['winnings'] = -300
    replay = Replay([hand])
    journal = PrefixJournal(tmp_path / 'journal.jsonl', {})
    with pytest.raises(ValueError, match='continue'):
        play_hand(SlumbotConnection(replay, HU200), replay, 'h', journal)


def test_scripted_players_are_admitted_and_other_depths_are_not():
    CONTRACT.admit(CheckCall())
    CONTRACT.admit(HU200)
    with pytest.raises(ValueError):
        CONTRACT.admit(SimpleNamespace(game=HU100_GAME, abstraction=HU100_SCHEMA, players=2, raise_cap=None))


def test_login_sends_credentials_only_to_the_transport_and_fails_closed():
    class Login:
        def __init__(self, response): self.response, self.bodies = response, []
        def post(self, endpoint, body):
            self.bodies.append((endpoint, body)); return self.response
    transport = Login({'token': 'account-token'})
    connection = SlumbotConnection(transport, HU200)
    connection.login('name', 'secret')
    assert connection.token == 'account-token' and transport.bodies == [('/slumbot/api/login', {'username': 'name', 'password': 'secret'})]
    with pytest.raises(ValueError, match='once'):
        connection.login('name', 'secret')
    failed = SlumbotConnection(Login({'error_msg': 'bad password'}), HU200)
    with pytest.raises(ValueError) as error:
        failed.login('name', 'secret')
    assert 'secret' not in str(error.value) and failed.failed


@pytest.fixture
def server():
    """A local stand-in for the service: JSON, an error status, a stall, or a non-JSON body."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/stall':
                time.sleep(1.0)
            code, payload = {'/error': (500, b'{}'), '/text': (200, b'not json')}.get(self.path, (200, json.dumps({'echo': body}).encode()))
            self.send_response(code); self.end_headers(); self.wfile.write(payload)
        def log_message(self, *args): pass
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
    yield f'http://127.0.0.1:{httpd.server_port}'
    httpd.shutdown()


def test_http_transport_bounds_requests_and_never_retries(server):
    transport = HttpTransport(server, timeout=0.3, min_interval=0.05, max_requests=5)
    assert transport.post('/ok', {'a': 1}) == {'echo': {'a': 1}}
    with pytest.raises(ServiceError, match='HTTP 500'):
        transport.post('/error', {'token': 'secret'})
    with pytest.raises(ServiceError, match='not JSON'):
        transport.post('/text', {})
    with pytest.raises(AmbiguousRequest) as error:
        transport.post('/stall', {'token': 'secret'})
    assert 'secret' not in str(error.value)
    assert transport.requests == 4
    transport.post('/ok', {})
    with pytest.raises(ServiceError, match='cap'):
        transport.post('/ok', {})
    assert transport.requests == 5
