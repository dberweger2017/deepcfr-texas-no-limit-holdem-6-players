"""Complete Slumbot hands: settlement, the hand runner, the journal and the HTTP transport.

`live-hands-20261013.json` holds 12 hands played against the live API on October 13, 2026 by a
probe bot (tokens removed): each item is a hand's responses, with the increment the client sent
before each one after the first.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import gzip
import statistics
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from src.arena.external.http import AmbiguousRequest, HttpTransport, ServiceError
from src.arena.external.journal import PrefixJournal, verify_journal
from src.arena.external.session import CheckCall, ending, hu200_average, play_hand, summarize, without_decisions
from src.arena.external.slumbot import CONTRACT, SlumbotConnection, settle, terminal_record, verify_terminal
from src.blueprint.abstraction import (HU100_EQUITY_SCHEMA, HU100_SCHEMA, HU200_EQUITY_SCHEMA,
    HU200_SCHEMA, HU20_EQUITY_SCHEMA, HU200_EQUITY_MENU_V2_SCHEMA, HU200_MENU_V2_SCHEMA)
from src.blueprint.solver import HU20_UNCAPPED_GAME
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
    assert summary['baseline_difference_bb_per_100']['n'] == 12
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


def test_runs_keep_only_terminals_and_decision_counts():
    from src.arena.external.session import Hand
    hand = Hand('h', [{'observation': 'large'}, {'observation': 'large'}], {'winnings': 100, 'client_pos': 0,
                'baseline_winnings': 0, 'action': 'b200f'})
    slim = without_decisions(hand)
    assert slim.terminal is hand.terminal and slim.decisions == [None, None]
    assert summarize([slim]) == summarize([hand])


def test_k50_hu200_policies_are_admitted_and_k50_at_other_depths_are_not():
    for schema in (HU200_EQUITY_SCHEMA, HU200_EQUITY_MENU_V2_SCHEMA, HU200_MENU_V2_SCHEMA):
        CONTRACT.admit(SimpleNamespace(game=HU200_GAME, abstraction=schema, players=2, raise_cap=None))
    for game, schema in ((HU100_GAME, HU100_EQUITY_SCHEMA), (HU20_UNCAPPED_GAME, HU20_EQUITY_SCHEMA),
                         (HU200_GAME, HU100_EQUITY_SCHEMA), (HU200_GAME, HU100_SCHEMA)):
        with pytest.raises(ValueError):
            CONTRACT.admit(SimpleNamespace(game=game, abstraction=schema, players=2, raise_cap=None))


def test_hu200_average_reads_k50_only_with_its_tables(monkeypatch, tmp_path):
    calls = []
    import src.blueprint.average as average
    monkeypatch.setattr(average, 'AveragePolicy', lambda *a, **k: calls.append((a, k)))
    path = tmp_path/'m.gz'
    def write(schema):
        with gzip.open(path,'wt') as stream:
            stream.write(json.dumps({'checkpoint_header':{'abstraction':schema}})+'\n')
    write(HU200_SCHEMA)
    hu200_average(path, 'sha')
    write(HU200_EQUITY_SCHEMA)
    hu200_average(path, 'sha', Path('tables'))
    assert calls == [((path, 'sha'), {'expected_schema': HU200_SCHEMA, 'sorted_rows': True, 'card_buckets': None,
                                        'cache': None, 'missing_key_fallback': 'uniform', 'cache_read_only': False, 'translation': None}),
                     ((path, 'sha'), {'expected_schema': HU200_EQUITY_SCHEMA, 'sorted_rows': True,
                                        'card_buckets': Path('tables'), 'cache': None, 'missing_key_fallback': 'uniform', 'cache_read_only': False, 'translation': None})]
    write(HU200_EQUITY_MENU_V2_SCHEMA)
    hu200_average(path,'sha',Path('tables'))
    assert calls[-1][1]['expected_schema']==HU200_EQUITY_MENU_V2_SCHEMA
    with pytest.raises(ValueError,match='schema'):
        hu200_average(path,'sha')
    write(HU100_EQUITY_SCHEMA)
    with pytest.raises(ValueError,match='schema'):
        hu200_average(path,'sha',Path('tables'))


def test_real_menu_v2_hu200_load_cache_and_analysis(tmp_path, monkeypatch):
    from src.blueprint import equity_buckets
    from src.blueprint.equity_buckets import EquityCards
    from src.arena.external.analysis import analyze
    from tests.native_equity_tables import write_tables
    from tests.test_hu200_evaluation import fixture_spec
    from scripts.play_slumbot import load_player
    from scripts import analyze_slumbot_run
    # These are legal public scenes from a recorded, non-all-in hand. Analysis
    # must query the model's v2 menu/key under its own K50 identity.
    scenes=[(r['hole_cards'],r['board']) for r in LIVE[0] if len(r['board'])>=3]
    tables=tmp_path/'tables'; tables.mkdir()
    write_tables(tables,scenes)
    cards=EquityCards(tables,pinned=False)
    cards.directory=tables
    monkeypatch.setattr(equity_buckets,'EQUITY_K50_SHA256',cards.sha256)
    spec=fixture_spec(tmp_path/'model',HU200_EQUITY_MENU_V2_SCHEMA,cards=cards)
    cache=tmp_path/'cache'
    model=hu200_average(spec['path'],spec['sha256'],tables,cache)
    CONTRACT.admit(model)
    warmed=hu200_average(spec['path'],spec['sha256'],tables,cache,cache_read_only=True)
    assert warmed.abstraction==model.abstraction==HU200_EQUITY_MENU_V2_SCHEMA
    client_model, _, identity=load_player(SimpleNamespace(model=Path(spec['path']),
        model_sha256=spec['sha256'],card_buckets=tables,policy_cache=cache,
        missing_key_fallback='uniform',cache_read_only=True,seed=271))
    assert client_model.abstraction==identity['abstraction']==HU200_EQUITY_MENU_V2_SCHEMA
    replay=Replay([LIVE[0]])
    journal=PrefixJournal(tmp_path/'journal.jsonl',{'fixture':True})
    play_hand(SlumbotConnection(replay,model),replay,'h0',journal)
    journal.close()
    result=analyze(tmp_path/'journal.jsonl',model=warmed,workers=1)
    assert result['hands']==1
    assert sum(sum(row.values()) for row in result['coverage_by_street'].values())==result['decisions']
    import sys
    monkeypatch.setattr(sys,'argv',['analyze_slumbot_run',str(tmp_path),'--model',spec['path'],
        '--model-sha256',spec['sha256'],'--card-buckets',str(tables),'--policy-cache',str(cache),
        '--cache-read-only','--workers','1'])
    analyze_slumbot_run.main()
    analyzed=json.loads((tmp_path/'analysis.json').read_text())
    assert analyzed['coverage_by_street']==result['coverage_by_street']
    with pytest.raises(ValueError,match='hash'):
        hu200_average(spec['path'],'f'*64,tables,cache)


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
    assert 'bad password' in str(error.value) and 'secret' not in str(error.value) and failed.failed
    echoed = SlumbotConnection(Login({'error_msg': 'no user with password secret'}), HU200)
    with pytest.raises(ValueError) as error:
        echoed.login('name', 'secret')
    assert 'secret' not in str(error.value)


@pytest.fixture
def server():
    """A local stand-in for the service: JSON, an error status, a stall, or a non-JSON body."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/stall':
                time.sleep(1.0)
            code, payload = {'/error': (500, b'{}'), '/text': (200, b'not json'),
                             '/refused': (400, b'{"error_msg": "No hand for token 0123456789abcdef0123"}')
                             }.get(self.path, (200, json.dumps({'echo': body}).encode()))
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
    with pytest.raises(ServiceError) as refused:
        transport.post('/refused', {'token': 'secret'})
    assert refused.value.status == 400 and str(refused.value) == '/refused returned HTTP 400: No hand for token ...'
    with pytest.raises(AmbiguousRequest) as error:
        transport.post('/stall', {'token': 'secret'})
    assert 'secret' not in str(error.value)
    assert transport.requests == 5
    transport = HttpTransport(server, timeout=0.3, min_interval=0.05, max_requests=1)
    transport.post('/ok', {})
    with pytest.raises(ServiceError, match='cap'):
        transport.post('/ok', {})
    assert transport.requests == 1


def test_a_rejected_request_abandons_only_its_hand_but_an_unreadable_answer_does_not():
    class Refusing:
        def __init__(self, error): self.error, self.calls = error, 0
        def post(self, endpoint, body):
            self.calls += 1
            if self.calls == 2:
                raise self.error
            return dict(LIVE[1][0], token='session-token-7f3a')
    connection = SlumbotConnection(Refusing(ServiceError('act returned HTTP 400', status=400)), HU200)
    connection.request('new_hand')
    with pytest.raises(ServiceError):
        connection.request('act', 'c')
    connection.abandon_hand()
    assert connection.request('new_hand')['token'] == 'session-token-7f3a'
    with pytest.raises(ValueError, match='rejected'):
        connection.new_session()
    refused = Refusing(ServiceError('act returned HTTP 400', status=400))
    connection = SlumbotConnection(refused, HU200)
    connection.request('new_hand')
    with pytest.raises(ServiceError):
        connection.request('act', 'c')
    connection.new_session()
    assert connection.token is None and connection.request('new_hand')['token'] == 'session-token-7f3a'
    unreadable = SlumbotConnection(Refusing(ServiceError('act returned a body that is not JSON')), HU200)
    unreadable.request('new_hand')
    with pytest.raises(ServiceError):
        unreadable.request('act', 'c')
    with pytest.raises(ValueError, match='abandoned'):
        unreadable.abandon_hand()
    with pytest.raises(ValueError, match='rejected'):
        unreadable.new_session()


def test_a_new_session_logs_in_again_and_session_totals_restart():
    from src.arena.external.session import Hand, session_accounting

    class Account:
        def __init__(self): self.bodies, self.calls = [], 0
        def post(self, endpoint, body):
            self.calls += 1; self.bodies.append((endpoint, body))
            if endpoint.endswith('login'):
                return {'token': f'account-token-{self.calls}'}
            if self.calls == 3:
                raise ServiceError('act returned HTTP 400', status=400)
            return dict(LIVE[1][0], token=body['token'])
    account = Account()
    connection = SlumbotConnection(account, HU200)
    connection.login('name', 'secret')
    connection.request('new_hand')
    with pytest.raises(ServiceError):
        connection.request('act', 'c')
    connection.new_session('name', 'secret')
    assert connection.token == 'account-token-4' and account.bodies[3] == ('/slumbot/api/login', {'username': 'name', 'password': 'secret'})
    terminal = lambda total, count, won: Hand('h', [], {'session_total': total, 'session_num_hands': count, 'winnings': won})
    # Session one: two verified hands. Session two restarts the count; its second hand was abandoned,
    # and Slumbot counted it as -50.
    hands = [terminal(50, 1, 50), terminal(250, 2, 200), terminal(-50, 1, -50), terminal(50, 3, 150)]
    assert session_accounting(hands) == {'slumbot_net_chips': 300, 'verified_net_chips': 350,
                                         'unverified_chips': -50, 'sessions': 2, 'slumbot_hands': 5}


def test_env_file_supplies_only_slumbot_credentials_and_is_never_committed(tmp_path):
    import subprocess
    from scripts.play_slumbot import read_env
    env = tmp_path / '.env'
    env.write_text('# comment\nSLUMBOT_USERNAME = player\nSLUMBOT_PASSWORD="p=ss"\nOTHER=x\nSLUMBOT_TOKEN=y\n')
    assert read_env(env) == {'SLUMBOT_USERNAME': 'player', 'SLUMBOT_PASSWORD': 'p=ss'}
    env.write_text('SLUMBOT_USERNAME=\nSLUMBOT_PASSWORD=\n')
    assert read_env(env) == {}
    assert read_env(tmp_path / 'missing') == {}
    root = Path(__file__).resolve().parents[1]
    assert subprocess.run(['git', 'check-ignore', '-q', '.env'], cwd=root).returncode == 0
    assert subprocess.run(['git', 'check-ignore', '-q', '.env.example'], cwd=root).returncode == 1


def test_an_ambiguous_request_abandons_only_its_hand_and_session_totals_count_it(tmp_path):
    from src.arena.external.session import session_accounting

    class Script:
        """Serves responses in order; AMBIGUOUS stands for a request that got no answer."""
        def __init__(self, outcomes, sent):
            self.outcomes, self.sent = list(outcomes), list(sent)
        def post(self, endpoint, body):
            outcome = self.outcomes.pop(0)
            if outcome == 'AMBIGUOUS':
                raise AmbiguousRequest('act gave no complete response')
            return {k: v for k, v in dict(outcome, token='session-token-7f3a').items() if k != 'sent'}
        def choose_action(self, view):
            return decode(self.sent.pop(0))

    a, b, c = (json.loads(json.dumps(h)) for h in (LIVE[1], LIVE[2], LIVE[4]))
    # Slumbot counted the abandoned hand b as a 100-chip loss before hand c.
    c[-1]['session_total'] = a[-1]['session_total'] - 100 + c[-1]['winnings']
    c[-1]['session_num_hands'] = a[-1]['session_num_hands'] + 2
    script = Script([*a, b[0], 'AMBIGUOUS', *c],
                    [r['sent'] for r in a[1:]] + [b[1]['sent']] + [r['sent'] for r in c[1:]])
    connection = SlumbotConnection(script, HU200)
    journal = PrefixJournal(tmp_path / 'journal.jsonl', {})
    hands = [play_hand(connection, script, 'h0', journal)]
    with pytest.raises(AmbiguousRequest):
        play_hand(connection, script, 'h1', journal)
    journal.failure('timeout-ambiguous')
    with pytest.raises(ValueError, match='stopped'):
        connection.request('new_hand')
    connection.abandon_hand()
    hands.append(play_hand(connection, script, 'h2', journal))
    journal.close()
    assert not script.outcomes and not script.sent
    assert verify_journal(tmp_path / 'journal.jsonl')['hands'] == 2
    accounting = session_accounting(hands)
    assert accounting['unverified_chips'] == -100 and accounting['slumbot_hands'] == 3
    with pytest.raises(ValueError, match='abandoned'):
        connection.abandon_hand()


def test_all_in_streets_and_exact_equity():
    from src.arena.external.analysis import allin_equity, allin_street
    assert allin_street('b20000c') == 0
    assert allin_street('b200b20000c///') == 0
    assert allin_street('b200c/b19800c//') == 1
    assert allin_street('b200c/kk/b19800c/') == 2
    assert allin_street('b200c/kk/kk/b19800c') == 3
    assert allin_street('b20000f') is None and allin_street('b200c/kk/kk/kk') is None
    royal = ['As', 'Ks', 'Qs', 'Js']
    assert allin_equity(['2c', '3d'], ['4h', '5c'], royal + ['Ts']) == 0.5      # every runout ties
    assert allin_equity(['Ts', '2c'], ['4h', '5c'], royal) == 1.0               # already a royal flush
    # Turn: aces against kings, whose only outs are the two remaining kings.
    assert allin_equity(['Ah', 'Ad'], ['Kh', 'Kd'], ['2c', '7s', '9d', 'Jc']) == 42 / 44
    # Flop: an exact count over all 990 turn and river runouts.
    flop = allin_equity(['Ah', 'Ad'], ['Kh', 'Kd'], ['2c', '7s', '9d'])
    assert 0.90 < flop < 0.93 and abs(flop * 1980 - round(flop * 1980)) < 1e-9


def test_analysis_of_the_live_fixture_journal(tmp_path):
    from src.arena.external.analysis import analyze
    replay = Replay(LIVE)
    connection = SlumbotConnection(replay, HU200)
    journal = PrefixJournal(tmp_path / 'journal.jsonl', {'fixture': True})
    for i in range(len(LIVE)):
        play_hand(connection, replay, f'h{i}', journal)
    journal.close()
    result = analyze(tmp_path / 'journal.jsonl', workers=1)
    assert result['hands'] == 12 and result['bb_per_100']['n'] == 12
    # ck/kk/b100b19900c/ is all-in on the turn with the client already drawing dead to a straight,
    # so its equity is 0 and the adjusted result equals the raw one.
    assert result['all_in_hands']['turn'] == 1
    assert result['all_in_adjusted_bb_per_100']['mean'] == result['bb_per_100']['mean']
    assert sum(v['hands'] for v in result['endings'].values()) == 12
    assert sum(sum(c.values()) for c in result['actions_by_street'].values()) == result['decisions']



def test_analysis_pools_runs_and_keeps_each_part(tmp_path):
    from src.arena.external.analysis import analyze

    def run(name, hands):
        replay = Replay(hands)
        connection = SlumbotConnection(replay, HU200)
        journal = PrefixJournal(tmp_path / f'{name}.jsonl', {'fixture': name})
        for i in range(len(hands)):
            play_hand(connection, replay, f'{name}{i}', journal)
        journal.close()
        return tmp_path / f'{name}.jsonl'

    whole = analyze(run('whole', LIVE), workers=1)
    pooled = analyze([run('a', LIVE[:5]), run('b', LIVE[5:])], workers=1)
    assert [p['hands'] for p in pooled['parts']] == [5, 7] and len(pooled['journal']) == 2
    for key in ('hands', 'decisions', 'bb_per_100', 'all_in_adjusted_bb_per_100', 'endings', 'actions_by_street',
                'responses_by_bet_size'):
        assert pooled[key] == whole[key]
    assert 'parts' not in whole



def test_a_run_for_one_account_refuses_another_username_before_any_request(tmp_path, monkeypatch, capsys):
    import scripts.play_slumbot as play
    monkeypatch.setattr(play, 'ENV_FILE', tmp_path / 'missing.env')
    for key in ('SLUMBOT_USERNAME', 'SLUMBOT_PASSWORD'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr('builtins.input', lambda prompt: 'someone-else')
    monkeypatch.setattr(play.getpass, 'getpass', lambda prompt: pytest.fail('asked for a password'))
    monkeypatch.setattr(play, 'HttpTransport', lambda *a, **k: pytest.fail('made a request'))
    monkeypatch.setattr('sys.argv', ['play_slumbot', '--hands', '1', '--login', '--account', 'deepcfr-model',
                                     '--out', str(tmp_path / 'run')])
    with pytest.raises(SystemExit) as stop:
        play.main()
    assert stop.value.code == 1 and 'for the Slumbot account deepcfr-model, not someone-else' in capsys.readouterr().err


def test_random_legal_plays_legal_actions_reproducibly_and_never_folds_for_free():
    from src.arena.external.session import RandomLegal
    from src.game.hand import Hand as GameHand, Table

    def actions(seed):
        player, chosen = RandomLegal(seed), []
        for deal in range(40):
            hand = GameHand.start(Table(('a', 'b'), (20000, 20000)), hand_id=f'r{deal}', seed=deal)
            while not hand.finished:
                view = hand.observe(hand.actor)
                action = player.choose_action(view)
                view.legal_actions.validate(action)
                assert not (action.kind == ActionKind.FOLD and view.legal_actions.call_amount == 0)
                chosen.append(action)
                hand = hand.apply(action)
        return chosen

    first = actions(7)
    assert first == actions(7) and first != actions(8)
    assert {a.kind for a in first} >= {ActionKind.FOLD, ActionKind.CALL, ActionKind.RAISE}


def test_run_charts_read_verified_winnings_and_running_intervals(tmp_path):
    from scripts.plot_slumbot_runs import running, winnings
    replay = Replay(LIVE)
    connection = SlumbotConnection(replay, HU200)
    run = tmp_path / 'run'; run.mkdir()
    journal = PrefixJournal(run / 'journal.jsonl', {'fixture': True})
    for i in range(len(LIVE)):
        play_hand(connection, replay, f'h{i}', journal)
    journal.close()
    hands = winnings([run])
    assert hands == [hand[-1]['winnings'] for hand in LIVE]
    rows = running(hands)
    assert rows[-1][0] == sum(hands) / 100 and rows[0][2] is None
    assert rows[-1][1] == pytest.approx(sum(hands) / len(hands))
    assert rows[-1][2] == pytest.approx(1.96 * statistics.stdev(hands) / len(hands) ** 0.5)


def test_style_replay_sizes_sides_and_all_ins():
    from src.arena.external.style import replay
    # The client is the small blind: it opens to 2 BB (half the pot after calling), Slumbot calls,
    # checks the flop and the client bets the pot.
    events, paid = replay({'action': 'b200c/kb400c/kk/kk', 'client_pos': 1})
    assert [(e['side'], e['kind']) for e in events[:5]] == [
        ('player', 'raise'), ('slumbot', 'call'), ('slumbot', 'check'), ('player', 'bet'), ('slumbot', 'call')]
    assert events[0]['size'] == 0.5 and events[3]['size'] == 1.0 and paid == [600, 600]
    events, paid = replay({'action': 'b200b15361b20000c///', 'client_pos': 0})
    assert [e['side'] for e in events] == ['slumbot', 'player', 'slumbot', 'player']
    assert events[2]['all_in'] and not events[1]['all_in'] and paid == [20000, 20000]


def test_style_profile_counts_statistics_and_contributions():
    from src.arena.external.style import profile
    hands = [
        {'action': 'b200c/kb400c/kk/kk', 'client_pos': 1, 'winnings': 600},    # Client opens and c-bets; showdown.
        {'action': 'f', 'client_pos': 1, 'winnings': -50},                      # Client folds its small blind.
        {'action': 'b250b750f', 'client_pos': 0, 'winnings': 250},              # Client 3-bets; Slumbot folds.
        {'action': 'cb300c/b200f', 'client_pos': 1, 'winnings': -300},          # Client limps, calls a raise, folds the flop c-bet.
    ]
    result = profile(hands)
    player, slumbot = result['player'], result['slumbot']
    assert player['vpip'] == {'rate': 0.75, 'of': 4} and player['pfr'] == {'rate': 0.5, 'of': 4}
    assert player['small_blind_first_action']['fold'] == {'rate': 1 / 3, 'of': 3}
    # Calling the big blind's raise after limping was a chance to 3-bet too.
    assert player['3bet'] == {'rate': 0.5, 'of': 2} and slumbot['fold_to_3bet'] == {'rate': 1.0, 'of': 1}
    assert player['flop_cbet'] == {'rate': 1.0, 'of': 1} and slumbot['flop_cbet'] == {'rate': 1.0, 'of': 1}
    assert player['against_flop_cbet']['fold'] == {'rate': 1.0, 'of': 1}
    assert result['streets_reached']['ended preflop']['hands'] == 2 and result['streets_reached']['showdown']['hands'] == 1
    assert sum(g['adds_bb_per_100'] for g in result['streets_reached'].values()) == pytest.approx(result['overall']['bb_per_100'])
    assert sum(g['adds_bb_per_100'] for g in result['matched_pot'].values()) == pytest.approx(result['overall']['bb_per_100'])
    # The 2 BB open is half the pot after calling; the 3-bet to 7.5 BB over 2.5 BB is pot-sized.
    assert player['bet_sizes'] == {'preflop': {'~½': 1, '~pot': 1}, 'flop': {'~pot': 1}}


def test_first_hands_cutoff_spans_pooled_runs(tmp_path):
    from src.arena.external.analysis import analyze
    from src.arena.external.style import terminals

    def run(name, hands):
        replay = Replay(hands)
        connection = SlumbotConnection(replay, HU200)
        (tmp_path / name).mkdir()
        journal = PrefixJournal(tmp_path / name / 'journal.jsonl', {'fixture': name})
        for i in range(len(hands)):
            play_hand(connection, replay, f'{name}{i}', journal)
        journal.close()
        return tmp_path / name

    a, b = run('a', LIVE[:5]), run('b', LIVE[5:])
    assert [t['winnings'] for t in terminals([a, b], 8)] == [hand[-1]['winnings'] for hand in LIVE[:8]]
    first = analyze([a / 'journal.jsonl', b / 'journal.jsonl'], workers=1, hands=8)
    assert first['hands'] == 8 and [p['hands'] for p in first['parts']] == [5, 3]
    assert analyze([a / 'journal.jsonl', b / 'journal.jsonl'], workers=1, hands=5)['journal'] == analyze(a / 'journal.jsonl', workers=1)['journal']
    with pytest.raises(ValueError, match='fewer than'):
        terminals([a], 6)
    with pytest.raises(ValueError, match='fewer than'):
        analyze(a / 'journal.jsonl', workers=1, hands=6)
