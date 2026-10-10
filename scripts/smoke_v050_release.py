"""Bounded packaged v0.5.0 HTTP/restart/journal integration; no strength metrics.

`release` plays restricted, free and spectator hands through the real CLI, restarts it
before a pending bot decision, then `audit` independently replays every journal."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from time import monotonic, sleep, time
import urllib.error
import urllib.request
from uuid import uuid4

from src.policies.v050_bundle import INFERENCE, MODEL, RELEASE


def put(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def browser_receipt(path, seconds=600):
    # Writers should publish with atomic rename. A remote copy can expose an
    # empty/partial file first; only a complete JSON object acknowledges closeout.
    until = monotonic() + seconds
    while monotonic() < until:
        try:
            result = json.loads(path.read_text())
            if isinstance(result, dict):
                return result
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        sleep(.2)
    raise TimeoutError('Browser allowance exhausted')


class Runtime:
    def __init__(self, bundle, data, out, source):
        self.bundle, self.data, self.out, self.source = bundle, data, out, source
        self.process = None
        self.starts = []
        self.port = 8768

    def start(self):
        started = monotonic()
        log = (self.out / f'server-{len(self.starts)}.log').open('x')
        self.process = subprocess.Popen([sys.executable, '-m', 'src.play_api.server',
            '--v050', str(self.bundle), '--stack-bb', '100',
            '--data-dir', str(self.data), '--source-version', self.source,
            '--port', str(self.port)], stdout=log, stderr=subprocess.STDOUT)
        log.close()
        while monotonic() - started < 900:
            if self.process.poll() is not None:
                raise RuntimeError('v0.5.0 server exited during startup')
            token = self.data / 'access.token'
            if token.exists():
                self.token = token.read_text().strip()
                try:
                    self.call('/api/models')
                    result = {'pid': self.process.pid, 'load_seconds': monotonic() - started}
                    self.starts.append(result)
                    return result
                except urllib.error.URLError:
                    pass
            sleep(.2)
        raise TimeoutError('Bounded CLI startup exhausted')

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(5)
            self.process = None

    def call(self, path, body=None, key=None, expected=200):
        headers = {'X-Play-Token': self.token}
        if body is not None:
            headers.update({'Content-Type': 'application/json', 'Idempotency-Key': key or uuid4().hex})
        request = urllib.request.Request(f'http://127.0.0.1:{self.port}' + path,
            data=None if body is None else json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status, value = response.status, json.load(response)
        except urllib.error.HTTPError as error:
            status, value = error.code, json.load(error)
        with (self.out / 'http-journal.jsonl').open('a') as stream:
            stream.write(json.dumps({'pid': self.process.pid, 'path': path, 'body': body,
                                     'status': status, 'response': value}) + '\n')
        assert status == expected, (path, status, value)
        return value


def request(state):
    return {'revision': state['revision'], 'handId': state['hand']['id']}


def prefix(state):
    return '/api/sessions/' + state['sessionId']


def exercise(runtime, counts, label):
    started = monotonic()
    result = {'sessions': [], 'hands': 0, 'off_menu_550_raises': 0, 'all_in_raises': 0}
    catalog = runtime.call('/api/models')
    assert catalog['default'] == RELEASE and len(catalog['models']) == 1
    assert catalog['models'][0]['inference'] == INFERENCE
    runtime.call('/api/sessions', {'modelVersion': 'v0.4.2'}, expected=400)
    runtime.call('/api/sessions', {'sessionType': 'spectator',
                                 'modelVersions': [RELEASE, 'v0.4.2']}, expected=400)
    for mode, hands in counts.items():
        if mode == 'spectator':
            state = runtime.call('/api/sessions', {'sessionType': 'spectator',
                                    'modelVersions': [RELEASE, RELEASE]})
        else:
            state = runtime.call('/api/sessions', {'playMode': mode, 'visibility': 'developer'})
        result['sessions'].append({'id': state['sessionId'], 'mode': mode, 'hands': hands})
        for number in range(hands):
            state = runtime.call(prefix(state) + '/hands', {'revision': state['revision']})
            assert state['table']['stack'] == 10000 and state['hand']['button'] == number % 2
            players = state['hand']['perspectives'][0]['players'] if mode == 'spectator' else state['hand']['players']
            assert sorted(p['stack'] for p in players) == [9900, 9950]
            for step in range(100):
                if state['phase'] == 'finished':
                    break
                if mode == 'spectator' or state['hand']['actor'] == 1:
                    state = runtime.call(prefix(state) + '/advance', request(state))
                else:
                    legal = state['hand']['legal']; body = request(state)
                    amount = 550 if mode == 'free' and number % 4 == 0 and step == 0 else legal['maxRaiseTo'] if mode == 'free' and number == 1 else None
                    if amount is not None and 'raise' in legal['kinds'] and legal['minRaiseTo'] <= amount <= legal['maxRaiseTo']:
                        body.update(kind='raise', raiseTo=amount)
                        result['off_menu_550_raises'] += amount == 550
                        result['all_in_raises'] += amount == 10000
                    else:
                        body.update(kind='check' if 'check' in legal['kinds'] else 'call', raiseTo=None)
                    state = runtime.call(prefix(state) + '/actions', body)
                if mode != 'spectator':
                    public = json.dumps(state)
                    assert not any(word in public for word in ('dealSeed', 'botRng', 'botDecisions', 'deck'))
            else:
                raise AssertionError('100-action hand bound')
            assert state['phase'] == 'finished'
            if mode == 'spectator':
                assert sum(state['sessionChips']) == 0
            result['hands'] += 1
        runtime.call(prefix(state) + '/history')
    result['seconds'] = monotonic() - started
    put(runtime.out / (label + '-summary.json'), result)
    return result


def audit(data, bundle, out):
    from src.policies.v050 import load_policy
    from src.play_api.play_audit import audit_state
    from src.play_api.spectator_audit import audit_states
    from src.play_api.service import PlayService, _model_info, PlayError
    from src.play_api.spectator import SpectatorService
    from src.play_api.configuration import PlayTable
    started = monotonic(); policy = load_policy(bundle)
    loaded = monotonic() - started
    base = data / RELEASE
    human = PlayService(base / 'private.sqlite', policy)
    identity = {'version': RELEASE, **_model_info(policy)}
    spectator = SpectatorService(base / 'spectator.sqlite', {RELEASE: policy}, {RELEASE: identity})
    try:
        human_rows = [json.loads(r[0]) for r in human.db.execute('SELECT state FROM sessions')]
        spectator_rows = [json.loads(r[0]) for r in spectator.db.execute('SELECT state FROM sessions')]
        for row in human_rows:
            audit_state(row, policy); human.verify_replay(row['sessionId'])
        for row in spectator_rows:
            spectator.verify_replay(row['sessionId'])
        spect = audit_states(spectator_rows, {RELEASE: policy}, {RELEASE: identity})
        try:
            PlayService(out / 'wrong-table.sqlite', policy, table=PlayTable())
        except ValueError:
            pass
        else:
            raise AssertionError('HU20 table admitted')
        if human_rows:
            policy.configure_translation(None); policy.adapter_id = 'direct-v1'
            try:
                changed = PlayService(base / 'private.sqlite', policy)
                try:
                    changed.state(human_rows[0]['sessionId'])
                finally:
                    changed.close()
            except PlayError as error:
                assert error.status == 409
            else:
                raise AssertionError('Changed inference admitted')
        counts = {m: sum(r['handsPlayed'] for r in human_rows if r['playMode'] == m) for m in ('free', 'restricted')}
        counts['spectator'] = spect['hands']
        modes = {}
        for row in human_rows:
            for hand in row['history']:
                for record in hand['botDecisions']:
                    mode = record['telemetry']['mode']; modes[mode] = modes.get(mode, 0) + 1
        result = {'status': 'verified', 'reader_load_seconds': loaded,
                  'audit_seconds': monotonic() - started - loaded, 'counts': counts,
                  'human_inference': modes, 'spectator': spect,
                  'model_sha256': MODEL['sha256'], 'inference': INFERENCE,
                  'rejections': ['HU20 table', 'changed inference identity'],
                  'independent_modules': ['play_audit.audit_state', 'spectator_audit.audit_states', 'service.verify_replay']}
        put(out / 'independent-audit.json', result)
        return result
    finally:
        human.close(); spectator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('pilot', 'main', 'release', 'audit'))
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'audit':
        print(json.dumps(audit(args.data, args.bundle, args.out))); return
    runtime = Runtime(args.bundle, args.data, args.out, args.source)
    try:
        startup = runtime.start()
        if args.mode == 'pilot':
            result = exercise(runtime, {'restricted': 1, 'free': 1, 'spectator': 1}, 'pilot')
            result['startup'] = startup
        else:
            # The same in-progress hand and exact reply survive a real fresh CLI
            # load in a different PID, including its stored idempotent ack.
            state = runtime.call('/api/sessions', {'playMode': 'free', 'visibility': 'developer'})
            state = runtime.call(prefix(state) + '/hands', {'revision': 0})
            key = uuid4().hex; body = {**request(state), 'kind': 'raise', 'raiseTo': 550}
            response = runtime.call(prefix(state) + '/actions', body, key)
            assert response['hand']['actor'] == 1 and response['phase'] != 'finished'
            old_pid = runtime.process.pid; runtime.stop(); restart = runtime.start()
            assert restart['pid'] != old_pid
            assert runtime.call(prefix(state) + '/actions', body, key) == response
            assert runtime.call(prefix(state)) == response
            advance_key = uuid4().hex; pending = request(response)
            response = runtime.call(prefix(response) + '/advance', pending, advance_key)
            assert runtime.call(prefix(response) + '/advance', pending, advance_key) == response
            # Finish this restart hand as one of eight free hands.
            for step in range(100):
                if response['phase'] == 'finished': break
                if response['hand']['actor'] == 1:
                    response = runtime.call(prefix(response) + '/advance', request(response))
                else:
                    legal = response['hand']['legal']
                    response = runtime.call(prefix(response) + '/actions', {**request(response),
                        'kind': 'check' if 'check' in legal['kinds'] else 'call', 'raiseTo': None})
            assert response['phase'] == 'finished'
            counts = {'restricted': 4, 'free': 7, 'spectator': 8} if args.mode == 'main' else {'restricted': 2, 'free': 3, 'spectator': 2}
            result = exercise(runtime, counts, args.mode)
            result['hands'] += 1; result['restart_hand_session'] = state['sessionId']
            result['starts'] = runtime.starts; result['lost_reply_verified'] = True
            if args.mode == 'main':
                put(args.out / 'browser-ready.json', {'port': runtime.port, 'pid': runtime.process.pid,
                                                       'started': time(), 'max_seconds': 600})
                result['browser'] = browser_receipt(args.out / 'browser-done.json')
        put(args.out / 'summary.json', result)
        print(json.dumps(result))
    finally:
        runtime.stop()


if __name__ == '__main__':
    main()
