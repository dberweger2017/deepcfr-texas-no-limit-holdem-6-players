"""Real HU200 CLI play, pending-decision restart and independent journal audit."""
import argparse
import json
from pathlib import Path
from time import monotonic
from uuid import uuid4

from scripts.smoke_hu100_release import Runtime, browser_receipt, prefix, put, request
from src.policies.hu200_bundle import INFERENCE, MODELS, verify_bundle


def finish(runtime, state, *, mode='free', wager=None):
    used = False
    for _ in range(100):
        if state['phase'] == 'finished': return state, used
        if mode == 'spectator' or state['hand']['actor'] == 1:
            state = runtime.call(prefix(state) + '/advance', request(state))
        else:
            legal = state['hand']['legal']
            body = request(state)
            amount = legal['maxRaiseTo'] if wager == 'all-in' else wager
            if (not used and amount is not None and 'raise' in legal['kinds']
                    and legal['minRaiseTo'] <= amount <= legal['maxRaiseTo']):
                body.update(kind='raise', raiseTo=amount); used = True
            else:
                body.update(kind='check' if 'check' in legal['kinds'] else 'call', raiseTo=None)
            state = runtime.call(prefix(state) + '/actions', body)
            assert not any(word in json.dumps(state) for word in ('dealSeed', 'botRng', 'botDecisions', 'deck'))
    raise AssertionError('100-action hand bound exceeded')


def exercise(runtime):
    catalog = runtime.call('/api/models')
    assert catalog['default'] == 'v0.5.5' and len(catalog['models']) == 1
    assert catalog['models'][0]['inference'] == INFERENCE
    result = {'sessions': [], 'off_menu_wagers': [], 'all_in_wagers': 0, 'hands': 0}
    runtime.call('/api/sessions', {'modelVersion': 'v0.5.3'}, expected=400)
    for mode, count in [('restricted', 4), ('free', 8), ('spectator', 12)]:
        options = ({'sessionType': 'spectator', 'modelVersions': ['v0.5.5'] * 2}
                   if mode == 'spectator' else {'playMode': mode, 'visibility': 'developer'})
        state = runtime.call('/api/sessions', options)
        result['sessions'].append({'id': state['sessionId'], 'mode': mode})
        for number in range(count):
            state = runtime.call(prefix(state) + '/hands', {'revision': state['revision']})
            assert state['table']['stack'] == 20000 and state['hand']['button'] == number % 2
            players = state['hand']['perspectives'][0]['players'] if mode == 'spectator' else state['hand']['players']
            assert sorted(p['stack'] for p in players) == [19900, 19950]
            wager = [550, 'all-in', 800, None][number % 4] if mode == 'free' else None
            state, used = finish(runtime, state, mode=mode, wager=wager)
            if used:
                if wager == 'all-in': result['all_in_wagers'] += 1
                else: result['off_menu_wagers'].append(wager)
            if mode == 'spectator': assert sum(state['sessionChips']) == 0
            result['hands'] += 1
        runtime.call(prefix(state) + '/history')
    assert {550, 800} <= set(result['off_menu_wagers']) and result['all_in_wagers']
    return result


def run(bundle, data, out, source, *, browser_done=None):
    runtime = Runtime(bundle, data, out, source, flag='--hu200-release', stack_bb=200,
                      port=8779, load_timeout=4 * 3600)
    try:
        runtime.start()
        state = runtime.call('/api/sessions', {'playMode': 'free', 'visibility': 'developer'})
        state = runtime.call(prefix(state) + '/hands', {'revision': state['revision']})
        body = {**request(state), 'kind': 'raise', 'raiseTo': 550}; key = uuid4().hex
        pending = runtime.call(prefix(state) + '/actions', body, key)
        assert pending['hand']['actor'] == 1 and pending['phase'] != 'finished'
        old_pid = runtime.process.pid; runtime.stop(); runtime.start()
        assert runtime.process.pid != old_pid
        assert runtime.call(prefix(state)) == pending
        assert runtime.call(prefix(state) + '/actions', body, key) == pending
        advance = request(pending); key = uuid4().hex
        response = runtime.call(prefix(pending) + '/advance', advance, key)
        assert runtime.call(prefix(pending) + '/advance', advance, key) == response
        finish(runtime, response)
        result = exercise(runtime)
        result.update(restart_session=state['sessionId'], restart_and_idempotency_verified=True)
        if browser_done is not None:
            put(out / 'browser-ready.json', {'source': source, 'port': runtime.port,
                                           'automated_hands_completed': result['hands'] + 1})
            result['browser'] = browser_receipt(browser_done, seconds=3 * 3600)
        runtime.stop(); result['starts'] = runtime.starts
        put(out / 'smoke-summary.json', result)
        return result
    finally:
        runtime.stop()


def audit(bundle, data, out):
    from src.policies.hu200_release import load_policy
    from src.play_api.service import PlayService, _model_info
    from src.play_api.spectator import SpectatorService
    from src.play_api.configuration import PlayTable
    from src.play_api.play_audit import audit_state
    from src.play_api.spectator_audit import audit_states
    start = monotonic(); release, policy = load_policy(bundle); loaded = monotonic() - start
    human = PlayService(data / release / 'private.sqlite', policy)
    identity = {'version': release, **_model_info(policy)}
    spectator = SpectatorService(data / release / 'spectator.sqlite', {release: policy}, {release: identity})
    try:
        humans = [json.loads(r[0]) for r in human.db.execute('SELECT state FROM sessions')]
        spectators = [json.loads(r[0]) for r in spectator.db.execute('SELECT state FROM sessions')]
        for row in humans:
            audit_state(row, policy); human.verify_replay(row['sessionId'])
        for row in spectators: spectator.verify_replay(row['sessionId'])
        spect = audit_states(spectators, {release: policy}, {release: identity})
        refused = []
        for stack in (2000, 10000):
            try: PlayService(out / f'wrong-table-{stack}.sqlite', policy, table=PlayTable(stack=stack))
            except ValueError: refused.append(f'table-{stack}')
            else: raise AssertionError('Wrong table admitted')
        policy.adapter_id = 'direct-v1'
        try: PlayService(data / release / 'private.sqlite', policy)
        except ValueError: refused.append('inference identity')
        else: raise AssertionError('Wrong adapter admitted')
        modes = {}
        for row in humans:
            for hand in row['history']:
                for decision in hand['botDecisions']:
                    mode = decision['telemetry']['mode']; modes[mode] = modes.get(mode, 0) + 1
        result = {'status': 'verified', 'reader_load_seconds': loaded,
                  'audit_seconds': monotonic() - start - loaded,
                  'human_completed': sum(r['handsPlayed'] for r in humans),
                  'human_partial': sum(bool(r['current'] and not r['current'].get('completed')) for r in humans),
                  'spectator': spect, 'human_inference': modes, 'refused': refused,
                  'model_sha256': MODELS['blend']['sha256'],
                  'independent_modules': ['play_audit.audit_state', 'spectator_audit.audit_states', 'service.verify_replay']}
        put(out / 'independent-audit.json', result); return result
    finally:
        human.close(); spectator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('play', 'audit'))
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source', required=True)
    parser.add_argument('--browser-done', type=Path)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    action = (lambda b, d, o, s: run(b, d, o, s, browser_done=args.browser_done)
              if args.stage == 'play' else audit(b, d, o))
    print(json.dumps(action(args.bundle, args.data, args.out, args.source)))


if __name__ == '__main__': main()
