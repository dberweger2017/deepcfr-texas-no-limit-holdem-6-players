"""Explicit bounded native-Chrome smoke when the collaborative preview is unavailable."""

import argparse
import base64
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

from tests.play_ui.browser_smoke import CDP, CHROME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8773)
    parser.add_argument('--data-dir', type=Path, default=Path('results/spectator-browser'))
    parser.add_argument('--attempt', default='01')
    parser.add_argument('--human-only', action='store_true')
    args = parser.parse_args()
    out = args.data_dir / f'headless-smoke-{args.attempt}'
    out.mkdir(parents=True, exist_ok=True)
    token = (args.data_dir / 'access.token').read_text().strip()
    debugging = 9233
    base = f'http://127.0.0.1:{args.port}'
    process = subprocess.Popen([CHROME, '--headless=new', '--no-first-run', '--disable-gpu',
                                '--window-size=1280,900', f'--user-data-dir={out.resolve() / "profile"}',
                                f'--remote-debugging-port={debugging}', '--remote-allow-origins=*', 'about:blank'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cdp = None
    report = {'status': 'running', 'spectator': [], 'human': []}
    try:
        for _ in range(100):
            try:
                pages = json.load(urlopen(f'http://127.0.0.1:{debugging}/json', timeout=1))
                page = next(p for p in pages if p['type'] == 'page')
                cdp = CDP(page['webSocketDebuggerUrl'])
                break
            except (OSError, StopIteration):
                time.sleep(.1)
        if cdp is None:
            raise RuntimeError('Chrome did not start')
        cdp.call('Page.enable'); cdp.call('Runtime.enable')
        cdp.call('Page.navigate', {'url': base})
        cdp.wait("document.readyState === 'complete' && typeof saved !== 'undefined'")
        cdp.js(f"document.querySelector('#token').value={json.dumps(token)};document.querySelector('#gate-form').requestSubmit()")
        cdp.wait("!document.querySelector('#setup').hidden && availableModels.length===2")
        checks_script = Path('tests/play_ui/spectator_browser_checks.js').read_text()
        pairings = [] if args.human_only else (['v0.4.0', 'v0.4.1'], ['v0.4.1', 'v0.4.1'], ['v0.4.0', 'v0.4.0'])
        for versions in pairings:
            cdp.js("document.querySelector('input[name=sessionType][value=spectator]').click()")
            cdp.js(f"document.querySelector('#bot-a-model').value={json.dumps(versions[0])}; document.querySelector('#bot-b-model').value={json.dumps(versions[1])}; spectatorIdentities(); document.querySelector('#create').click()")
            cdp.wait("typeof state !== 'undefined' && state?.sessionType==='spectator' && !busy")
            cdp.js(checks_script)
            for _ in range(80):
                if cdp.js("state.handsPlayed===2"):
                    break
                cdp.js('spectatorSmoke.step()')
            else:
                raise RuntimeError('Spectator hand limit')
            # The displayed historical distribution must remain inspectable.
            cdp.js("document.querySelector('#past-hands').click()")
            cdp.wait("document.querySelectorAll('#events > details').length===2")
            cdp.js("document.querySelector('#events > details').open=true; document.querySelector('#events .past-decision').open=true")
            cdp.wait("!!document.querySelector('#events .probability-table')")
            result = cdp.js("({sessionId:state.sessionId,versions:state.models.map(m=>m.version),hands:state.handsPlayed,checks:spectatorSmoke.checks,paused:!spectatorPlayback.running})")
            assert result['versions'] == versions and result['paused'] and result['hands'] == 2
            report['spectator'].append(result)
            screenshot = cdp.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': True})
            (out / f'{versions[0]}-vs-{versions[1]}.png').write_bytes(base64.b64decode(screenshot['data']))
            # Recovery loads the same complete session and always resumes paused.
            session_id = result['sessionId']
            previous_document = cdp.js('performance.timeOrigin')
            cdp.call('Page.navigate', {'url': base})
            cdp.wait(f"performance.timeOrigin !== {json.dumps(previous_document)} && document.readyState === 'complete' && typeof state !== 'undefined' && state?.sessionId === {json.dumps(session_id)} && !busy")
            assert cdp.js("({id:state.sessionId,hands:state.handsPlayed,paused:!spectatorPlayback.running})") == {
                'id': session_id, 'hands': 2, 'paused': True}
            result['reload'] = {'previousDocument': previous_document,
                                'recoveredDocument': cdp.js('performance.timeOrigin'),
                                'sessionId': session_id, 'paused': True}
            cdp.js("document.querySelector('#new-session').click()")
            cdp.wait("typeof state !== 'undefined' && !document.querySelector('#setup').hidden && state===null")
        for version in ('v0.4.1', 'v0.4.0'):
            for mode in ('restricted', 'free'):
                cdp.js("document.querySelector('input[name=sessionType][value=casual]').click()")
                cdp.js(f"document.querySelector('#model-version').value={json.dumps(version)}; document.querySelector('input[name=playMode][value={mode}]').click(); document.querySelector('#create').click()")
                cdp.wait("typeof state !== 'undefined' && state?.sessionType==='casual' && !busy")
                cdp.js("document.querySelector('#new-hand').click()")
                cdp.wait("typeof state !== 'undefined' && state?.hand?.actor===0 && !busy")
                if mode == 'free':
                    cdp.js("document.querySelector('.raise-row input[type=text]').value='2.01'; document.querySelector('.raise-row .primary').click()")
                    cdp.wait("!busy && !processingBot && (state.phase==='finished' || state.hand.actor===0)")
                for _ in range(100):
                    if cdp.js("state.phase==='finished'"):
                        break
                    cdp.js("[...document.querySelectorAll('#controls button')].find(b=>/^(check|call)/i.test(b.textContent)).click()")
                    cdp.wait("!busy && !processingBot && (state.phase==='finished' || state.hand.actor===0)")
                else:
                    raise RuntimeError('Human hand did not settle')
                report['human'].append(cdp.js("({sessionId:state.sessionId,model:state.model.sha256,mode:state.playMode,hands:state.handsPlayed})"))
                cdp.js("document.querySelector('#new-session').click()")
                cdp.wait("typeof state !== 'undefined' && !document.querySelector('#setup').hidden && state===null")
        report['status'] = 'passed'
    finally:
        (out / 'browser-checks.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        if cdp is not None:
            cdp.socket.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=5)
    print(json.dumps({'status': report['status'], 'spectator_hands': sum(r['hands'] for r in report['spectator']),
                      'displayed_decisions': sum(len(r['checks']) for r in report['spectator']),
                      'human_hands': sum(r['hands'] for r in report['human'])}))


if __name__ == '__main__':
    main()
