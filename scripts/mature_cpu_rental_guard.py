"""Independent network-only cutoff for exact, prospectively owned pod names."""

import argparse
import json
import os
from pathlib import Path
import time
import tomllib
import urllib.error
import urllib.request


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
    temporary.replace(path)


def api(key_path, path, method='GET', data=None):
    key = tomllib.loads(key_path.read_text())['apikey']
    request = urllib.request.Request('https://api.runpod.io' + path,
        data=None if data is None else json.dumps(data).encode(), method=method,
        headers={'Authorization': 'Bearer ' + key, 'User-Agent': 'runpodctl/1.14.5',
                 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        # Retain only a bounded provider message, excluding echoed request data.
        try:
            response = json.loads(error.read())
            message = response.get('message', response.get('error', ''))
            message = message if isinstance(message, str) else ''
            message = message.replace(key, '[redacted]')[:300]
        except Exception:
            message = ''
        raise RuntimeError(f'Provider HTTP {error.code}, {method} {path.split("?")[0]}: {message}') from None


def owned_pods(pods, names):
    return [pod for pod in pods if pod['name'] in names]


def check_quote(config, pod, maximum_total_hourly):
    return (not pod.get('gpu') and pod.get('cpu', {}).get('id') == config['cpu_id']
            and pod['cpu'].get('vcpuCount') == config['vcpus']
            and pod['cpu'].get('memory') == config['ram_gb']
            and 0 < pod.get('cost', 0) <= maximum_total_hourly)


def watch(lease, key_path):
    state = json.loads(lease.read_text())
    names = set(state['names'])
    out = lease.with_name('watchdog.json')
    if len(names) != 6 or state['deadline'] - state['started'] > 7200:
        raise ValueError('Six unique names and immutable <= two-hour lease required')
    existing = owned_pods(api(key_path, '/v2/pods')['pods'], names)
    if existing:
        raise ValueError('Owned names existed before arming')
    write(out, {'status': 'armed', 'pid': os.getpid(), 'heartbeat': time.time(),
                'deadline': state['deadline'], 'names': sorted(names)})
    while time.time() < state['deadline']:
        if lease.with_name('operator-finished.json').exists():
            remaining = owned_pods(api(key_path, '/v2/pods')['pods'], names)
            if not remaining:
                write(out, {'status': 'operator-finished', 'heartbeat': time.time(),
                            'no_owned_pods': True})
                return
        write(out, {'status': 'armed', 'pid': os.getpid(), 'heartbeat': time.time(),
                    'deadline': state['deadline']})
        time.sleep(min(15, max(0, state['deadline'] - time.time())))
    terminated = []
    for attempt in range(10):
        try:
            # Exact frozen names also cover a creation whose response was lost.
            for pod in owned_pods(api(key_path, '/v2/pods')['pods'], names):
                api(key_path, '/v2/pods/' + pod['id'], 'DELETE')
                terminated.append(pod['id'])
            if not owned_pods(api(key_path, '/v2/pods')['pods'], names):
                write(out, {'status': 'cutoff-verified', 'heartbeat': time.time(),
                            'terminated_ids': sorted(set(terminated))})
                return
        except Exception as error:
            write(out, {'status': 'cutoff-error', 'heartbeat': time.time(),
                        'attempt': attempt + 1, 'error_type': type(error).__name__})
        time.sleep(10)
    raise RuntimeError('Owned rental cutoff could not be verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lease', type=Path, required=True)
    parser.add_argument('--key', type=Path, required=True)
    args = parser.parse_args()
    watch(args.lease, args.key)
