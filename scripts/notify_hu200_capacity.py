"""Deliver bounded current-campaign stage events to its existing T3 thread."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import time
import urllib.parse
import urllib.request

BASE = Path('/home/dberweger/hu200-capacity-20261015')


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'cygnus-x1' or args.base.resolve() != BASE:
        raise ValueError('Only the current PR301 root on cygnus-x1 is admitted')
    url = (BASE / 'campaign-events-url.txt').read_text().strip()
    target = urllib.parse.urlsplit(url)
    if target.scheme != 'https' or target.hostname != 'relay.t3.codes' or not target.path.startswith('/v1/hooks/'):
        raise ValueError('Unrecognized app event endpoint')
    clock = read(BASE / 'training-clock.json')
    if (BASE / 'training-deadline-amendment-20261016.json').exists():
        from scripts.recover_hu200_capacity import effective_deadline
        deadline = effective_deadline(BASE) + 3600
    else:
        deadline = clock['deadline_epoch'] + 3600
    ledger = BASE / 'callback-sent-events.json'
    sent = read(ledger) or {}

    def deliver(event):
        if event in sent:
            return True
        payload = dict(event=event, experiment='PR301', observed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        request = urllib.request.Request(url, data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'User-Agent': 'T3Code-HU200Capacity-Observer/1.0'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if not 200 <= response.status < 300:
                    return False
                receipt = dict(payload=payload, http_status=response.status)
        except Exception as exc:
            # Never log the capability URL or response bodies.
            print(json.dumps(dict(event=event, delivery='pending', error_type=type(exc).__name__,
                http_status=getattr(exc, 'code', None))), flush=True)
            return False
        sent[event] = receipt
        temporary = ledger.with_suffix('.tmp')
        with temporary.open('w') as stream:
            json.dump(sent, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(ledger)
        print(json.dumps(receipt), flush=True)
        return True

    while time.time() < deadline:
        failed = read(BASE / 'training-failed.json')
        done = read(BASE / 'training-complete.json')
        if failed is not None:
            if deliver('training_failed'):
                return
        elif done is not None:
            if deliver('training_complete'):
                return
        else:
            deliver('continuation_hook_ready')
            twins = read(BASE / 'twin-gates.json')
            memory = read(BASE / 'prefix-memory-admission.json')
            if twins and twins.get('status') == 'all-three-exact' and memory and memory.get('status') == 'passed':
                deliver('twin_prefixes_and_memory_verified')
            if read(BASE / 'twins-restored.json'):
                deliver('three_twins_restored_and_hash_verified')
            for nodes in (2000000000, 5000000000, 10000000000):
                if read(BASE / ('milestone-' + str(nodes) + '.json')):
                    deliver('all_three_' + str(nodes) + '_saves_audited')
        time.sleep(30)
    print(json.dumps(dict(status='event_observer_expired', scientific_work_untouched=True)), flush=True)


if __name__ == '__main__':
    main()
