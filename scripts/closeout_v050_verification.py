"""Audit/archive-only recovery of the recorded browser completion-copy race."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
from time import monotonic, time

import psutil
from scripts.run_v050_verification import Verification, put, resource
from src.policies.v050_bundle import sha, verify_bundle

SOURCE = '155356b9dc656ecdb217beba166d5ee5f4d44c34'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--cloud', type=Path, required=True)
    args = parser.parse_args(); out = args.out
    baseline = json.loads((out / 'baseline.json').read_text())
    previous = json.loads((out / 'resources-summary-resumed.json').read_text())
    failure = json.loads((out / 'failure-resumed.json').read_text())
    browser = json.loads((out / 'main/browser-done.json').read_text())
    log = (out / 'main.log').read_text()
    if (failure['error'] != "RuntimeError('main failed')" or failure['source'] != SOURCE
            or 'json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)' not in log
            or previous['failure'] is not None or baseline['cap_seconds'] != 3600
            or psutil.pid_exists(66345) or psutil.pid_exists(78839)
            or browser['complete_hands'] != 2 or browser['partial_hands'] != 1
            or any((out / name).exists() for name in ('closeout-continuation.json', 'main/independent-audit.json', 'archive-receipt.json'))):
        raise ValueError('Exact inactive browser-copy race proof differs')
    guard = object.__new__(Verification)
    guard.out = out; guard.started = baseline['started']; guard.deadline = baseline['deadline']
    guard.tick_start = monotonic() - (time() - guard.started)
    guard.lock = (Path.home() / 'Local/.hu100-m4-research.lock').open('a+')
    fcntl.flock(guard.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    guard.stop = threading.Event(); guard.failure = None; guard.child = None
    guard.handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    for sig in guard.handlers: signal.signal(sig, guard.interrupt)
    guard.resource_log = out / 'resources-closeout.jsonl'
    guard.summary_name = 'resources-summary-closeout.json'
    guard.thread = None; guard.samples = []; guard.peak = previous['peak_family_rss_bytes']
    guard.baseline = baseline['host']; resource.OUT = out
    for name in ('resources.jsonl', 'resources-resumed.jsonl'):
        for line in (out / name).read_text().splitlines():
            row = json.loads(line)
            if (resource.limits(row, guard.baseline['swap_bytes'], row['family_rss_bytes'])
                    or row['family_rss_bytes'] >= resource.FAMILY_SOFT or row['swap_bytes'] > 3000000000):
                raise ValueError('Prior actual guard breach')
    guard.check()
    quote = 2 * json.loads((out / 'admission.json').read_text())['pilot_reader_load_seconds'] + 300
    if guard.deadline - time() < quote:
        raise TimeoutError('Existing audit/archive closeout cannot fit original budget')
    put(out / 'closeout-continuation.json', {'source': SOURCE,
        'closeout_source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'started': time(), 'original_started': guard.started, 'original_deadline': guard.deadline,
        'monitoring_gap_seconds': time() - previous['finished'], 'quote_seconds': quote,
        'reason': 'remote browser acknowledgment exposed empty file before copy completed',
        'original_failure_sha256': sha(out / 'main.log'), 'no_new_hands': True,
        'no_server_restart_or_model_copy': True, 'original_baseline_preserved': True})
    subprocess.run(['git', 'archive', '-o', str(out / 'source-closeout.tar'), 'HEAD'], check=True)
    guard.thread = threading.Thread(target=guard.monitor, daemon=True); guard.thread.start()
    try:
        guard.stable_admission()
        verify_bundle(args.bundle, SOURCE)
        guard.run('main-audit', [sys.executable, '-m', 'scripts.smoke_v050_release', 'audit',
            '--bundle', str(args.bundle), '--out', str(out / 'main'), '--data', str(out / 'main-data'), '--source', SOURCE])
        for name in ('resources.jsonl', 'resources-resumed.jsonl'):
            (out / (name.replace('.jsonl', '-closed.jsonl'))).write_bytes((out / name).read_bytes())
        (out / 'resources-closeout-prearchive.jsonl').write_bytes(guard.resource_log.read_bytes())
        guard.run('archive', [sys.executable, '-m', 'scripts.archive_v050_evidence',
            '--out', str(out), '--cloud', str(args.cloud), '--source', SOURCE])
        guard.check()
    except BaseException as error:
        put(out / 'failure-closeout.json', {'error': repr(error), 'at': time(), 'source': SOURCE})
        raise
    finally:
        print(json.dumps(guard.finish()))


if __name__ == '__main__': main()
