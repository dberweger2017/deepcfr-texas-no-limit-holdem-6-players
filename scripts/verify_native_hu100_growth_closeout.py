"""Verify exact recorded source after an untracked-binary manifest refusal.

Read frozen completed play/replay/reproduction without repeating any hand. The
original failed report/stage stays immutable. Only the named non-source binary
exception is admitted, after independently checking the exact Git source hash.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from time import time

from scripts.run_native_hu100_growth import ROOT, BASELINE, claim, read, snapshot, seal, lock, identity
from scripts.hu20_scaling_supervise import run as supervise
from scripts.report_native_hu100_growth import report
from src.arena.schedule import digest
from src.policies.files import file_hash

FROZEN_SOURCE = 'ed9be57b0201b2967186fc9a1077545cbb294286'
BINARY = 'b547173b838d896888973f4cf283ba66a60b61781a29a7ed719ebcbadb448f2c'


def git_source_fingerprint(source):
    # Reconstruct from Git blobs, not the potentially dirty filesystem.
    blob = subprocess.check_output(['git', 'archive', source])
    values = {}
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        for member in archive:
            if member.isfile() and (member.name.endswith('.py') or member.name.startswith('requirements')):
                values[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    return digest(values)


def validate(run):
    s = read(run / 'state.json'); q = read(run / 'frozen-final.json')
    if (s['status'] != 'failed' or s.get('error') != "ValueError('Terminal child/guard failure: report')"
        or s['pins']['source'] != FROZEN_SOURCE or q['source'] != FROZEN_SOURCE
        or q['deadline'] != s['deadline'] or time() >= s['deadline']
        or s['operations'].get('report', {}).get('status') != 'failed'):
        raise ValueError('Only the exact frozen summary refusal is admitted')
    if file_hash(run / 'settings.json') != q['settings_sha256']:
        raise ValueError('Fixed model settings changed')
    if q['blocks_per_opponent'] != 2048 or q['final_root'] != 2026100820412:
        raise ValueError('Recorded final sample differs')
    g = read(run / 'report-guard/campaign.json')
    if (g['failure'] is not None or g['attempts'][0]['guard_failure'] is not None
        or g['attempts'][0]['exit_code'] != 1
        or "ValueError: Changed/dirty source or model" not in (run / 'report-guard/report.log').read_text()):
        raise ValueError('Not the recorded source-metadata refusal')
    # Qualification at every play already rejected changed tracked files. Rehash
    # all stored outputs before permitting any postprocessing exception.
    for name, operation in s['operations'].items():
        if name == 'report': continue
        if operation['status'] != 'complete': raise ValueError('Incomplete science operation')
        for path, expected in operation['outputs'].items():
            if file_hash(Path(path)) != expected: raise ValueError('Completed science output changed')
        if file_hash(run / (name + '-guard/campaign.json')) != operation['guard_sha256']:
            raise ValueError('Completed guard changed')
    if file_hash(run / 'frozen-final.json') != s['freeze_sha256']:
        raise ValueError('Final counts changed')
    if file_hash(run / 'frozen-schedule.json') != q['schedule_sha256']:
        raise ValueError('Schedule changed')
    fingerprint = git_source_fingerprint(FROZEN_SOURCE)
    dirty = []
    for path in run.glob('*/**/manifest.json'):
        m = read(path)
        if m['revision'] != FROZEN_SOURCE or m['source_sha256'] != fingerprint:
            raise ValueError('Recorded executed source differs from frozen Git blobs')
        dirty.append({'path': str(path), 'manifest_sha256': file_hash(path), 'dirty': m['dirty']})
    if len(dirty) != 40 or not all(m['dirty'] for m in dirty):
        raise ValueError('Expected exactly forty retained pilot/final/reproduction manifests')
    if file_hash(ROOT / 'bin/hu20-trainer') != BINARY:
        raise ValueError('Named untracked binary changed')
    return s, fingerprint, dirty


def worker(run, out, dest):
    source = identity()
    execution = read(run / 'postprocessing-claim.json')
    if execution['launcher_pid'] != os.getppid() or execution['controller_sha256'] != file_hash(Path(__file__)):
        raise ValueError('Only the claimed supervised controller child is admitted')
    claim(run / 'postprocessing-worker-claim.json', {'pid': os.getpid(), 'parent': os.getppid()})
    for name in ('qualification', 'source-review'):
        r = read(ROOT / 'results' / (name + '.json'))
        if r['source'] != source or r['status'] != 'passed':
            raise ValueError('Exact new-source qualification/review required')
    s, fingerprint, manifests = validate(run)
    out.mkdir(exist_ok=False)
    claim(out / 'state.json', {'status': 'postprocessing', 'deadline': s['deadline'],
          'original_stage_sha256': file_hash(run / 'state.json'), 'source': source})
    claim(out / 'source-provenance.json', {'status': 'verified', 'source': FROZEN_SOURCE,
        'git_source_fingerprint': fingerprint, 'recorded_manifests': manifests,
        'exception': 'untracked bin/hu20-trainer with pinned binary hash; no source file change',
        'binary_sha256': BINARY, 'original_failed_stage_preserved': True})
    report(run, output=out, verified_source_fingerprint=fingerprint)
    # Include all science and failed report/initial setup. Do not alter raw files.
    # Seal reads the original stage root; final metadata lives beside it.
    seal(run, dest, extra_paths={
        **{'postprocessing/' + p.name: p for p in out.iterdir() if p.is_file()},
        'original-evaluation-source.tar': ROOT / 'results/source-ed9.tar',
        'original-evaluation-review.json': ROOT / 'results/source-review-ed9.json',
        'original-evaluation-qualification.json': ROOT / 'results/qualification-ed9.json',
        'original-training-source.tar': ROOT / 'results/source-f579.tar'})
    claim(out / 'complete.json', {'status': 'verified', 'finished': time(), 'deadline': s['deadline'],
          'no_hands_repeated': True, 'result_sha256': file_hash(out / 'result.json'),
          'archive_receipt_sha256': file_hash(run / 'archive-receipt.json')})


def launch(run, out, dest):
    with lock(ROOT / 'results/phase.lock'):
        original = read(run / 'state.json')
        if original['deadline'] - time() < 240: raise ValueError('No same-deadline closeout budget')
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], text=True).strip()
        # Added before moving source; this exact file was the only untracked file
        # at the original rejected report, recorded in the amendment receipt.
        if status != '?? bin/hu20-trainer': raise ValueError('Untracked content beyond the exact binary exception')
        s = snapshot(run, original['deadline']); claim(run / 'postprocessing-admission.json', s)
        if s['refused']: raise ValueError('Fresh closeout admission refusal')
        claim(run / 'postprocessing-claim.json', {'source': identity(), 'started': time(),
              'original_deadline': original['deadline'], 'destination': str(dest),
              'launcher_pid': os.getpid(), 'controller_sha256': file_hash(Path(__file__))})
        command = [sys.executable, '-m', 'scripts.verify_native_hu100_growth_closeout',
                   '--worker', '--run', str(run), '--out', str(out), '--destination', str(dest)]
        guard = supervise([{'name': 'verify-report-archive', 'command': command}], ROOT / 'results/postprocessing-guard',
            original['deadline'], swap_before=BASELINE, require_ac=True, rss_gib=10, disk_gib=15.5,
            swap_gib=.5, system_memory_guard=True)
        if guard['status'] != 'complete': raise ValueError('Postprocessing closeout failed; no retry')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--worker', action='store_true')
    for n in ('run', 'out', 'destination'): p.add_argument('--' + n, type=Path, required=True)
    a = p.parse_args(); (worker if a.worker else launch)(a.run, a.out, a.destination)
