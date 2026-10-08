"""Fill only an unstarted last panel under the unchanged original campaign cap.

The terminal launcher receipt remains immutable. This cannot retry partial play,
repeat a passed stage, replace evidence, change counts, or start a new clock.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import time

from scripts.hu20_scaling_supervise import run as supervise
from scripts.run_native_hu100_learning_curves import (
    AUTHORIZED_ROOT, BASELINE, CONFIG, admission, durable_claim, write,
)
from src.arena.schedule import canonical
from src.policies.files import file_hash

SOURCE = '51597e88706465c33dc51577890989aa303fefca'
ROOT = AUTHORIZED_ROOT
RUN = ROOT / 'results/hu100-learning-curves/run-01'
CONFIG_HASH = '0bf1c2c8ebbedfb5c79a518315fbdadf4555e6377a2118e3f1977354a1991b30'
FREEZE_HASH = 'c95cfdb870dd3bfdbb7ba2bd9ddf31dd7e2e8e075925dd6e60af728dd25fec11'


def read(path):
    return json.loads(path.read_text())


def validate():
    from scripts.run_native_hu100_learning_curves import worker_identity, authorized_config
    worker_identity()
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != SOURCE
            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
            or authorized_config(ROOT / CONFIG) != CONFIG_HASH
            or file_hash(RUN / 'frozen-final.json') != FREEZE_HASH):
        raise ValueError('Original source/config/count freeze changed')
    state = read(RUN / 'state.json')
    freeze = read(RUN / 'frozen-final.json')
    guard = read(RUN / 'guard/campaign.json')
    if (state['phase'] != 'failed' or not state['terminal']
            or state['failure'] != "ValueError('Deadline/system headroom admission failure')"
            or guard['status'] != 'incomplete' or guard['failure'] is not None
            or guard['attempts'][0]['guard_failure'] is not None
            or freeze['blocks_per_opponent'] != 2048 or freeze['final_hands'] != 122880
            or freeze['deadline'] != state['deadline'] or freeze['source'] != SOURCE):
        raise ValueError('Only the recorded between-stage admission refusal is continuable')
    # A live prior job, partial fifth panel, or partially created log always blocks.
    listing = subprocess.check_output(['ps', '-axo', 'pid=,ppid='], text=True)
    live = {int(line.split()[0]) for line in listing.splitlines() if line.strip()}
    if state['worker_pid'] in live or guard['identity']['pid'] in live:
        raise ValueError('Original job still alive')
    for nodes in (100691, 1001382, 5001210, 10001922):
        if (read(RUN / f'final/{nodes}/complete.json')['status'] != 'complete'
                or read(RUN / f'final-{nodes}-audit.json')['status'] != 'verified'
                or not read(RUN / f'final-reproduction/{nodes}/complete.json')['reproduced_all_hands_and_decisions']):
            raise ValueError('Earlier stage incomplete; never repeat it')
    for path in (RUN / 'final/11042440', RUN / 'final-reproduction/11042440',
                 RUN / 'summary.json', *RUN.glob('final-11042440-*.log'),
                 RUN / 'final-11042440-audit.json'):
        if path.exists():
            raise ValueError('Last panel was already attempted; never retry it')
    admission(ROOT, state['deadline'])
    return state, freeze


def worker():
    state, freeze = validate()
    claim = read(RUN / 'continuation-claim.json')
    if claim['launcher_pid'] != os.getppid() or claim['script_sha256'] != file_hash(Path(__file__)):
        raise ValueError('Worker must belong to this one continuation launcher')
    durable_claim({'pid': os.getpid(), 'parent': os.getppid()}, RUN / 'continuation-worker-claim.json')
    settings = read(ROOT / CONFIG)
    model = settings['models'][-1]
    config = RUN / 'final-11042440-config.json'
    expected = {k: v for k, v in settings.items() if k != 'models'}
    expected['model'] = model
    if canonical(read(config)) != canonical(expected):
        raise ValueError('Last panel config changed')
    args = ['--config', str(config), '--source', SOURCE, '--blocks', '2048',
            '--root', str(freeze['final_root'])]

    def command(label, arguments):
        if (file_hash(ROOT / CONFIG) != CONFIG_HASH or file_hash(RUN / 'frozen-final.json') != FREEZE_HASH):
            raise ValueError('Frozen inputs changed')
        admission(ROOT, state['deadline'])
        with (RUN / (label + '.log')).open('xb') as log:
            subprocess.run([sys.executable, '-m', *arguments], stdout=log, stderr=subprocess.STDOUT, check=True)

    command('final-11042440-play', ['scripts.evaluate_native_hu100_baseline', *args,
            '--out', str(RUN / 'final/11042440'), '--reference-run', str(RUN / 'final/100691')])
    command('final-11042440-audit', ['scripts.audit_native_hu100_baseline', '--run',
            str(RUN / 'final/11042440'), '--out', str(RUN / 'final-11042440-audit.json')])
    command('final-11042440-reproduce', ['scripts.evaluate_native_hu100_baseline', *args,
            '--out', str(RUN / 'final-reproduction/11042440'), '--reproduce', str(RUN / 'final/11042440'),
            '--reference-run', str(RUN / 'final-reproduction/100691')])
    command('final-cross-checkpoint-audit', ['scripts.report_native_hu100_learning_curves',
            '--run', str(RUN), '--config', str(ROOT / CONFIG), '--out', str(RUN / 'summary.json')])
    write(RUN / 'continuation-complete.json', {'status': 'complete', 'completed_at': time(),
          'source': SOURCE, 'script_sha256': file_hash(Path(__file__)), 'original_deadline': state['deadline'],
          'original_state_sha256': file_hash(RUN / 'state.json'), 'frozen_final_sha256': FREEZE_HASH,
          'earlier_stages_repeated': False, 'new_execution_cap': False})


def launch(review):
    state, freeze = validate()
    script_hash = file_hash(Path(__file__))
    if read(review).get('status') != 'passed' or read(review).get('continuation_script_sha256') != script_hash:
        raise ValueError('Independent continuation source review required')
    processes = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,%cpu=,rss=,comm='], text=True)
    for line in processes.splitlines():
        p = line.split(None, 4)
        if (len(p) == 5 and int(p[0]) != os.getpid() and (int(p[3]) > 512 * 1024 or float(p[2]) > 50)
                and any(k in p[4].lower() for k in ('python', 'hu20-trainer', 'cargo', 'rustc', 'node', 'java'))):
            raise ValueError('Competing heavy worker; never stop another job: ' + line)
    from scripts.run_native_hu100_learning_curves import worker_identity
    durable_claim({'launcher_pid': os.getpid(), 'script_sha256': script_hash,
                   'started_at': time(), 'original_deadline': state['deadline'], 'outcomes_inspected': False,
                   'review_sha256': file_hash(review), 'chip': worker_identity()}, RUN / 'continuation-claim.json')
    record = supervise([{'name': 'unstarted-last-panel', 'command': [sys.executable,
              str(Path(__file__).absolute()), '--worker']}], RUN / 'continuation-guard', state['deadline'],
              swap_before=BASELINE, require_ac=True, rss_gib=10, disk_gib=15.5, swap_gib=.5, system_memory_guard=True)
    write(RUN / 'continuation-closeout.json', {'status': record['status'], 'closed_at': time(),
          'original_deadline': state['deadline'], 'guard_sha256': file_hash(RUN / 'continuation-guard/campaign.json')})
    return record['status'] == 'complete'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--review', type=Path); p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if args.worker:
        worker()
    else:
        if not args.review: p.error('--review required')
        sys.exit(not launch(args.review))
