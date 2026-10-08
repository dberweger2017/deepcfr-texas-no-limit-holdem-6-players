"""One bounded sequential evaluation, using the existing external resource guard."""

import argparse
import json
from math import floor, isfinite
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import time

from scripts.hu20_scaling_supervise import run as supervise
from scripts.native_hu_followup_limits import memory_snapshot, unsafe_memory
from scripts.run_tp20_campaign import swap_bytes
from src.arena.schedule import digest
from src.policies.files import file_hash

BASELINE = 'vm.swapusage: total = 1024.00M  used = 448.81M  free = 575.19M  (encrypted)'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.update-tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def quote(pilot, replay, reproduction, remaining, proposed=2048):
    """Outcome-blind 2x quote; counts freeze before any final hand exists."""
    play = sum(p['seconds'] for p in pilot['panel_costs'])
    repeat = sum(p['seconds'] for p in reproduction['panel_costs'])
    fixed = pilot['wall_seconds'] - play + reproduction['wall_seconds'] - repeat
    per_block = (play + repeat + replay['seconds']) / pilot['blocks_per_opponent']
    if not all(isfinite(x) and x > 0 for x in (fixed, per_block, remaining)):
        raise ValueError('Invalid measured quote')
    reserve = 180
    available = remaining - reserve - 2 * fixed
    count = min(proposed, max(0, 32 * floor(available / (2 * per_block) / 32)))
    return {'blocks_per_opponent': count, 'final_hands': count * 20,
            'measured_fixed_seconds': fixed, 'measured_play_replay_reproduction_seconds_per_block': per_block,
            'allowance': 2, 'closeout_reserve_seconds': reserve,
            'predicted_seconds': 2 * (fixed + count * per_block),
            'remaining_seconds_at_quote': remaining, 'outcomes_inspected': False}


def admission(out, deadline):
    if time() >= deadline or unsafe_memory(memory_snapshot(), admission=True, rss_gib=10):
        raise ValueError('Deadline/system headroom admission failure')
    swap = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True)
    power = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
    if (swap_bytes(swap) - swap_bytes(BASELINE) > .5 * 1024**3
            or shutil.disk_usage(out).free < 15.5 * 1024**3 or 'AC Power' not in power):
        raise ValueError('Swap/disk/AC admission failure')


def worker(config, out, source):
    state = read(out / 'state.json')
    if state['phase'] != 'claimed' or state['source'] != source or state['config_sha256'] != file_hash(config):
        raise ValueError('Already claimed/changed worker inputs; never retry')
    state.update(phase='pilot', worker_pid=os.getpid()); write(out / 'state.json', state)
    settings = read(config)
    deadline = state['deadline']; py = str(Path(sys.executable).absolute())

    def command(label, arguments):
        admission(out, deadline)
        state['phase'] = label; write(out / 'state.json', state)
        with (out / (label + '.log')).open('xb') as log:
            subprocess.run([py, '-m', *arguments], stdout=log, stderr=subprocess.STDOUT, check=True)

    def stages(prefix, blocks, root):
        args = ['--config', str(config), '--source', source, '--blocks', str(blocks), '--root', str(root)]
        command(prefix + '-play', ['scripts.evaluate_native_hu100_baseline', *args, '--out', str(out / prefix)])
        command(prefix + '-audit', ['scripts.audit_native_hu100_baseline', '--run', str(out / prefix),
                                  '--out', str(out / (prefix + '-audit.json'))])
        command(prefix + '-reproduce', ['scripts.evaluate_native_hu100_baseline', *args,
                                       '--out', str(out / (prefix + '-reproduction')),
                                       '--reproduce', str(out / prefix)])

    try:
        stages('pilot', settings['pilot_blocks_per_opponent'], settings['pilot_root'])
        inputs = [out / 'pilot/complete.json', out / 'pilot-audit.json', out / 'pilot-reproduction/complete.json']
        pilot, replay, reproduction = map(read, inputs)
        if pilot['status'] != 'complete' or replay['status'] != 'verified' or not reproduction['reproduced_all_hands_and_decisions']:
            raise ValueError('Pilot completeness gate failed')
        frozen = {**quote(pilot, replay, reproduction, deadline - time(), settings['proposed_final_blocks_per_opponent']),
                  'source': source, 'config_sha256': file_hash(config), 'final_root': settings['final_root'],
                  'pilot_receipts': {str(p.relative_to(out)): file_hash(p) for p in inputs},
                  'frozen_at': time(), 'deadline': deadline}
        with (out / 'frozen-final.json').open('x') as f:
            f.write(json.dumps(frozen, indent=2, sort_keys=True) + '\n')
        state['frozen_final_sha256'] = file_hash(out / 'frozen-final.json'); write(out / 'state.json', state)
        if frozen['blocks_per_opponent'] < 32:
            state.update(phase='complete-no-final-budget', terminal=True)
        else:
            admission(out, deadline)
            if (file_hash(out / 'frozen-final.json') != state['frozen_final_sha256']
                    or deadline - time() < frozen['predicted_seconds'] + frozen['closeout_reserve_seconds']):
                raise ValueError('Fresh final time-fit admission failed')
            stages('final', frozen['blocks_per_opponent'], settings['final_root'])
            state.update(phase='complete', terminal=True, final_hands=frozen['final_hands'], completed_at=time())
        write(out / 'state.json', state)
    except BaseException as error:
        state.update(phase='failed', terminal=True, failure=repr(error), stopped_at=time())
        write(out / 'state.json', state)
        raise


def launch(config, out, qualification, review):
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if (read(qualification).get('status') != 'verified' or read(qualification).get('source') != source
            or read(review).get('status') != 'passed' or read(review).get('reviewed_source') != source):
        raise ValueError('Exact-source qualification and independent review required')
    if read(config)['execution_cap_seconds'] != 1800:
        raise ValueError('Owner 30-minute cap differs')
    processes = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,%cpu=,rss=,comm='], text=True)
    for line in processes.splitlines():
        p = line.split(None, 4)
        if (len(p) == 5 and int(p[0]) != os.getpid() and int(p[3]) > 512 * 1024
                and any(k in p[4].lower() for k in ('python', 'hu20-trainer', 'cargo', 'rustc', 'node', 'java'))):
            raise ValueError('Competing heavy worker; never stop another job: ' + line)
    out.mkdir(parents=True, exist_ok=False)
    started = time(); deadline = started + 1800
    state = {'phase': 'claimed', 'source': source, 'config_sha256': file_hash(config),
             'started_at': started, 'deadline': deadline, 'terminal': False,
             'qualification_sha256': file_hash(qualification), 'review_sha256': file_hash(review),
             'swap_baseline': BASELINE, 'original_campaign_unchanged': True}
    write(out / 'state.json', state)
    command = [str(Path(sys.executable).absolute()), '-m', 'scripts.run_native_hu100_baseline',
               '--worker', '--config', str(config), '--out', str(out), '--source', source]
    guard = supervise([{'name': 'hu100-playing-baseline', 'command': command}], out / 'guard', deadline,
                      swap_before=BASELINE, require_ac=True, rss_gib=10, disk_gib=15.5, swap_gib=.5,
                      system_memory_guard=True)
    state = read(out / 'state.json')
    if guard['status'] != 'complete':
        state.update(phase='failed', terminal=True, guard_status=guard['status'], guard_failure=guard['failure'])
        write(out / 'state.json', state)
    write(out / 'closeout.json', {'guard_status': guard['status'], 'state': state,
                                'guard_sha256': file_hash(out / 'guard/campaign.json'), 'closed_at': time()})
    return guard['status'] == 'complete'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--qualification', type=Path)
    p.add_argument('--review', type=Path); p.add_argument('--worker', action='store_true'); p.add_argument('--source')
    a = p.parse_args()
    if a.worker:
        worker(a.config.absolute(), a.out.absolute(), a.source)
    else:
        if not a.qualification or not a.review:
            p.error('--qualification and --review required')
        sys.exit(not launch(a.config.absolute(), a.out.absolute(), a.qualification, a.review))
