"""Resource supervisor for exactly one Linux trainer; no provider secrets."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

from scripts.hu20_platform_pilot import write
from scripts.mature_cpu_linux_worker import owned_rss, read_limit


def execute(a):
    if sys.platform != 'linux' or platform.python_version() != '3.11.14':
        raise ValueError('Pinned Linux Python required')
    plan = json.loads(a.plan.read_text())
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    limits = plan['engineering']
    memory = read_limit(('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'))
    if memory is None or not .9*plan['cells'][json.loads(a.parent.read_text())['cell']]['ram_gb']*10**9 <= memory <= 1.1*plan['cells'][json.loads(a.parent.read_text())['cell']]['ram_gb']*2**30:
        raise ValueError('Actual allocated memory is unknown')
    swap_path = Path('/sys/fs/cgroup/memory.swap.current')
    def swap():
        if swap_path.exists():
            return int(swap_path.read_text())
        combined = read_limit(('/sys/fs/cgroup/memory/memory.memsw.usage_in_bytes',))
        resident = read_limit(('/sys/fs/cgroup/memory/memory.usage_in_bytes',))
        if combined is None or resident is None:
            raise ValueError('Actual swap usage is unknown')
        return max(0, combined-resident)
    before_swap = swap()
    state = dict(status='preflight', started=time.time(), pid=os.getpid(), memory_limit_bytes=memory,
                 swap_before_bytes=before_swap, peak_owned_rss_bytes=0, attempts=[],
                 kernel=platform.uname()._asdict(), affinity_logical_cpus=sorted(os.sched_getaffinity(0)))
    for filename in ('/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
                     '/sys/fs/cgroup/cpu/cpu.cfs_period_us', '/sys/fs/cgroup/memory.max',
                     '/sys/fs/cgroup/memory.swap.max'):
        path = Path(filename)
        if path.exists():
            (root/(path.name+'.txt')).write_text(path.read_text())
    for name, cmd in (('lscpu', ['lscpu']), ('topology', ['lscpu', '-e=CPU,CORE,SOCKET,ONLINE']),
                      ('packages', [sys.executable, '-m', 'pip', 'freeze'])):
        (root / (name+'.txt')).write_text(subprocess.check_output(cmd, text=True))
    write(root / 'worker.json', state)
    from hashlib import sha256
    if sha256(Path('src/blueprint/cards_v2.py').read_bytes()).hexdigest()!=plan['descriptor_sha256']:
        raise ValueError('Frozen descriptor differs')
    child = None

    def phase(name, args):
        nonlocal child
        attempt = dict(name=name, started=time.time(), status='running')
        state['attempts'].append(attempt)
        with (root / (name+'.log')).open('w') as log:
            child = subprocess.Popen([sys.executable, '-m', 'scripts.dr2x2_worker', *args],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            attempt['pid'] = child.pid
            reason, signaled, last_change, last_nodes = None, None, time.time(), None
            try:
                while child.poll() is None:
                    current = owned_rss(child.pid)
                    free = shutil.disk_usage(root).free
                    swap_growth = swap()-before_swap
                    state['peak_owned_rss_bytes'] = max(state['peak_owned_rss_bytes'], current)
                    state.update(heartbeat=time.time(), current_owned_rss_bytes=current,
                                 swap_growth_bytes=swap_growth, free_disk_bytes=free)
                    with (root/'resources.jsonl').open('a') as record:
                        record.write(json.dumps(dict(time=time.time(), phase=name, owned_rss_bytes=current,
                            swap_growth_bytes=swap_growth, free_disk_bytes=free,
                            cgroup_memory_bytes=read_limit(('/sys/fs/cgroup/memory.current',
                                '/sys/fs/cgroup/memory/memory.usage_in_bytes')),
                            progress=state.get('progress'))) + '\n')
                    if name == 'training':
                        progress = root / 'training' / 'progress.json'
                        if progress.exists():
                            info = json.loads(progress.read_text())
                            if info['completed_nodes'] != last_nodes:
                                last_nodes, last_change = info['completed_nodes'], time.time()
                            state['progress'] = info
                        control = json.loads(a.control.read_text())
                        if control.get('stop') or time.time() >= control['lease_until']:
                            reason = control.get('stop') or 'Controller lease expired'
                        elif time.time()-last_change > limits['stall_seconds']:
                            reason = 'No complete-node progress within stall allowance'
                    if current >= min(plan['cells'][json.loads(a.parent.read_text())['cell']]['ram_gb']*.8*10**9, memory*limits['container_rss_fraction']):
                        reason = 'Owned RSS / actual allocation headroom'
                    if free < limits['min_free_disk_gib']*2**30:
                        reason = 'Pod free disk guard'
                    if swap_growth > limits['max_swap_growth_gib']*2**30:
                        reason = 'Pod swap growth guard'
                    if reason and signaled is None:
                        attempt['stop_reason'] = reason
                        os.killpg(child.pid, signal.SIGTERM)
                        signaled = time.time()
                    if signaled and time.time()-signaled > 300:
                        os.killpg(child.pid, signal.SIGKILL)
                    write(root / 'worker.json', state)
                    time.sleep(2)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=300)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            attempt.update(finished=time.time(), exit=child.returncode,
                           status='complete' if child.returncode == 0 else 'failed')
            write(root / 'worker.json', state)
            if child.returncode:
                raise RuntimeError(name+' failed; retained attempt and last complete state')

    common = ['--plan', str(a.plan.resolve()), '--parent', str(a.parent.resolve())]
    try:
        if not a.resume:
            phase('preflight-direct', ['preflight', *common, '--out', str(root/'preflight-direct')])
            phase('preflight-resume', ['preflight', *common, '--out', str(root/'preflight-resume'),
                                      '--resume', str(root/'preflight-direct')])
            direct = json.loads((root/'preflight-direct/result.json').read_text())
            resumed = json.loads((root/'preflight-resume/result.json').read_text())
            reference = json.loads(a.reference.read_text())
            checks = {}
            checks['same_frozen_source'] = direct['source']['source'] == resumed['source']['source'] == reference['source']['source']
            checks['pinned_engine'] = all(json.loads(row['source']['engine_origin'])['vcs_info']['commit_id'] ==
                                        plan['runtime']['engine'] for row in (direct, resumed, reference))
            for key in ('added_nodes', 'iteration', 'next_streams', 'suffix_work_sha256', 'suffix_trace'):
                checks[key] = direct[key] == resumed[key] == reference[key]
            checks['checkpoint_bytes'] = direct['final'] == resumed['final'] == reference['final']
            checks['policy_payload'] = (direct['current']['uncompressed_sha256'] ==
                resumed['current']['uncompressed_sha256'] == reference['current']['uncompressed_sha256'])
            checks['only_known_gzip_os_byte'] = (direct['current']['os_normalized_sha256'] ==
                resumed['current']['os_normalized_sha256'] == reference['current']['os_normalized_sha256'])
            checks['linux_policy_bytes'] = direct['current'] == resumed['current']
            write(root/'preflight-parity.json', dict(checks=checks, passed=all(checks.values()),
                  gzip_os_header_note='Linux 3 / Darwin 19; policy payload must match exactly'))
            if not all(checks.values()):
                raise ValueError('Linux / M1 fixed-work fixed-work or fresh-process resume divergence')
        else:
            # A material environment change must use a fresh preflight first.
            if not a.reference.exists() or not json.loads(a.reference.read_text()).get('resume_environment_validated'):
                raise ValueError('Recovery requires explicit environment-parity certificate')
        state['status'] = 'training'
        phase('training', ['train', *common, '--out', str(root/'training'), '--control', str(a.control.resolve()),
                          *(['--resume', str(a.resume.resolve())] if a.resume else [])])
        saved = json.loads((root/'training/saved.jsonl').read_text().splitlines()[-1])
        if saved['requested_total_nodes'] != plan['target_total_nodes']:
            raise ValueError('Final requested checkpoint missing')
        for suffix in ('a', 'b'):
            phase('final-reload-'+suffix, ['verify-final', *common, '--out', str(root/('final-reload-'+suffix)),
                                  '--resume', str(root/'training'/(saved['id']+'.record.json'))])
        if json.loads((root/'final-reload-a/result.json').read_text()) != json.loads((root/'final-reload-b/result.json').read_text()):
            raise ValueError('Final independent fresh-process next-step mismatch')
        state['status'] = 'complete'
    except Exception as exc:
        state.update(status='incident', failure=f'{type(exc).__name__}: {exc}')
    finally:
        state['finished'] = time.time()
        write(root/'worker.json', state)
        from src.blueprint.windowed import _hash
        write(root/'manifest.json',dict(files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=_hash(p)) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('manifest.json','finished.json')}))
        write(root/'finished.json', state)
    return state['status'] != 'complete'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('plan', 'parent', 'root', 'control', 'reference'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--resume', type=Path)
    raise SystemExit(execute(p.parse_args()))
