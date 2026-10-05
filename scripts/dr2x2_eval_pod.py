"""One Linux evaluation worker, exact-plan parity gate and closed evidence seal."""
import argparse
import fcntl
import json
from importlib.metadata import distribution, version
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from scripts.hu20_platform_pilot import write
from scripts.mature_cpu_linux_worker import owned_rss, read_limit
from src.arena.schedule import digest
from src.diagnostics.saved_hu20 import file_hash


def update_control(path, fields):
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=json.loads(path.read_text()) if path.exists() else {}
        value.update(fields);write(path,value)


def same_reference(reference, actual):
    expected={(x['cell'],x['readout']):x for x in reference['tasks']}
    observed={(x['cell'],x['readout']):x for x in actual['tasks']}
    return (reference['status']==actual['status']=='complete'
            and reference['seed']==actual['seed']
            and reference['closed_hands']==actual['closed_hands']==160
            and len(reference['tasks'])==len(actual['tasks'])==4
            and reference['plan_sha256']==actual['plan_sha256']
            and expected.keys()==observed.keys()
            and all(expected[k]['scientific_fingerprints']==observed[k]['scientific_fingerprints']
                    and expected[k]['hands']==observed[k]['hands'] for k in expected))


def projected_playing_seconds(plan, actual):
    projected=0.
    for task in actual['tasks']:
        projected+=2*task['load_seconds']
        for timing in task['panels']:
            panel=next(p for p in plan['panels'] if p['name']==timing['panel'])
            blocks=panel['readout_blocks']+(panel['blocks'] if task['readout']=='current' else 0)
            projected+=2*timing['max_hand_seconds']*2*blocks
    return projected


def execute(a):
    if sys.platform!='linux' or platform.python_version()!='3.11.14':
        raise ValueError('Pinned Linux runtime required')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source:
        raise ValueError('Frozen evaluation source differs')
    origin=json.loads(distribution('pokers').read_text('direct_url.json'))
    if (origin['vcs_info']['commit_id']!='5db20e3d5d6862b32a7402035c1340b622d3b005'
            or version('numpy')!='1.26.4' or version('scipy')!='1.17.1'):
        raise ValueError('Pinned engine/numerical environment differs')
    plan=json.loads(a.plan.read_text());reference=json.loads(a.reference.read_text())
    if digest(plan)!=a.plan_sha or reference['plan_sha256']!=a.plan_sha:
        raise ValueError('Portable plan/reference differs')
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=False)
    memory=read_limit(('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'))
    if memory is None or not 14*10**9<=memory<=18*2**30:
        raise ValueError('Allocated16GB memory is unknown or differs')
    swap_path=Path('/sys/fs/cgroup/memory.swap.current')
    def swap():
        if swap_path.exists():return int(swap_path.read_text())
        both=read_limit(('/sys/fs/cgroup/memory/memory.memsw.usage_in_bytes',))
        resident=read_limit(('/sys/fs/cgroup/memory/memory.usage_in_bytes',))
        if both is None or resident is None:raise ValueError('Allocated swap usage unknown')
        return max(0,both-resident)
    before=swap();state={'pid':os.getpid(),'seed':a.seed,'source':a.source,'plan_sha256':a.plan_sha,
                         'status':'admission','started':time.time(),'memory_limit_bytes':memory,'peak_owned_rss_bytes':0,
                         'engine_origin':origin,'kernel':platform.uname()._asdict(),
                         'affinity_logical_cpus':sorted(os.sched_getaffinity(0))}
    for filename in ('/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.swap.max'):
        path=Path(filename)
        if path.exists():write(root/(path.name+'.json'),{'value':path.read_text()})
    child=None
    def phase(name, admission):
        nonlocal child
        cmd=[sys.executable,'-m','scripts.evaluate_dr2x2_ac','--plan',str(a.plan.resolve()),'--seed',str(a.seed),
             '--inputs',str(a.inputs.resolve()),'--out',str(root/name),
             *(['--admission'] if admission else ['--control',str(a.control.resolve())])]
        with (root/(name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(child_pid=child.pid,status=name);last_progress=None;last_change=time.time()
            while child.poll() is None:
                rss=owned_rss(child.pid);growth=swap()-before;lease=json.loads(a.control.read_text())
                state['peak_owned_rss_bytes']=max(state['peak_owned_rss_bytes'],rss)
                progress_path=root/name/'progress.json'
                if progress_path.exists():
                    progress=json.loads(progress_path.read_text());coordinate=tuple(progress.get(k) for k in ('cell','readout','stage','panel','closed_blocks'))
                    if coordinate!=last_progress:last_progress=coordinate;last_change=time.time()
                    state['progress']=progress
                state.update(heartbeat=time.time(),owned_rss_bytes=rss,swap_growth_bytes=growth,free_disk_bytes=shutil.disk_usage(root).free)
                write(root/'worker.json',state)
                with (root/'resources.jsonl').open('a') as f:f.write(json.dumps(state,sort_keys=True)+'\n')
                if (lease.get('stop') or time.time()>lease['lease_until'] or rss>=6*2**30
                        or growth>.5*2**30 or shutil.disk_usage(root).free<8*2**30
                        or time.time()-last_change>900):
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise RuntimeError('Evaluation resource/lease/stall stop; preserve partials')
                time.sleep(3)
            if child.returncode:raise RuntimeError(f'{name} failed exit{child.returncode}; no blind retry')
    try:
        phase('admission',True)
        actual=json.loads((root/'admission'/'summary.json').read_text())
        passed=same_reference(reference,actual)
        write(root/'linux-parity.json',{'passed':passed,'source':a.source,'plan_sha256':a.plan_sha,
              'reference_file_sha256':file_hash(a.reference),'actual_file_sha256':file_hash(root/'admission'/'summary.json'),
              'hands':actual['closed_hands'],'scientific_comparison':'exact fingerprints; latency removed; completion/fallback retained'})
        if not passed:raise ValueError('Linux scientific playing/reference divergence')
        projection=projected_playing_seconds(plan,actual)
        admitted=projection<=10800
        write(root/'linux-capacity.json',{'passed':admitted,'projected_playing_seconds':projection,
              'admission_allowance_seconds':10800,'fixed_margin':2,
              'note':'capacity gate before outcomes, not a healthy-run wall cutoff; budget watchdog remains authoritative'})
        if not admitted:raise ValueError('Linux capacity exceeds prospectively quoted allowance')
        update_control(a.control,{'admitted_plan_sha256':a.plan_sha,'linux_reference_match':True})
        phase('evaluation',False)
        actual=json.loads((root/'evaluation'/'summary.json').read_text())
        if actual['closed_hands']!=55296:raise ValueError('Frozen per-seed work incomplete')
        state['status']='complete'
    except Exception as exc:
        state.update(status='failed-retained',failure=f'{type(exc).__name__}: {exc}')
    finally:
        state.update(finished=time.time(),child_pid=None)
        write(root/'worker.json',state)
        write(root/'manifest.json',{'files':{str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':file_hash(p)}
              for p in root.rglob('*') if p.is_file() and p.name not in ('manifest.json','finished.json')}})
        write(root/'finished.json',state)
    return state['status']!='complete'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','inputs','reference','root','control'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--source',required=True);p.add_argument('--plan-sha',required=True)
    raise SystemExit(execute(p.parse_args()))
