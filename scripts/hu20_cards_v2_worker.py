"""One isolated Linux pod: resource guards, exact recovery gate, frozen campaign."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from scripts.hu20_platform_pilot import compare,write
from scripts.mature_cpu_linux_worker import guard_reason,owned_rss,read_limit,stop_child
from src.diagnostics.saved_hu20 import file_hash


def execute(plan,seed,out,deadline,reference):
    if sys.platform!='linux' or platform.python_version()!=plan['python']:
        raise ValueError('Frozen Linux Python required')
    if file_hash(Path('src/blueprint/cards_v2.py'))!=plan['descriptor_sha256']:
        raise ValueError('Descriptor hash differs')
    import importlib.metadata
    origin=json.loads(importlib.metadata.distribution('pokers').read_text('direct_url.json') or '{}')
    if origin.get('vcs_info',{}).get('commit_id')!=plan['engine_revision']:
        raise ValueError('Frozen native engine revision differs')
    memory=read_limit(('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'))
    if memory is None or not 60*10**9<=memory<=70*2**30:
        raise ValueError('Cannot verify approved 64GB container memory')
    def swap():
        p=Path('/sys/fs/cgroup/memory.swap.current')
        if p.exists():return int(p.read_text())
        combined=read_limit(('/sys/fs/cgroup/memory/memory.memsw.usage_in_bytes',))
        resident=read_limit(('/sys/fs/cgroup/memory/memory.usage_in_bytes',))
        if combined is None or resident is None:raise ValueError('Cannot verify container swap')
        return max(0,combined-resident)
    swap_before=swap();out.mkdir(parents=True,exist_ok=False);state={'status':'running','started':time.time(),
        'deadline':deadline,'seed':seed,'memory_limit_bytes':memory,'attempts':[],
        'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'plan_sha256':sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    (out/'lscpu.txt').write_text(subprocess.check_output(['lscpu'],text=True))
    for name in ('cpu.max','memory.max','memory.swap.max'):
        p=Path('/sys/fs/cgroup')/name
        if p.exists():(out/(name+'.txt')).write_text(p.read_text())
    limits={**plan['limits'],'max_rss_fraction_of_container_memory':plan['limits']['max_container_fraction']}
    def command(name,args):
        attempt={'name':name,'started':time.time(),'status':'running'};state['attempts'].append(attempt);write(out/'worker.json',state)
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.Popen([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            reason=None
            try:
                while child.poll() is None:
                    sample={'time':time.time(),'phase':name,'owned_rss_bytes':owned_rss(child.pid),
                            'swap_growth_bytes':swap()-swap_before,'free_disk_bytes':shutil.disk_usage(out).free}
                    with (out/'resources.jsonl').open('a') as stream:stream.write(json.dumps(sample,sort_keys=True)+'\n')
                    reason=guard_reason(sample['owned_rss_bytes'],memory,sample['swap_growth_bytes'],sample['free_disk_bytes'],sample['time'],deadline,limits)
                    if reason:stop_child(child);break
                    time.sleep(1)
            except BaseException:
                stop_child(child);raise
            attempt.update(finished=time.time(),exit_code=child.wait(),guard_failure=reason,
                           status='complete' if child.returncode==0 and reason is None else 'failed')
        write(out/'worker.json',state)
        if attempt['status']!='complete':raise RuntimeError('Owned phase failed: '+name)
    root=Path('configs/diagnostics')
    try:
        command('tests',['-m','pytest','-q','tests/test_hu20_cards_v2.py','tests/test_hu20_card_v2_campaign.py',
                 'tests/test_blueprint_hu20.py','tests/test_blueprint_native_reopening.py'])
        for name,resume in [('recovery-direct',None),('recovery-resumed',out/'recovery-direct')]:
            args=['-m','scripts.hu20_platform_pilot','run','--plan',str(root/'hu20-card-v2-recovery.json'),'--out',str(out/name)]
            if resume:args+=['--resume',str(resume)]
            command(name,args)
        for left,name in [(out/'recovery-direct','linux-resume'),(reference,'m1-linux')]:
            r=compare(left,out/'recovery-resumed',out/(name+'.json'))
            if not r['equal']:raise ValueError('Full-state parity failed: '+name)
        command('training',['-m','scripts.train_hu20_cards_v2','--plan',str(root/'hu20-card-v2-run.json'),
                 '--seed',str(seed),'--out',str(out/'training'),'--deadline',str(deadline-1800)])
        command('evaluation',['-m','scripts.evaluate_hu20_cards_v2','--plan',str(root/'hu20-card-v2-run.json'),
                 '--seed',str(seed),'--training',str(out/'training'),'--baseline-plan',str(root/'hu20-stackoff-v1.json'),
                 '--out',str(out/'evaluation'),'--deadline',str(deadline)])
        state['status']='complete'
    except Exception as exc:state.update(status='failed',failure=f'{type(exc).__name__}: {exc}')
    finally:
        state['finished']=time.time();write(out/'worker.json',state)
        write(out/'manifest.json',{str(p.relative_to(out)):{'sha256':file_hash(p),'bytes':p.stat().st_size}
              for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'})
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','out','reference'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    r=execute(json.loads(a.plan.read_text()),a.seed,a.out,a.deadline,a.reference)
    print(json.dumps({'status':r['status'],'seed':a.seed}),flush=True);raise SystemExit(r['status']!='complete')
