"""Owner-authorized fixed M1 reference; no playing evaluation or extra seeds."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

SOURCE = 'e18f0079a14addc90938acca8c30795e8af09691'
ARCHIVE_SHA = 'ca096bc89593f1f9dee4106389fb698a5fedb2bc9352c89d769d60db7018cb39'
CLONE = Path('/Users/dberweger/Local/hu20-platform-parity-pr129')
ARCHIVE = Path('/Users/dberweger/Local/runpod-hu20-parity-artifacts/hu20-linux-pilot.tar')
ROOT = CLONE/'results/m1-platform-pilot'
STATE = CLONE/'reference-status.json'
os.chdir(CLONE)
sys.path.insert(0,str(CLONE))
from scripts.run_exact_ranker_experiment import owned_rss, swap

def write(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    temporary.replace(path)

ROOT.mkdir(parents=True,exist_ok=False)
started=time.time()
clock={'started':started,'deadline':started+3600,'swap_start_mib':swap()}
state={'status':'running','pid':os.getpid(),'source':SOURCE,'clock':clock,'phases':[],
       'owner_approved_host':'M1','pinned_engine_reused_without_rebuild':True}
write(STATE,state)
samples=[]
child=None

def guard(pid=0):
    memory,_=owned_rss(pid)
    sw=swap()
    free=shutil.disk_usage(ROOT).free
    ac='AC Power' in subprocess.check_output(['pmset','-g','batt'],text=True)
    row={'time':time.time(),'owned_rss_bytes':memory,'swap_mib':sw,
         'swap_growth_mib':max(0,sw-clock['swap_start_mib']),'free_disk_bytes':free,'ac':ac,'child_pid':pid}
    samples.append(row)
    if row['time']>=clock['deadline'] or memory>10.5*2**30 or row['swap_growth_mib']>512 or free<8*2**30 or not ac:
        raise RuntimeError('M1 time/RSS/swap/disk/AC guard reached')

try:
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=SOURCE:
        raise ValueError('Reference source changed')
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()!=ARCHIVE_SHA:
        raise ValueError('Linux archive transport checksum mismatch')
    commands=[
      ('extract-linux',['tar','-xf',str(ARCHIVE),'-C',str(ROOT)]),
      ('focused-tests',[sys.executable,'-m','pytest','-q','tests/test_hu20_platform_pilot.py']),
      ('direct',[sys.executable,'-m','scripts.hu20_platform_pilot','run','--plan','configs/blueprint/runpod-hu20-parity.json','--out',str(ROOT/'direct')]),
      ('resumed',[sys.executable,'-m','scripts.hu20_platform_pilot','run','--plan','configs/blueprint/runpod-hu20-parity.json','--out',str(ROOT/'resumed'),'--resume',str(ROOT/'direct')]),
      ('resume-comparison',[sys.executable,'-m','scripts.hu20_platform_pilot','compare','--left',str(ROOT/'direct'),'--right',str(ROOT/'resumed'),'--out',str(ROOT/'resume-comparison.json')]),
      ('platform-comparison',[sys.executable,'-m','scripts.hu20_platform_pilot','compare','--left',str(ROOT/'direct'),'--right',str(ROOT/'results/platform-pilot/direct'),'--out',str(ROOT/'platform-comparison.json')]),
    ]
    for phase, command in commands:
        guard()
        attempt={'name':phase,'command':command,'started':time.time()}
        state['phases'].append(attempt);write(STATE,state)
        with (ROOT/f'{phase}.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            attempt['pid']=child.pid;write(STATE,state)
            while child.poll() is None:
                guard(child.pid);time.sleep(1)
            attempt.update(exit_code=child.returncode,finished=time.time());write(STATE,state)
            child=None
            if attempt['exit_code']:
                raise RuntimeError(f'{phase} failed; retained, no retry')
    guard()
    inventory=json.loads((ROOT/'results/linux-inventory.json').read_text())
    for name,record in inventory.items():
        p=ROOT/'results'/name
        if p.stat().st_size!=record['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'Linux member hash mismatch: {name}')
    direct=[json.loads(x) for x in (ROOT/'direct/iterations.jsonl').read_text().splitlines()]
    linux=[json.loads(x) for x in (ROOT/'results/platform-pilot/direct/iterations.jsonl').read_text().splitlines()]
    resumed=[json.loads(x) for x in (ROOT/'resumed/iterations.jsonl').read_text().splitlines()]
    midpoint=json.loads((ROOT/'direct/midpoint.json').read_text())
    a=json.loads((ROOT/'direct/result.json').read_text());b=json.loads((ROOT/'results/platform-pilot/direct/result.json').read_text())
    checks={'linux_members_verified':len(inventory),'all_non_timing_iterations_equal':direct==linux,
            'resume_suffix_equal':resumed==direct[midpoint['iteration']:],
            'next_rng_streams_equal':a['next_streams']==b['next_streams'],
            'completed_counts_equal':all(a[k]==b[k] for k in ('completed_nodes','iteration','entries','next_nodes','next_iteration'))}
    checks['transport']={}
    for name in ('final.json.gz','current.json.gz','next.json.gz'):
        aa=(ROOT/'direct'/name).read_bytes();bb=(ROOT/'results/platform-pilot/direct'/name).read_bytes()
        offsets=[i for i,(x,y) in enumerate(zip(aa,bb)) if x!=y]
        checks['transport'][name]={'differing_byte_offsets':offsets,'sizes_equal':len(aa)==len(bb),
            'm1_gzip_os':aa[9],'linux_gzip_os':bb[9],'deflate_and_crc_equal':aa[10:]==bb[10:]}
    write(ROOT/'independent-checks.json',checks)
    if not all(checks[k] for k in ('all_non_timing_iterations_equal','resume_suffix_equal','next_rng_streams_equal','completed_counts_equal')):
        raise ValueError('Meaningful work/RNG/resume divergence')
    state.update(status='complete',finished=time.time(),elapsed_seconds=time.time()-started)
except Exception as exc:
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        child.wait(timeout=10)
    state.update(status='failed',failure=f'{type(exc).__name__}: {exc}',finished=time.time())
    raise
finally:
    write(ROOT/'resources.json',samples);write(STATE,state)
    records={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(ROOT.rglob('*')) if p.is_file()}
    write(ROOT/'final-manifest.json',{'source':SOURCE,'clock':clock,'status':state['status'],'files':records})
