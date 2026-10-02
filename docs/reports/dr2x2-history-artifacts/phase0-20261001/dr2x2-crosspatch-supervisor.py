import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/Users/dberweger/.codex/worktrees/hu20-history-2x2/deepcfr-texas-no-limit-holdem-6-players/results/dr2x2-history-preflight-m1-20261001')
BASE=Path('/Users/dberweger/.codex/worktrees/hu20-card-v2/deepcfr-texas-no-limit-holdem-6-players')
PYTHONS={'3.11.14':'/Users/dberweger/Local/hu20-platform-parity-pr129/.venv/bin/python','3.11.15':'/Users/dberweger/Local/deepcfr-texas-no-limit-holdem-6-players/.venv/bin/python'}
OUT=ROOT/'crosspatch';OUT.mkdir(exist_ok=False)
files=['src/blueprint/abstraction.py','src/blueprint/solver.py','src/blueprint/artifact.py','src/blueprint/cards_v2.py']
source={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in files}
state={'pid':os.getpid(),'status':'waiting-for-density-worker-exit','source_files':source,'comparisons':[]}
def write(): (OUT/'summary.json').write_text(json.dumps(state,indent=2)+'\n')
write()
try:
 deadline=time.monotonic()+300
 while json.loads((ROOT/'supervisor.json').read_text())['status']=='running':
  if time.monotonic()>deadline:raise TimeoutError('Queue wait expired; no second worker launched')
  time.sleep(2)
 assert all(hashlib.sha256((BASE/p).read_bytes()).hexdigest()==h for p,h in source.items())
 state['status']='running';write()
 env=os.environ.copy();env['PYTHONPATH']=str(BASE)
 for schema in ['hu20-native-reopening-ordered-history-card-v1','hu20-native-reopening-ordered-history-card-v2']:
  for seed in [2026093001,2026093002,2026093003]:
   results={}
   for version,python in PYTHONS.items():
    for phase in ['direct','fresh-next']:
     dest=OUT/f'{schema}-{seed}-{version}-{phase}'
     args=[python,'/tmp/dr2x2-crosspatch-check.py','--out',str(dest),'--schema',schema,'--seed',str(seed)]
     if phase=='fresh-next':args+=['--resume',str(OUT/f'{schema}-{seed}-{version}-direct'/'checkpoint.gz')]
     with (OUT/f'{dest.name}.log').open('xb') as log:subprocess.run(args,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
     results[version,phase]=json.loads((dest/'result.json').read_text())
   for phase in ['direct','fresh-next']:
    a={k:v for k,v in results['3.11.14',phase].items() if k!='python'};b={k:v for k,v in results['3.11.15',phase].items() if k!='python'}
    if a!=b:raise ValueError(f'Crosspatch state/bytes differ {schema}/{seed}/{phase}: {[k for k in a if a[k]!=b[k]]}')
   state['comparisons'].append({'schema':schema,'seed':seed,'direct':results['3.11.14','direct'],'fresh-next':results['3.11.14','fresh-next'],'passed':True});write()
 assert all(hashlib.sha256((BASE/p).read_bytes()).hexdigest()==h for p,h in source.items())
 state['status']='complete'
except Exception as exc:state.update(status='failed',failure=f'{type(exc).__name__}: {exc}')
finally:state['finished']=time.time();write()
