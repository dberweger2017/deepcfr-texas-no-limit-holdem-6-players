import json,os,subprocess,time
from pathlib import Path
BASE=Path('/Users/dberweger/.codex/worktrees/hu20-history-2x2/deepcfr-texas-no-limit-holdem-6-players')
ROOT=BASE/'results/dr2x2-history-preflight-m1-20261001'
PY='/Users/dberweger/Local/hu20-platform-parity-pr129/.venv/bin/python'
PLAN='configs/blueprint/dr2x2-history-preflight.json'
SOURCE=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip()
assert SOURCE.startswith('da42ac5')
state={'pid':os.getpid(),'source':SOURCE,'status':'running','started':time.time(),'completed':[],'active':None}
def save():
 p=ROOT/'supervisor.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
def phase(args,log):
 with log.open('xb') as out:
  child=subprocess.Popen([PY,'-m','scripts.preflight_hu20_history',*args],cwd=BASE,stdout=out,stderr=subprocess.STDOUT)
  state['child_pid']=child.pid;save();return child.wait()
try:
 save()
 for seed in [2026093001,2026093002,2026093003]:
  for cell in ['full','compressed']:
   name=f'{cell}-{seed}';out=ROOT/name;assert not out.exists()
   state['active']=name;save()
   code=phase(['worker','--plan',PLAN,'--seed',str(seed),'--cell',cell,'--probe',str(ROOT/'corpus'),'--out',str(out)],ROOT/f'{name}.log')
   result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {'status':'no-result'}
   state['completed'].append({'task':name,'exit_code':code,'status':result['status']});save()
   if code or result['status']!='complete':raise RuntimeError(f'Preserved failed worker {name}; no retry')
 state['active']='density-report';save()
 code=phase(['report','--plan',PLAN,'--out',str(ROOT)],ROOT/'density-report.log')
 if code:raise RuntimeError('Report failure retained')
 state['status']='complete';state['density_gate']=json.loads((ROOT/'density-gate.json').read_text())['status']
except Exception as exc:state.update(status='blocked',failure=f'{type(exc).__name__}: {exc}')
finally:state.update(active=None,child_pid=None,finished=time.time());save()
