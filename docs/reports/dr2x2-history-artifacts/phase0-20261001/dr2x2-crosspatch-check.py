import argparse,gzip,json,subprocess,sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from random import Random
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.blueprint.solver import BlueprintTrainer,PilotConfig,HU20_UNCAPPED_GAME,_seed
from src.blueprint.artifact import save_training,export_policy,load_training
from src.game.hand import Table
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--schema',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--resume',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
config=PilotConfig(seed=a.seed,raise_cap=None,max_nodes=250000,max_entries=4000000,max_seconds=300,abstraction=a.schema,game=HU20_UNCAPPED_GAME)
t=load_training(a.resume) if a.resume else BlueprintTrainer(Table(('player-0','player-1'),(2000,2000)),config)
assert t.config==config
completed=0;work=sha256();rows=[]
while completed<100000 if not a.resume else not rows:
 r=t.step();completed+=r.nodes;d=asdict(r)
 for k in ('elapsed_seconds','replay_seconds','worker_rss_sum_bytes'):d.pop(k)
 work.update(json.dumps(d,sort_keys=True,separators=(',',':')).encode()+b'\n');rows.append(d)
cp=a.out/'checkpoint.gz';exp=a.out/'current.gz';ch=save_training(t,cp);eh=export_policy(t,exp)
rng=[]
for seat in (0,1):
 for stream in ('deal','action'):
  s=_seed(a.seed,t.iteration+1,seat,0,stream);r=Random(s);rng.append([seat,stream,s,r.getstate(),[r.random() for _ in range(8)],r.getstate()])
result={'python':sys.version,'schema':a.schema,'seed':a.seed,'iteration':t.iteration,'completed_nodes':completed,'entries':len(t.nodes),'work_sha256':work.hexdigest(),'rng_sha256':sha256(json.dumps(rng,sort_keys=True).encode()).hexdigest(),'checkpoint_sha256':ch,'export_sha256':eh,'checkpoint_payload_sha256':sha256(gzip.decompress(cp.read_bytes())).hexdigest(),'export_payload_sha256':sha256(gzip.decompress(exp.read_bytes())).hexdigest()}
(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
