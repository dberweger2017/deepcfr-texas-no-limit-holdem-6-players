import sys,json
from pathlib import Path
from hashlib import sha256
from src.blueprint.solver import BlueprintTrainer,PilotConfig,HU20_UNCAPPED_GAME
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.blueprint.artifact import save_training,export_policy
from src.game.hand import Table
p=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=False)
t=BlueprintTrainer(Table(('player-0','player-1'),(2000,2000)),PilotConfig(seed=2026093001,raise_cap=None,max_nodes=250000,max_entries=4000000,max_seconds=300,abstraction=HU20_UNCAPPED_SCHEMA,game=HU20_UNCAPPED_GAME))
work=[]
for _ in range(32):
 r=t.step();work.append((r.iteration,r.nodes,r.terminals,r.entries,r.updated_keys,r.attempted_work))
d={'iteration':t.iteration,'entries':len(t.nodes),'work_sha256':sha256(json.dumps(work,sort_keys=True).encode()).hexdigest(),'checkpoint_sha256':save_training(t,p/'checkpoint.gz'),'export_sha256':export_policy(t,p/'current.gz')}
(p/'result.json').write_text(json.dumps(d,indent=2)+'\n')
