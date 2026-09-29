"""Common-attack comparison: target menus differ, fixed adversary contracts do not."""

import argparse
import gzip
import json
from pathlib import Path
from random import Random
from time import time

from scripts.evaluate_robustness import Uniform, load, play
from scripts.tp20_common import append, guard, interruptible, seal
from scripts.train_hu20 import rss, system, write_json
from src.arena.policies import make_policy
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import HU20_SCHEMA, HU20_UNCAPPED_SCHEMA
from src.blueprint.artifact import HU20_FORMAT, HU20_UNCAPPED_FORMAT, load_training
from src.blueprint.windowed import _hash
from src.blueprint.solver import regret_match
from src.diagnostics.robustness import LBRConfig

ATTACKS=(('Pressure-original-cap2','pressure','menu'),('Pressure-native','pressure','native'),
         ('Minraise-original-cap2','minraise','menu'),('Minraise-native','minraise','native'),
         ('Passive','passive','menu'))
SECONDARY=('loose_passive','loose_aggressive','tight_passive','tight_aggressive','pot_pressure','hu20_uniform')


class Target:
    def __init__(self,spec):
        self.source=load(spec);self.raise_cap=self.source.raise_cap
        if spec.get('checkpoint_path'):
            if _hash(Path(spec['checkpoint_path']))!=spec['checkpoint_sha256']:raise ValueError('Checkpoint hash')
            trainer=load_training(Path(spec['checkpoint_path']))
            if trainer.config.game!=self.source.game or trainer.iteration!=self.source.description['iteration']:
                raise ValueError('Checkpoint/inference identity mismatch')
            if self.source.description['strategy']!='current' or set(trainer.nodes)!=set(self.source.entries):
                raise ValueError('Final-current export lineage')
            for key,node in trainer.nodes.items():
                if self.source.entries[key]!=(node.names,regret_match(tuple(node.regrets))):
                    raise ValueError('Checkpoint/current-policy probabilities differ')
            self.visits={k:n.visits for k,n in trainer.nodes.items()}
    def distribution(self,v):return self.source.distribution(v)


class UniformPlayer:
    def __init__(self,seed):self.random=Random(seed)
    def choose_action(self,v):
        menu,p,_=Uniform().distribution(v)
        return self.random.choices(menu,weights=p,k=1)[0].action


def specifications(plan,root):
    specs=[]
    for seed in plan['training_seeds']:
        for arm in ('A','B'):
            run=root/'training'/f'{arm}-{seed}';r=json.loads((run/'result.json').read_text())
            if r['status']!='complete' or r['initial_entries']!=0:raise ValueError('Unfinished/nonfresh training arm')
            for m in r['milestones']:
                i=m['index'];specs.append({'name':f'{arm}-{seed}-{i}','seed':seed,'arm':arm,'milestone':i,'players':2,
                    'path':str((run/f'current-{i}.json.gz').resolve()),'sha256':m['policy_sha256'],
                    'checkpoint_path':str((run/f'checkpoint-{i}.json.gz').resolve()),'checkpoint_sha256':m['checkpoint_sha256'],
                    'format':HU20_FORMAT if arm=='A' else HU20_UNCAPPED_FORMAT,
                    'abstraction':HU20_SCHEMA if arm=='A' else HU20_UNCAPPED_SCHEMA,'dual_menu_telemetry':True})
    return specs+plan['reference_policies']


def run(plan,root,out,deadline):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);interruptible();started=time();specs=specifications(plan,root)
    write_json(out/'models.json',specs);write_json(out/'plan.json',plan)
    attempts=[];hands=0;result={'status':'incomplete','failure':None,'started':started}
    with gzip.open(out/'hands.jsonl.gz','wt') as saved:
        def emit(row,spec,attacker):
            nonlocal hands
            row.update(arm=spec['arm'],training_seed=spec.get('seed'),milestone=spec['milestone'],attacker=attacker)
            saved.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');saved.flush();hands+=1
        try:
            tasks=[('cheap',s) for s in specs]+[('lbr',s) for s in specs if s['arm'] in ('A','B') and s['milestone']==3]+[('secondary',s) for s in specs if s['arm'] in ('A','B') and s['milestone']==3]
            for phase,spec in tasks:
                guard(plan,out,deadline);source=Target(spec)
                if phase=='cheap':panels=[(label,rule,contract,plan['cheap_blocks'],plan['stress_root']+plan['attack_root_offsets'][i],None) for i,(label,rule,contract) in enumerate(ATTACKS)]
                elif phase=='lbr':panels=[('LBR-original-cap2','lbr','menu',plan['lbr_blocks'],plan['lbr_root'],None)]
                else:panels=[(name,name,'secondary',plan['secondary_panel_blocks'],plan['secondary_root']+i,name) for i,name in enumerate(SECONDARY)]
                for label,rule,contract,blocks,rootseed,secondary in panels:
                    attempt={'policy':spec['name'],'attacker':label,'phase':phase,'requested_blocks':blocks,'completed_blocks':0,'status':'running','started':time()}
                    attempts.append(attempt);write_json(out/'attempts.json',attempts)
                    for b in range(blocks):
                        guard(plan,out,deadline)
                        for rot in (0,1):
                            rivals=None
                            if secondary:
                                actionseed=stream_seed(rootseed,'test','action',2,b,1)
                                rivals={1:UniformPlayer(actionseed) if secondary=='hu20_uniform' else make_policy(secondary,actionseed)}
                            play(source,spec,(rule,),contract,b,rot,rootseed,phase,LBRConfig(plan['chance_samples'],plan['lbr_seconds']),
                                 lambda row:emit(row,spec,label),opponent_policies=rivals)
                        attempt['completed_blocks']+=1
                        if b%64==0:write_json(out/'progress.json',{'hands':hands,'attempt':attempt,'elapsed':time()-started,'rss_bytes':rss()})
                    attempt.update(status='complete',seconds=time()-attempt['started']);write_json(out/'attempts.json',attempts)
                del source
            result['status']='complete'
        except Exception as exc:
            result['failure']=f'{type(exc).__name__}: {exc}'
            if attempts and attempts[-1]['status']=='running':attempts[-1].update(status='failed',failure=result['failure'])
    result.update(hands=hands,elapsed_seconds=time()-started,peak_rss_bytes=rss(),swap_after=system(['sysctl','vm.swapusage']))
    write_json(out/'attempts.json',attempts);write_json(out/'result.json',result);seal(out);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();r=run(json.loads(a.plan.read_text()),a.root,a.out,a.deadline)
    print(json.dumps(r,sort_keys=True));return r['status']!='complete'

if __name__=='__main__':raise SystemExit(main())
