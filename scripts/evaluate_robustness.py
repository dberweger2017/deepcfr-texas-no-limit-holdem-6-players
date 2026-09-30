"""Sequential saved-policy stress/LBR evaluator; no training or variant selection."""
import argparse
import gzip
import json
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from random import Random
from time import perf_counter,time

from scripts.evaluate_hu20 import rss,system,write_json
from scripts.tp20_common import interruptible,seal
from src.arena.catalog import Checkpoint
from src.arena.schedule import digest,stream_seed
from src.blueprint.abstraction import choices,information_key,HU20_SCHEMA,TP20_SCHEMA
from src.blueprint.artifact import FrozenBlueprint,HU20_FORMAT,TP20_FORMAT
from src.blueprint.windowed import _hash
from src.diagnostics.robustness import LocalBestResponse,LBRConfig,ReactiveAttack
from src.game.hand import Hand,Table
from src.game.observation import ActionTaken
from src.game.types import ActionKind


class Uniform:
    def distribution(self,v):
        menu=choices(v,free_fold=False)
        return menu,(1/len(menu),)*len(menu),False


def load(spec):
    if spec['name']=='uniform': return Uniform()
    path=Path(spec['path'])
    return FrozenBlueprint(Checkpoint(spec['name'],str(path),spec['sha256'],
        spec.get('format', HU20_FORMAT if spec['players']==2 else TP20_FORMAT)),path)


def guard(plan,out,deadline):
    if time()>=deadline: raise TimeoutError('Absolute campaign deadline')
    if rss()>plan['limits']['max_rss_gib']*1024**3: raise MemoryError('RSS guard')
    if shutil.disk_usage(out).free<plan['limits']['min_free_gib']*1024**3: raise RuntimeError('Free disk guard')


def play(source,spec,rules,contract,block,rotation,root,phase,config,emit,resource_only=False,opponent_policies=None):
    n=spec['players']; logical=tuple((seat-rotation)%n for seat in range(n))
    ids=tuple(f'player-{i}' for i in logical)
    deal=stream_seed(root,'validation' if resource_only else 'test','deal',n,block)
    rngs=[Random(stream_seed(root,'test','action',n,block,i)) for i in range(n)]
    table=Table(ids,(2000,)*n,button=block%n)
    hand=Hand.start(table,hand_id=f'robustness-{phase}-{n}-{block}',seed=deal)
    declared_rules=rules
    if len(rules)==2 and block%2: rules=rules[::-1]
    rivals=opponent_policies if opponent_policies is not None else {i:(LocalBestResponse(source,stream_seed(root,'test','opponent',n,block,i),config)
               if rules[i-1]=='lbr' else ReactiveAttack(rules[i-1],contract)) for i in range(1,n)}
    off_menu=False; original_off_menu=False; trace=[];timings=[];decisions=Counter();keys=Counter()
    try:
        for index in range(1000):
            if hand.finished:break
            view=hand.observe(hand.actor);who=logical[view.seat]
            original_menu=choices(view,free_fold=False)
            target_menu=choices(view,raise_cap=getattr(source,'raise_cap',2),free_fold=False)
            menu=original_menu;trained=None
            begin=perf_counter()
            if who==0:
                menu,probabilities,trained=source.distribution(view)
                action=rngs[who].choices(menu,weights=probabilities,k=1)[0].action
                key=information_key(view,menu,schema=spec.get('abstraction',HU20_SCHEMA if n==2 else TP20_SCHEMA))
                keys[(view.street.value,key)]+=1
                decisions[(view.street.value,'trained' if trained else 'fallback','offmenu-history' if off_menu else 'menu-history')]+=1
            else:
                action=rivals[who].choose_action(view)
            view.legal_actions.validate(action)
            onmenu=any(c.action==action for c in (target_menu if spec.get("dual_menu_telemetry") else menu))
            elapsed=perf_counter()-begin
            row={'index':index,'seat':view.seat,'logical_player':who,'street':view.street.value,
                 'kind':action.kind.value,'raise_to':action.raise_to,'on_training_menu':onmenu,
                 'street_raises':sum(isinstance(e,ActionTaken) and e.street==view.street and e.action.kind==ActionKind.RAISE for e in view.history),
                 'preceding_off_menu':off_menu,'target_trained':trained,'seconds':elapsed}
            if spec.get('dual_menu_telemetry'):
                row.update(on_original_cap2_menu=any(c.action==action for c in original_menu),
                           on_target_menu=any(c.action==action for c in target_menu),
                           preceding_original_off_menu=original_off_menu,
                           preceding_target_off_menu=off_menu,
                           target_abstraction=spec['abstraction'],
                           target_key=key if who==0 else None,
                           target_visits=(getattr(source,'visits',{}).get(key,0) if who==0 and hasattr(source,'visits') else None))
            if who==0 and hasattr(source,'last_translation'):
                row['translation']=source.last_translation
            if who and isinstance(rivals[who],LocalBestResponse):
                row['lbr']=rivals[who].telemetry[-1]
            trace.append(row);timings.append(elapsed);off_menu|=not onmenu
            original_off_menu |= not any(c.action==action for c in original_menu)
            hand=hand.apply(action)
        if not hand.finished:raise RuntimeError('Decision cap')
        net=tuple(hand.events[-1].stacks[s]-2000 for s in range(n))
        if sum(net)!=0: raise ValueError('Chip conservation')
        result={'status':'complete','net_chips_by_seat':None if resource_only else net,'target_chips':None if resource_only else net[rotation]}
    except Exception as exc:
        result={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
    row={'policy':spec['name'],'players':n,'rules':declared_rules,'actual_rival_order':rules,'contract':contract,'block':block,'rotation':rotation,
         'deal_seed':deal,'root_seed':root,'button':block%n,'phase':phase,'actions':trace,
         'event_digest':digest([repr(e) for e in hand.events]),'decision_telemetry':[{'coordinates':k,'count':v} for k,v in sorted(decisions.items())],
         'reached_keys':[{'street':s,'key':k,'decisions':v} for (s,k),v in sorted(keys.items())],**result}
    emit(row)
    if result['status']!='complete':raise RuntimeError(result['error'])
    return sum(timings)


def run(plan,out,phase,deadline):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);interruptible();start=time()
    write_json(out/'plan.json',plan)
    write_json(out/'manifest.json',{'plan_digest':digest(plan),'phase':phase,'started':start,'deadline':deadline,
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'swap_before':system(['sysctl','vm.swapusage'])})
    total=0;attempts=[];status='complete';failure=None
    f=gzip.open(out/'hands.jsonl.gz','wt')
    def emit(row):
        nonlocal total
        f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush();total+=1
    try:
        specs=(plan['preflight_policies'] if phase=='preflight' else plan['policies'])
        tasks=[('preflight',s) for s in specs] if phase=='preflight' else (
            [('stress',s) for s in specs]+[('lbr',s) for s in specs if s['players']==2 and s['name'] in plan['lbr_targets']])
        for task,spec in tasks:
            guard(plan,out,deadline);source=load(spec)
            panels=[]
            if phase=='preflight':
                panels=[(('pressure',),'menu',8,0), (('pressure',),'native',8,0),
                        (('lbr',),'menu',4,2),(('lbr',),'menu',4,8)] if spec['players']==2 else [(('pressure','passive'),'native',8,0)]
            else:
                lineups=plan['hu_lineups'] if spec['players']==2 else plan['tp_lineups']
                for i,lineup in enumerate(lineups):
                    for contract in ('menu','native'):
                        panels.append((tuple(lineup),contract,plan['hu_blocks'] if spec['players']==2 else plan['tp_blocks'],0))
                if spec['name'] in plan['lbr_targets'] and spec['players']==2:
                    panels.append((('lbr',),'menu',plan['lbr_blocks'],plan['chance_samples']))
                panels=[panel for panel in panels if bool(panel[3])==(task=='lbr')]
            for panel_index,(rules,contract,count,samples) in enumerate(panels):
                # Same target schedules across checkpoints/seeds/uniform. Contracts
                # also paired; lineup streams separate; calibration disjoint.
                root=plan['preflight_root'] if phase=='preflight' else plan['stress_root']
                if samples:root=plan['calibration_root']+samples if phase=='preflight' else plan['lbr_root']
                elif phase!='preflight':root+= (plan['hu_lineups'] if spec['players']==2 else plan['tp_lineups']).index(list(rules))
                attempt={'policy':spec['name'],'rules':rules,'contract':contract,'samples':samples,'blocks_requested':count,'started':time(),'completed_blocks':0}
                attempts.append(attempt);write_json(out/'attempts.json',attempts)
                for block in range(count):
                    guard(plan,out,deadline)
                    for rotation in range(spec['players']):
                        play(source,spec,rules,contract,block,rotation,root,phase,
                             LBRConfig(max(1,samples),plan['lbr_seconds']),emit,phase=='preflight')
                    attempt['completed_blocks']+=1
                    if block%128==0:
                        write_json(out/'progress.json',{'hands':total,'attempt':attempt,'elapsed':time()-start,'peak_rss_bytes':rss()})
                attempt.update(status='complete',seconds=time()-attempt['started']);write_json(out/'attempts.json',attempts)
            del source
    except Exception as exc:
        status='incomplete';failure=f'{type(exc).__name__}: {exc}'
    finally:
        f.close();write_json(out/'attempts.json',attempts)
        result={'status':status,'failure':failure,'hands':total,'seconds':time()-start,'peak_rss_bytes':rss(),
                'swap_after':system(['sysctl','vm.swapusage'])}
        write_json(out/'result.json',result);seal(out)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=('preflight','confirmation'),required=True);p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args();result=run(json.loads(args.plan.read_text()),args.out,args.phase,args.deadline)
    print(json.dumps(result));return result['status']!='complete'

if __name__=='__main__':raise SystemExit(main())
