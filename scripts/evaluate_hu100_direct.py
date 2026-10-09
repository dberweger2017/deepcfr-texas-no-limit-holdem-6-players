"""Policy-vs-policy HU100 duplicate matches with streaming replay and reproduction."""
import argparse
from dataclasses import asdict
import gzip
import json
from math import isclose, isfinite, sqrt
from pathlib import Path
from statistics import mean, stdev
import subprocess
from time import perf_counter

from scipy.stats import t
from src.arena.catalog import Checkpoint
from src.arena.registry import PolicyRegistry
from src.arena.runner import public_events, run_schedule
from src.arena.schedule import Plan, Scenario, build_schedule, canonical, digest, schedule_document, stream_seed
from src.blueprint.abstraction import HU100_SCHEMA, choices, information_key
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.policies.files import file_hash


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def action_seed(root,block,rotation,role):
    return stream_seed(root,'test','action','hu100-direct-v1',block.scenario,block.index,rotation,role)


def make_plan(specs,blocks,root,rung):
    models=tuple(Checkpoint(**{k:s[k] for k in ('name','path','sha256','format')}) for s in specs)
    return Plan((Scenario('direct-'+rung,(10000,10000)),),candidate=models[0].name,
        baseline=models[1].name,opponents=(models[1].name,),blocks=blocks,root_seed=root,
        split='test',models=models,max_decisions=1000)


def validate(registry,specs,plan):
    if registry.plan.models!=plan.models:raise ValueError('Registry identity differs')
    for spec in specs:
        model=registry.models[spec['name']]
        if (file_hash(model.source_path)!=spec['sha256'] or model.source_path.stat().st_size!=spec['bytes']
                or model.description['entries']!=spec['entries'] or model.description['iteration']!=spec['iteration']
                or model.description['source_checkpoint_sha256']!=spec['source_checkpoint_sha256']
                or model.identity['stacks']!=[10000,10000] or model.identity['small_blind']!=50
                or model.identity['big_blind']!=100 or model.identity['chip_unit']!='0.01'):
            raise ValueError('Pinned direct model/game differs')
        model.configure_translation(None)


class Probe:
    """Trace the sampler without giving its policy any evaluator context."""
    def __init__(self,player,model,context,emit):
        self.player,self.model,self.context,self.emit=player,model,context,emit
    def choose_action(self,view):
        action=self.player.choose_action(view)
        menu,probabilities,known=self.model.distribution(view)
        key=information_key(view,menu,schema=HU100_SCHEMA)
        if not any(c.action==action and p>0 for c,p in zip(menu,probabilities)):
            raise ValueError('Direct action outside positive-probability native menu')
        visits=int(self.model.visits.get(key,0))
        self.emit({**self.context,'hand_id':view.hand_id,'seat':view.seat,'street':view.street.value,
            'action':asdict(action),'key':key,'menu':[asdict(c) for c in menu],'probabilities':list(probabilities),
            'lookup':'missing-key' if not known else 'zero-mass' if key in self.model.zero_mass else 'positive-mass-known-key',
            'visits':visits})
        return action


def execute(specs,blocks,root,rung,out,source,registry=None,reproduce=None):
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=source:
        raise ValueError('Source differs')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Clean committed source required')
    out.mkdir(parents=True,exist_ok=False)
    plan=make_plan(specs,blocks,root,rung)
    started=perf_counter();tick=started
    if registry is None:registry=PolicyRegistry(plan)
    registry.plan=plan
    validate(registry,specs,plan)
    load_seconds=perf_counter()-tick
    schedule=build_schedule(plan)
    put(out/'inputs.json',{'source':source,'models':specs,'plan':asdict(plan),'translation':None,
        'schedule':schedule_document(plan),'plan_sha256':digest(asdict(plan))})
    contexts={}
    private_seeds=set()
    def seed_factory(block,rotation,arm,role):
        seed=action_seed(root,block,rotation,role)
        if seed in private_seeds:raise ValueError('Private stream collision')
        private_seeds.add(seed)
        contexts[seed]={'block':block.index,'rotation':rotation,'role':role,'seed':seed}
        return seed
    hands=decisions=0
    repeat_h=gzip.open(reproduce/'hands.jsonl.gz','rt') if reproduce else None
    repeat_d=gzip.open(reproduce/'decisions.jsonl.gz','rt') if reproduce else None
    tick=perf_counter()
    try:
        with gzip.open(out/'hands.jsonl.gz','wt') as hand_stream,gzip.open(out/'decisions.jsonl.gz','wt') as traces:
            def emit_decision(row):
                nonlocal decisions
                text=canonical(row)
                if repeat_d and json.loads(next(repeat_d))!=row:raise ValueError('Reproduced direct decision differs')
                traces.write(text+'\n');decisions+=1
            def factory(name,seed):
                model=registry.models[name]
                return Probe(registry.make_policy(name,seed),model,{**contexts.pop(seed),'policy':name},emit_decision)
            def emit(row,timing):
                nonlocal hands
                if row['status']!='completed':
                    hand_stream.write(canonical(row)+'\n');hand_stream.flush()
                    raise ValueError('Failed direct hand: '+str(row['error']))
                if repeat_h and json.loads(next(repeat_h))!=row:raise ValueError('Reproduced direct hand differs')
                hand_stream.write(canonical(row)+'\n');hands+=1
            if not run_schedule(plan,schedule,emit,factory=factory,
                    seed_factory=seed_factory,arms=('candidate',)):
                raise ValueError('Direct runner stopped')
        if reproduce and (next(repeat_h,None) is not None or next(repeat_d,None) is not None):
            raise ValueError('Extra retained reproduction rows')
    finally:
        if repeat_h:repeat_h.close()
        if repeat_d:repeat_d.close()
    play_seconds=perf_counter()-tick
    pins={p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in out.glob('*.jsonl.gz')}
    put(out/'complete.json',{'status':'complete','hands':hands,'decisions':decisions,'files':pins,
        'seconds':perf_counter()-started,'load_or_validation_seconds':load_seconds,'play_seconds':play_seconds,
        'reproduced_all_hands_and_decisions':bool(reproduce)})
    return registry,json.loads((out/'complete.json').read_text())


def audit(out,output,report=False):
    tick=perf_counter()
    inputs=json.loads((out/'inputs.json').read_text());done=json.loads((out/'complete.json').read_text())
    plan=Plan.from_dict(inputs['plan'])
    if done['status']!='complete' or inputs['translation'] is not None or inputs['plan_sha256']!=digest(inputs['plan']):
        raise ValueError('Foreign direct plan')
    for name,pin in done['files'].items():
        if (out/name).stat().st_size!=pin['bytes'] or file_hash(out/name)!=pin['sha256']:
            raise ValueError('Raw direct bytes changed')
    schedule={b.index:b for b in build_schedule(plan)}
    expected={(b,r) for b in range(plan.blocks) for r in (0,1)}
    seen=set();values={};hands=actions=0;coverage={}
    with gzip.open(out/'hands.jsonl.gz','rt') as rows,gzip.open(out/'decisions.jsonl.gz','rt') as traces:
        for line in rows:
            row=json.loads(line);coord=(row['block'],row['rotation'])
            if coord not in expected or coord in seen or row['arm']!='candidate' or row['status']!='completed':
                raise ValueError('Duplicate/foreign/failed direct hand')
            if row['outcome_sha256']!=digest({k:v for k,v in row.items() if k!='outcome_sha256'}):
                raise ValueError('Direct outcome changed')
            seen.add(coord);block=schedule[row['block']];rotation=row['rotation']
            ids=tuple(f'player-{(seat-rotation)%2}' for seat in (0,1))
            hand=Hand.start(Table(ids,(10000,10000),block.button,50,100,'0.01'),
                hand_id=f'table-0/{block.index}/{rotation}/0',seed=block.deal_seeds[0])
            for event in row['events']:
                if event['event']!='ActionTaken':continue
                if hand.finished or hand.actor!=event['seat']:raise ValueError('Direct replay actor differs')
                view=hand.observe(hand.actor);d=json.loads(next(traces));role=int(view.player_id!='player-0')
                action=Action(ActionKind(event['action']['kind']),event['action']['raise_to'])
                view.legal_actions.validate(action)
                menu=choices(view,raise_cap=None,free_fold=False)
                if (d['block']!=block.index or d['rotation']!=rotation or d['role']!=role
                        or d['seed']!=action_seed(plan.root_seed,block,rotation,role)
                        or d['hand_id']!=view.hand_id or d['seat']!=view.seat or d['street']!=view.street.value
                        or d['action']!=event['action'] or d['key']!=information_key(view,menu,schema=HU100_SCHEMA)
                        or d['policy']!=(plan.candidate if role==0 else plan.baseline)
                        or d['menu']!=json.loads(canonical([asdict(c) for c in menu]))
                        or len(d['probabilities'])!=len(menu) or not all(isfinite(p) and p>=0 for p in d['probabilities'])
                        or not isclose(sum(d['probabilities']),1,abs_tol=1e-8)
                        or not any(c.action==action and p>0 for c,p in zip(menu,d['probabilities']))):
                    raise ValueError('Direct private stream/menu/probability/key differs')
                if d['lookup'] not in ('missing-key','zero-mass','positive-mass-known-key') or d['visits']<0:
                    raise ValueError('Unknown direct coverage classification')
                if d['lookup']!='positive-mass-known-key' and d['probabilities']!=[1/len(menu)]*len(menu):
                    raise ValueError('Direct fallback differs')
                band='0' if d['visits']==0 else '1-9' if d['visits']<10 else '10-99' if d['visits']<100 else '100+'
                cell=coverage.setdefault(d['policy']+'/'+d['street'],{'decisions':0,'lookup':{},'visit_bands':{}})
                cell['decisions']+=1
                for key,value in (('lookup',d['lookup']),('visit_bands',band)):
                    cell[key][value]=cell[key].get(value,0)+1
                hand=hand.apply(action);actions+=1
            net=[p.stack-10000 for p in hand.observe(0).players]
            if (not hand.finished or public_events(hand.events)!=row['events'] or net!=row['net_chips']
                    or sum(net)!=0 or net[rotation]!=row['candidate_chips'] or row['participants']!=list(ids)):
                raise ValueError('Direct events/settlement differs')
            values[coord]=net[rotation];hands+=1
        if next(traces,None) is not None or seen!=expected or hands!=done['hands'] or actions!=done['decisions']:
            raise ValueError('Incomplete direct replay')
    result={'status':'verified','hands_replayed':hands,'actions_replayed':actions,
        'seconds':perf_counter()-tick,'settlement_sum_zero':True,'all_native_menu_actions':True,'coverage':coverage}
    if report:
        series=[(values[b,0]+values[b,1])/2 for b in range(plan.blocks)]
        center=mean(series);sd=stdev(series) if len(series)>1 else 0
        margin=float(t.ppf(.975,len(series)-1))*sd/sqrt(len(series))
        bonf=float(t.ppf(1-.05/6,len(series)-1))*sd/sqrt(len(series))
        result.update(bb_per_100=center,ci95=[center-margin,center+margin],half_width=margin,
            bonferroni_three_ci=[center-bonf,center+bonf],blocks=plan.blocks,
            decision='improves' if center-margin>0 else 'declines' if center+margin<0 else 'inconclusive')
    put(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--specs',type=Path,required=True);p.add_argument('--blocks',type=int,required=True)
    p.add_argument('--root',type=int,required=True);p.add_argument('--rung',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',required=True)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args();specs=json.loads(a.specs.read_text());tick=perf_counter()
    registry,primary=execute(specs,a.blocks,a.root,a.rung,a.out/'play',a.source)
    checked=audit(a.out/'play',a.out/'replay.json',report=not a.pilot)
    _,repeat=execute(specs,a.blocks,a.root,a.rung,a.out/'reproduction',a.source,registry,a.out/'play')
    put(a.out/'costs.json',{'seconds':perf_counter()-tick,'primary':primary,'repeat':repeat,
        'audit_seconds':checked['seconds'],'blocks':a.blocks,'outcomes_inspected_for_quote':False})

if __name__=='__main__':main()
