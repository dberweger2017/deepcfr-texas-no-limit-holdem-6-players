"""HU100/HU200 duplicate matches with streaming replay and reproduction."""
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
from src.arena.policies import make_policy
from src.arena.heuristics import STYLES
from src.arena.runner import public_events, run_schedule
from src.arena.schedule import Plan, Scenario, build_schedule, canonical, digest, stream_seed
from src.blueprint.abstraction import HU100_SCHEMA, HU200_SCHEMA, STACK_BY_SCHEMA, EQUITY_SCHEMAS, schema_choices, information_key
from src.blueprint.equity_buckets import EquityCards
from src.blueprint.average import missing_check_call
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.policies.files import file_hash


PANEL_OPPONENTS = ('random', 'check_call', 'tight_aggressive', 'loose_aggressive',
                   'pot_pressure', 'half_pot_pressure', 'matched_pot_pressure')


def local_cache(path):
    """Resolve an explicit cache outside the project's synced research storage."""
    if path is None:return None
    resolved=Path(path).expanduser().resolve()
    synced=Path.home()/'Local/Research-Cloud'
    if resolved.is_relative_to(synced.resolve()) or any(
            'GoogleDrive' in part or part in ('CloudStorage','Mobile Documents') for part in resolved.parts):
        raise ValueError('Policy cache must be outside synced storage')
    return resolved


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def action_seed(root,block,rotation,role):
    return stream_seed(root,'test','action','hu100-direct-v1',block.scenario,block.index,rotation,role)


def make_plan(specs,blocks,root,rung, *, stack_bb=100, opponent=None):
    if stack_bb not in (100,200):raise ValueError("Direct depth must be 100 or 200 BB")
    if len(specs)!=2 or specs[0]["name"]==specs[1]["name"]:
        raise ValueError("Pin two distinctly named averages")
    if opponent is not None and opponent not in PANEL_OPPONENTS:
        raise ValueError("Unknown scripted opponent")
    models=tuple(Checkpoint(**{k:s[k] for k in ('name','path','sha256','format','abstraction','card_buckets') if k in s}) for s in specs)
    return Plan((Scenario('direct-'+rung if opponent is None else opponent,(stack_bb*100,)*2),),candidate=models[0].name,
        baseline=models[1].name,opponents=(opponent or models[1].name,),blocks=blocks,root_seed=root,
        split='test',models=models,max_decisions=1000)


def validate(registry,specs,plan):
    stacks=plan.scenarios[0].stacks
    if registry.plan.models!=plan.models:raise ValueError('Registry identity differs')
    if any(s.chip_unit!='0.01' or s.stacks!=stacks or stacks not in ((10000,10000),(20000,20000)) or s.small_blind!=50 or s.big_blind!=100 for s in plan.scenarios):
        raise ValueError('Fixed HU100/HU200 scenario required')
    for spec in specs:
        model=registry.models[spec['name']]
        if (model.spec!=next(m for m in plan.models if m.name==spec['name'])
                or file_hash(model.source_path)!=spec['sha256'] or model.source_path.stat().st_size!=spec['bytes']
                or model.description['entries']!=spec['entries'] or model.description['iteration']!=spec['iteration']
                or model.description['source_checkpoint_sha256']!=spec['source_checkpoint_sha256']
                or model.identity['stacks']!=list(stacks) or model.identity['small_blind']!=50
                or model.identity['big_blind']!=100 or model.players!=2
                or STACK_BY_SCHEMA[model.abstraction]!=stacks[0]
                or model.identity['game']!=model.game):
            raise ValueError('Pinned direct model/game differs')
        model.configure_translation(None)
        model.configure_missing_key_fallback(spec.get("missing_key_fallback", "uniform"))


class Probe:
    """Trace the sampler without giving its policy any evaluator context."""
    def __init__(self,player,model,context,emit):
        self.player,self.model,self.context,self.emit=player,model,context,emit
    def choose_action(self,view):
        action=self.player.choose_action(view)
        menu,probabilities,known=self.model.distribution(view)
        key=self.model.information_key(view,menu)
        if not any(c.action==action and p>0 for c,p in zip(menu,probabilities)):
            raise ValueError('Direct action outside positive-probability native menu')
        visits=int(self.model.visits.get(key,0))
        row = {**self.context,'hand_id':view.hand_id,'seat':view.seat,'street':view.street.value,
            'action':asdict(action),'key':key,'menu':[asdict(c) for c in menu],'probabilities':list(probabilities),
            'lookup':'missing-key' if not known else 'zero-mass' if key in self.model.zero_mass else 'positive-mass-known-key',
            'visits':visits}
        if self.model.missing_key_fallback != 'uniform':
            row.update(missing_key_fallback=self.model.missing_key_fallback, fallback_applied=not known)
        self.emit(row)
        return action


class ScriptedProbe:
    def __init__(self,player,context,emit):
        self.player,self.context,self.emit=player,context,emit
    def choose_action(self,view):
        action=self.player.choose_action(view)
        self.emit({**self.context,'hand_id':view.hand_id,'seat':view.seat,
                   'street':view.street.value,'action':asdict(action)})
        return action


def execute(specs,blocks,root,rung,out,source,registry=None,reproduce=None, *, stack_bb=100, opponent=None, cache=None, cache_read_only=False):
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=source:
        raise ValueError('Source differs')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Clean committed source required')
    out.mkdir(parents=True,exist_ok=False)
    plan=make_plan(specs,blocks,root,rung,stack_bb=stack_bb,opponent=opponent)
    cache=local_cache(cache)
    started=perf_counter();tick=started
    if registry is None:registry=PolicyRegistry(plan,sorted_average_rows=True,policy_cache=cache,
                                                share_average_storage=True,cache_read_only=cache_read_only)
    registry.plan=plan
    validate(registry,specs,plan)
    load_seconds=perf_counter()-tick
    schedule=build_schedule(plan)
    put(out/'inputs.json',{'source':source,'models':specs,'plan':asdict(plan),'translation':None,
        'plan_sha256':digest(asdict(plan)),**({'scripted_opponent':opponent,'opponent_style':asdict(STYLES[opponent]) if opponent in STYLES else None} if opponent else {})})
    contexts={}
    private_seeds=set()
    def seed_factory(block,rotation,arm,role):
        seed=action_seed(root,block,rotation,role)
        coordinate=(seed,arm) if opponent else seed
        if coordinate in private_seeds:raise ValueError('Private stream collision')
        private_seeds.add(coordinate)
        contexts[seed]={'block':block.index,'rotation':rotation,'role':role,'seed':seed,
                        **({'arm':arm} if opponent else {})}
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
                context={**contexts.pop(seed),'policy':name}
                player=registry.make_policy(name,seed)
                if name not in registry.models:return ScriptedProbe(player,context,emit_decision)
                return Probe(player,registry.models[name],context,emit_decision)
            def emit(row,timing):
                nonlocal hands
                if row['status']!='completed':
                    hand_stream.write(canonical(row)+'\n');hand_stream.flush()
                    raise ValueError('Failed direct hand: '+str(row['error']))
                if repeat_h and canonical(json.loads(next(repeat_h)))!=canonical(row):raise ValueError('Reproduced direct hand differs')
                hand_stream.write(canonical(row)+'\n');hands+=1
            if not run_schedule(plan,schedule,emit,factory=factory,
                    seed_factory=seed_factory,arms=('candidate','baseline') if opponent else ('candidate',)):
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


def audit(out,output,report=False,cost_only=False):
    tick=perf_counter()
    inputs=json.loads((out/'inputs.json').read_text());done=json.loads((out/'complete.json').read_text())
    plan=Plan.from_dict(inputs['plan'])
    stack=plan.scenarios[0].stacks[0]
    opponent=inputs.get('scripted_opponent')
    expected_plan=make_plan(inputs['models'],plan.blocks,plan.root_seed,
        plan.scenarios[0].name.removeprefix('direct-'),stack_bb=stack//100,opponent=opponent)
    if plan!=expected_plan:raise ValueError('Foreign direct/panel plan')
    if opponent and inputs.get('opponent_style')!=(asdict(STYLES[opponent]) if opponent in STYLES else None):
        raise ValueError('Scripted style differs')
    # Replay owns pinned tables independently of the policy instances used to play.
    models={m.name:m for m in plan.models}
    cards={m.name:EquityCards(Path(m.card_buckets)) for m in plan.models if m.abstraction in EQUITY_SCHEMAS}
    if (done['status']!='complete' or inputs['translation'] is not None
            or inputs['plan_sha256']!=digest(inputs['plan'])
            or set(done['files'])!={'hands.jsonl.gz','decisions.jsonl.gz'}):
        raise ValueError('Foreign direct plan')
    for name,pin in done['files'].items():
        if (out/name).stat().st_size!=pin['bytes'] or file_hash(out/name)!=pin['sha256']:
            raise ValueError('Raw direct bytes changed')
    schedule={b.index:b for b in build_schedule(plan)}
    arms=('candidate','baseline') if opponent else ('candidate',)
    expected={(b,r,a) for b in range(plan.blocks) for r in (0,1) for a in arms}
    seen=set();values={};hands=actions=0;coverage={}
    with gzip.open(out/'hands.jsonl.gz','rt') as rows,gzip.open(out/'decisions.jsonl.gz','rt') as traces:
        for line in rows:
            row=json.loads(line);coord=(row['block'],row['rotation'],row['arm'])
            if (coord not in expected or coord in seen or row['arm'] not in arms or row['status']!='completed'
                    or row['scenario']!=plan.scenarios[0].name or row['mode']!='fixed' or row['hand']!=0
                    or row['initial_stacks']!=[stack,stack] or row['big_blind']!=100
                    or row['opponents']!=list(plan.opponents) or row['reloads']!={} or row['error'] is not None):
                raise ValueError('Duplicate/foreign/failed direct hand')
            if row['outcome_sha256']!=digest({k:v for k,v in row.items() if k!='outcome_sha256'}):
                raise ValueError('Direct outcome changed')
            seen.add(coord);block=schedule[row['block']];rotation=row['rotation']
            ids=tuple(f'player-{(seat-rotation)%2}' for seat in (0,1))
            hand=Hand.start(Table(ids,(stack,stack),block.button,50,100,'0.01'),
                hand_id=f'table-0/{block.index}/{rotation}/0',seed=block.deal_seeds[0])
            scripted=make_policy(opponent,action_seed(plan.root_seed,block,rotation,1)) if opponent else None
            for event in row['events']:
                if event['event']!='ActionTaken':continue
                if hand.finished or hand.actor!=event['seat']:raise ValueError('Direct replay actor differs')
                view=hand.observe(hand.actor);d=json.loads(next(traces));role=int(view.player_id!='player-0')
                action=Action(ActionKind(event['action']['kind']),event['action']['raise_to'])
                view.legal_actions.validate(action)
                policy=(plan.candidate if row['arm']=='candidate' else plan.baseline) if role==0 else (opponent or plan.baseline)
                if (d['block']!=block.index or d['rotation']!=rotation or d['role']!=role
                        or d['seed']!=action_seed(plan.root_seed,block,rotation,role)
                        or d['hand_id']!=view.hand_id or d['seat']!=view.seat or d['street']!=view.street.value
                        or d['action']!=event['action'] or d['policy']!=policy
                        or (opponent and d.get('arm')!=row['arm'])):
                    raise ValueError('Direct private stream/action identity differs')
                if scripted is not None and role==1:
                    if scripted.choose_action(view)!=action:
                        raise ValueError('Scripted replay action differs')
                else:
                    spec=models[policy]
                    menu=schema_choices(view,spec.abstraction or (HU200_SCHEMA if stack==20000 else HU100_SCHEMA))
                    key=information_key(view,menu,schema=spec.abstraction or (HU200_SCHEMA if stack==20000 else HU100_SCHEMA),equity_cards=cards.get(policy))
                    if (d['key']!=key or d['menu']!=json.loads(canonical([asdict(c) for c in menu]))
                            or len(d['probabilities'])!=len(menu) or not all(isfinite(p) and p>=0 for p in d['probabilities'])
                            or not isclose(sum(d['probabilities']),1,abs_tol=1e-8)
                            or not any(c.action==action and p>0 for c,p in zip(menu,d['probabilities']))):
                        raise ValueError('Direct private stream/menu/probability/key differs')
                    if d['lookup'] not in ('missing-key','zero-mass','positive-mass-known-key') or d['visits']<0:
                        raise ValueError('Unknown direct coverage classification')
                    fallback = next(s for s in inputs['models'] if s['name'] == policy).get('missing_key_fallback', 'uniform')
                    applied = fallback == 'check-call' and d['lookup'] == 'missing-key'
                    if fallback not in ('uniform', 'check-call'):
                        raise ValueError('Unknown direct fallback')
                    if fallback != 'uniform' and (d.get('missing_key_fallback') != fallback
                            or d.get('fallback_applied') != applied):
                        raise ValueError('Direct fallback telemetry differs')
                    if d['lookup'] != 'positive-mass-known-key':
                        expected_distribution = list(missing_check_call(view, menu)) if applied else [1/len(menu)]*len(menu)
                        if d['probabilities'] != expected_distribution: raise ValueError('Direct fallback differs')
                    band='0' if d['visits']==0 else '1-9' if d['visits']<10 else '10-99' if d['visits']<100 else '100+'
                    cell=coverage.setdefault(d['policy']+'/'+d['street'],{'decisions':0,'lookup':{},'visit_bands':{}})
                    cell['decisions']+=1
                    if fallback != 'uniform':
                        cell['fallback_decisions'] = cell.get('fallback_decisions', 0) + int(applied)
                    for key,value in (('lookup',d['lookup']),('visit_bands',band)):
                        cell[key][value]=cell[key].get(value,0)+1
                hand=hand.apply(action);actions+=1
            net=[p.stack-stack for p in hand.observe(0).players]
            if (not hand.finished or canonical(public_events(hand.events))!=canonical(row['events']) or net!=row['net_chips']
                    or sum(net)!=0 or net[rotation]!=row['candidate_chips'] or row['participants']!=list(ids)):
                raise ValueError('Direct events/settlement differs')
            values[coord]=net[rotation];hands+=1
        if next(traces,None) is not None or seen!=expected or hands!=done['hands'] or actions!=done['decisions']:
            raise ValueError('Incomplete direct replay')
    result={'status':'verified','hands_replayed':hands,'actions_replayed':actions,
        'seconds':perf_counter()-tick,'settlement_sum_zero':True,'all_native_menu_actions':opponent is None,'coverage':coverage}
    if cost_only:
        series=[sum(values[b,r,'candidate']- (values[b,r,'baseline'] if opponent else 0) for r in (0,1))/2
                for b in range(plan.blocks)]
        result['paired_sd_bb_per_100']=stdev(series) if len(series)>1 else None
        if opponent:
            result['arm_sd_bb_per_100']={arm:stdev([sum(values[b,r,arm] for r in (0,1))/2
                for b in range(plan.blocks)]) if plan.blocks>1 else None for arm in arms}
    if report:
        if plan.blocks<2:raise ValueError('At least two blocks required for an interval')
        series=[sum(values[b,r,'candidate']- (values[b,r,'baseline'] if opponent else 0) for r in (0,1))/2
                for b in range(plan.blocks)]
        center=mean(series);sd=stdev(series) if len(series)>1 else 0
        margin=float(t.ppf(.975,len(series)-1))*sd/sqrt(len(series))
        bonf=float(t.ppf(1-.05/6,len(series)-1))*sd/sqrt(len(series))
        result.update(bb_per_100=center,ci95=[center-margin,center+margin],half_width=margin,
            bonferroni_three_ci=[center-bonf,center+bonf],blocks=plan.blocks,
            decision='improves' if center-margin>0 else 'declines' if center+margin<0 else 'inconclusive')
    if report and opponent:
        # Compare models within each shared deal/seat block, never individual hands.
        result['arms']={}
        for arm in arms:
            samples=[sum(values[b,r,arm] for r in (0,1))/2 for b in range(plan.blocks)]
            center=mean(samples);sd=stdev(samples)
            margin=float(t.ppf(.975,plan.blocks-1))*sd/sqrt(plan.blocks)
            result['arms'][arm]={'bb_per_100':center,'ci95':[center-margin,center+margin]}
        contrast=result['bb_per_100'];sd=stdev(series)
        margin=float(t.ppf(1-.05/(2*len(PANEL_OPPONENTS)),plan.blocks-1))*sd/sqrt(plan.blocks)
        result['bonferroni_seven_ci']=[contrast-margin,contrast+margin]
        result['paired_sd_bb_per_100']=sd
    put(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--specs',type=Path,required=True);p.add_argument('--blocks',type=int,required=True)
    p.add_argument('--root',type=int,required=True);p.add_argument('--rung',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',required=True)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--stack-bb',type=int,choices=(100,200),default=100)
    p.add_argument('--policy-cache',type=Path)
    p.add_argument('--cache-read-only',action='store_true')
    a=p.parse_args()
    if a.stack_bb==200 and a.policy_cache is None:p.error('HU200 requires an explicit --policy-cache outside synced storage')
    specs=json.loads(a.specs.read_text());tick=perf_counter()
    registry,primary=execute(specs,a.blocks,a.root,a.rung,a.out/'play',a.source,stack_bb=a.stack_bb,cache=a.policy_cache,cache_read_only=a.cache_read_only)
    checked=audit(a.out/'play',a.out/'replay.json',report=not a.pilot,cost_only=a.pilot)
    _,repeat=execute(specs,a.blocks,a.root,a.rung,a.out/'reproduction',a.source,registry,a.out/'play',stack_bb=a.stack_bb,cache=a.policy_cache,cache_read_only=a.cache_read_only)
    put(a.out/'costs.json',{'seconds':perf_counter()-tick,'primary':primary,'repeat':repeat,
        'audit_seconds':checked['seconds'],
        **({'paired_sd_bb_per_100':checked['paired_sd_bb_per_100']} if a.pilot else {}),'blocks':a.blocks,'outcomes_inspected_for_quote':False})

if __name__=='__main__':main()
