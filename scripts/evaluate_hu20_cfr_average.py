"""Fresh paired current/average diagnostic panels; no training or promotion."""
import argparse
from collections import Counter, defaultdict
import gc
import gzip
import json
from pathlib import Path
from random import Random
import resource
import subprocess
from time import perf_counter

from src.arena.catalog import Checkpoint
from src.arena.policies import make_policy
from src.arena.report import estimate
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import choices, information_key, HU20_UNCAPPED_SCHEMA
from src.blueprint.artifact import FrozenBlueprint, HU20_UNCAPPED_FORMAT
from src.diagnostics.cached_lbr import SharedProbabilityCache
from src.blueprint.average import AveragePolicy
from src.diagnostics.exact_ranker import RankedCachedLocalBestResponse
from src.diagnostics.robustness import LBRConfig, ReactiveAttack
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.selective_stackoff import SelectiveStackoff
from src.diagnostics.stackoff_tails import snapshot, hand_tails
from src.arena.runner import public_events
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


class Uniform:
    def __init__(self,seed):self.random=Random(seed)
    def choose_action(self,view):return self.random.choice(choices(view,raise_cap=None,free_fold=False)).action


def opponent(panel,source,seed):
    rule=panel['rule']
    if rule=='uniform':return Uniform(seed)
    if rule=='selective_stackoff':return SelectiveStackoff(seed)
    if rule=='lbr':return RankedCachedLocalBestResponse(source,seed,SharedProbabilityCache(source),LBRConfig(4,5))
    if panel['contract']=='style-native':return make_policy(rule,seed)
    return ReactiveAttack(rule,panel['contract'])


def play(source,spec,panel,root,block,rotation,guard=lambda:None,*,rival=None):
    deal_seed=stream_seed(root,'test','deal',2,block)
    action_random=Random(stream_seed(root,'test','action',2,block,0))
    if rival is None:rival=opponent(panel,source,stream_seed(root,'test','opponent',2,block,1))
    hand_id=f"cfr-average/{panel['name']}/{block}/{rotation}"
    hand=Hand.start(Table(('seat0','seat1'),(2000,2000),button=block%2),hand_id=hand_id,seed=deal_seed)
    actions=[];coverage=Counter();start=perf_counter()
    for index in range(1000):
        guard()
        if hand.finished:break
        view=hand.observe(hand.actor);logical=int(hand.actor!=rotation);key=None;mass_status=None
        if not logical:
            menu,p,trained=source.distribution(view);key=information_key(view,menu,schema=HU20_UNCAPPED_SCHEMA)
            mass_status='missing' if not trained else 'zero_mass' if key in getattr(source,'zero_mass',()) else 'positive_mass' if isinstance(source,AveragePolicy) else 'current'
            coverage[mass_status]+=1;coverage[view.street.value+':'+mass_status]+=1
            action=action_random.choices(menu,weights=p,k=1)[0].action
        else:
            menu=choices(view,raise_cap=None,free_fold=False);p=None;trained=None
            action=rival.choose_action(view)
        observed=snapshot(view,menu,p,trained,None);observed['logical_player']=logical
        entry={'index':index,'seat':hand.actor,'logical_player':logical,'street':view.street.value,
            'kind':action.kind.value,'raise_to':action.raise_to,'observation':observed,
            'target_key':key,'average_mass_status':mass_status}
        if logical and isinstance(rival,RankedCachedLocalBestResponse):entry['lbr']=rival.telemetry[-1]
        view.legal_actions.validate(action);actions.append(entry);hand=hand.apply(action)
    if not hand.finished:raise RuntimeError('Diagnostic hand decision limit')
    chips=[p.stack-2000 for p in hand.observe(0).players]
    replay=Hand.start(hand.table,hand_id=hand_id,seed=deal_seed)
    for a in actions:
        if replay.actor!=a['seat']:raise ValueError('Replay actor differs')
        replay=replay.apply(Action(ActionKind(a['kind']),a['raise_to']))
    events=digest(public_events(hand.events))
    if not replay.finished or [p.stack-2000 for p in replay.observe(0).players]!=chips or digest(public_events(replay.events))!=events or sum(chips)!=0:
        raise ValueError('Native replay/settlement differs')
    row={'status':'complete','policy':spec['name'],'strategy':spec['strategy'],'seed':spec['seed'],'players':2,
        'panel':panel['name'],'contract':panel['contract'],'block':block,'rotation':rotation,'button':block%2,
        'deal_seed':deal_seed,'root_seed':root,'hand_id':hand_id,'actions':actions,'target_chips':chips[rotation],
        'net_chips_by_seat':chips,'public_events_sha256':events,'native_replay_verified':True,
        'coverage':dict(coverage),'seconds':perf_counter()-start}
    row['tails']=hand_tails(row)
    return row


def summarize(rows):
    groups=defaultdict(list);panels=[];paired={}
    for r in rows:groups[r['seed'],r['strategy'],r['panel']].append(r)
    for (seed,strategy,name),items in sorted(groups.items()):
        positions=defaultdict(dict);counts=Counter();coverage=Counter();keys=set();street_keys=defaultdict(set);limited=0;parts=defaultdict(lambda:[0,0])
        for r in items:
            position='button' if r['rotation']==r['button'] else 'big_blind'
            if r['block'] in positions[position]:raise ValueError('Duplicate paired coordinate')
            positions[position][r['block']]=r['target_chips'];counts.update(r['tails']['counts']);coverage.update(r['coverage'])
            part=parts[r['tails']['first_large_raise_response']];part[0]+=1;part[1]+=r['target_chips']
            for a in r['actions']:
                if a['target_key']:keys.add(a['target_key']);street_keys[a['street']].add(a['target_key'])
                if 'lbr' in a:limited+=not a['lbr']['completed']
        if set(positions['button'])!=set(positions['big_blind']):raise ValueError('Incomplete position pair')
        blocks=sorted(positions['button']);series=[(positions['button'][b]+positions['big_blind'][b])/2 for b in blocks]
        paired[seed,strategy,name]=(blocks,series,positions)
        panels.append({'seed':seed,'strategy':strategy,'panel':name,'hands':len(items),'counts':dict(counts),
            'coverage':dict(coverage),'distinct_target_keys':len(keys),'distinct_keys_by_street':{k:len(v) for k,v in street_keys.items()},
            'overall':estimate(series),'positions':{p:estimate([v[b] for b in blocks]) for p,v in positions.items()},
            'paired_block_chips':series,'blocks':blocks,'limited_lbr_decisions':limited,
            'whole_hand_partitions':{p:{'hands':n,'sum_chips':chips,'bb_per_100':chips/n} for p,(n,chips) in parts.items()}})
    changes=[];aggregates=[]
    for seed,name in sorted({(s,p) for s,_,p in paired}):
        if (seed,'current',name) not in paired or (seed,'average',name) not in paired:continue
        b,current,cp=paired[seed,'current',name];ab,average,ap=paired[seed,'average',name]
        if b!=ab:raise ValueError('Current/average block pairing differs')
        delta=[a-c for a,c in zip(average,current)]
        changes.append({'seed':seed,'panel':name,'overall':estimate(delta),'paired_delta_chips':delta,
            'positions':{p:estimate([ap[p][i]-cp[p][i] for i in b]) for p in cp}})
    for name in sorted({c['panel'] for c in changes}):
        selected=[c for c in changes if c['panel']==name]
        if len(selected)==3:
            aggregates.append({'panel':name,'average_minus_current':estimate([sum(v)/3 for v in zip(*(c['paired_delta_chips'] for c in selected))]),
                'scope':'paired fresh blocks; conditional on three original lineages; exploratory 95%'})
    return {'panels':panels,'changes':changes,'three_lineage_changes':aggregates}


def run(plan,inputs,averages,out):
    out.mkdir(parents=True,exist_ok=False);start=perf_counter();rows=[];loaded=[];failure=None
    source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    def guard():
        if perf_counter()-start>plan['max_seconds']:raise TimeoutError('Frozen worker deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>6*1024**3:raise MemoryError('M1 process peak guard')
    try:
        for spec in plan['models']:
            guard();path=(inputs if spec['strategy']=='current' else averages)/spec['path'];begun=perf_counter()
            if path.stat().st_size!=spec['bytes'] or file_hash(path)!=spec['sha256']:raise ValueError('Policy bytes differ before loading')
            source=(FrozenBlueprint(Checkpoint(spec['name'],str(path),spec['sha256'],HU20_UNCAPPED_FORMAT),path)
                    if spec['strategy']=='current' else AveragePolicy(path,spec['sha256']))
            if (source.description['training_seed']!=spec['seed'] or source.description['iteration']!=spec['iteration']
                or source.abstraction!=HU20_UNCAPPED_SCHEMA or source.raise_cap is not None):raise ValueError('Policy identity differs')
            if spec['strategy']=='average' and source.description['source_checkpoint_sha256']!=spec['checkpoint_sha256']:
                raise ValueError('Average checkpoint provenance differs')
            loaded.append({'model':spec,'description':source.description,'load_seconds':perf_counter()-begun})
            with gzip.open(out/(spec['name']+'.hands.jsonl.gz'),'wt') as f:
                for panel in plan['panels']:
                    begin=perf_counter()
                    for block in range(panel['blocks']):
                        for rotation in (0,1):
                            row=play(source,spec,panel,plan['root'],block,rotation,guard);rows.append(row)
                            f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush()
                    print(json.dumps({'model':spec['name'],'panel':panel['name'],'hands':2*panel['blocks'],'seconds':perf_counter()-begin}),flush=True)
            del source;gc.collect()
    except Exception as exc:failure=f'{type(exc).__name__}: {exc}'
    result={'status':'incomplete' if failure else 'complete','failure':failure,'source':source_sha,'plan':plan,
            'plan_sha256':digest(plan),'hands':len(rows),'loaded':loaded,'seconds':perf_counter()-start,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if not failure:result.update(summarize(rows))
    (out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'manifest.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in out.iterdir()},indent=2,sort_keys=True)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--averages',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=run(json.loads(a.plan.read_text()),a.inputs,a.averages,a.out)
    print(json.dumps({k:result[k] for k in ('status','failure','hands','seconds','peak_rss_bytes')}))
    if result['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
