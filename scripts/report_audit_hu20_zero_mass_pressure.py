"""Report and independently replay the declared four-arm native-pressure check."""
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics
import time
from scipy.stats import t
from src.arena.report import estimate
from src.arena.runner import public_events
from src.arena.schedule import digest
from src.blueprint.abstraction import choices, information_key, HU20_UNCAPPED_SCHEMA
from src.diagnostics.stackoff_tails import snapshot, hand_tails
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind

ARMS=('Tprime','T','Oprime','O')
CONTRASTS=(('Tprime','T'),('Oprime','O'))

def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)
def write(path,value):
    with gzip.open(path,'xt') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
def sha(path):
    import hashlib
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stats(values):
    n=len(values);mean=math.fsum(values)/n
    var=math.fsum((v-mean)**2 for v in values)/(n-1)
    margin=float(t.ppf(.975,n-1))*math.sqrt(var/n)
    return {'blocks':n,'bb_per_100':mean,'ci95':[mean-margin,mean+margin] if n>=30 and var else None}
def compare(a,b):
    assert a['blocks']==b['blocks'] and math.isclose(a['bb_per_100'],b['bb_per_100'],abs_tol=1e-9)
    assert (a['ci95'] is None)==(b['ci95'] is None)
    if a['ci95'] is not None:assert all(math.isclose(x,y,abs_tol=1e-9) for x,y in zip(a['ci95'],b['ci95'],strict=True))

def analyze(plan,directory,replay):
    started=time.perf_counter();cells={};decisions=0;coverage=defaultdict(Counter);files=[]
    count=plan['panels'][0]['blocks'];assert len(plan['panels'])==1 and plan['panels'][0]['name']=='native-pressure'
    for spec in plan['models']:
        name=spec['name'];result=read(directory/(name+'.result.json.gz'))
        assert result['status']=='complete' and result['plan_sha256']==digest(plan)
        path=directory/(name+'.hands.jsonl.gz');files.append({'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size})
        seen=set()
        with gzip.open(path,'rt') as f:
            for r in map(json.loads,f):
                b,rotation=r['block'],r['rotation'];coordinate=b,rotation
                assert coordinate not in seen and b in range(count) and rotation in (0,1);seen.add(coordinate)
                assert r['root_seed']==plan['root'] and r['policy']==name and r['seed']==spec['seed'] and r['arm']==spec['arm']
                assert r['button']==b%2 and r['native_replay_verified'] and r['panel']=='native-pressure'
                lineage=int(str(spec['seed'])[-1]);cells[spec['arm'],lineage,b,rotation]=r['target_chips']
                coverage[spec['arm'],lineage].update(r['coverage'])
                if not replay:continue
                seed=(2<<62)|(int(digest((2,plan['root'],'deal',(2,b)))[:16],16)&(2**62-1))
                assert r['deal_seed']==seed and r['hand_id']==f'cfr-average/native-pressure/{b}/{rotation}'
                hand=Hand.start(Table(('seat0','seat1'),(2000,2000),button=r['button']),hand_id=r['hand_id'],seed=seed)
                checked=Counter()
                for index,a in enumerate(r['actions']):
                    assert hand.actor==a['seat'] and a['index']==index
                    view=hand.observe(hand.actor);logical=int(hand.actor!=rotation)
                    assert a['logical_player']==logical and a['street']==view.street.value
                    menu=choices(view,raise_cap=None,free_fold=False)
                    for choice in menu:view.legal_actions.validate(choice.action)
                    observed=snapshot(view,menu,a['observation']['probabilities'],a['observation']['trained'],None)
                    observed['logical_player']=logical
                    assert a['observation']==json.loads(json.dumps(observed))
                    if not logical:
                        assert a['target_key']==information_key(view,menu,schema=HU20_UNCAPPED_SCHEMA)
                        status=a['average_mass_status'];assert status in ('missing','zero_mass','positive_mass')
                        assert bool(observed['trained'])==(status!='missing')
                        probs=observed['probabilities'];assert all(math.isfinite(v) and v>=0 for v in probs) and math.isclose(math.fsum(probs),1,abs_tol=1e-9)
                        if status=='missing' or (status=='zero_mass' and spec['arm'] in ('T','O')):
                            assert all(math.isclose(v,1/len(probs),abs_tol=1e-9) for v in probs)
                        checked[status]+=1;checked[a['street']+':'+status]+=1
                    action=Action(ActionKind(a['kind']),a['raise_to']);view.legal_actions.validate(action);hand=hand.apply(action);decisions+=1
                chips=[p.stack-2000 for p in hand.observe(0).players]
                assert hand.finished and chips==r['net_chips_by_seat'] and sum(chips)==0 and chips[rotation]==r['target_chips']
                assert digest(public_events(hand.events))==r['public_events_sha256'] and dict(checked)==r['coverage'] and hand_tails(r)==r['tails']
        assert seen=={(b,r) for b in range(count) for r in (0,1)} and result['hands']==2*count
        print(name+' '+('independently replayed' if replay else 'reported'),flush=True)
    statistic=stats if replay else estimate
    absolute={arm:statistic([math.fsum(cells[arm,l,b,r] for l in (1,2,3) for r in (0,1))/6 for b in range(count)]) for arm in ARMS}
    contrasts={};details=[]
    for a,c in CONTRASTS:
        label=a+'-'+c
        contrasts[label]=statistic([math.fsum(cells[a,l,b,r]-cells[c,l,b,r] for l in (1,2,3) for r in (0,1))/6 for b in range(count)])
        for lineage in (None,1,2,3):
            ls=(1,2,3) if lineage is None else (lineage,)
            for position in (None,'button','big_blind'):
                values=[]
                for b in range(count):
                    rs=(0,1) if position is None else (b%2,) if position=='button' else (1-b%2,)
                    values.append(math.fsum(cells[a,l,b,r]-cells[c,l,b,r] for l in ls for r in rs)/(len(ls)*len(rs)))
                details.append({'contrast':label,'lineage':lineage,'position':position,**statistic(values)})
    output={'plan_sha256':digest(plan),'hands':len(cells),'absolute':absolute,'contrasts':contrasts,'lineage_position_contrasts':details,
        'coverage_by_lineage':[{'arm':a,'lineage':l,'counts':dict(c)} for (a,l),c in sorted(coverage.items())],
        'raw_files':files,'seconds':time.perf_counter()-started,'scope':'native-pressure only; fixed 12288 blocks; no release gate'}
    if replay:
        reported=read(directory/'summary.json.gz')
        for key in absolute:compare(absolute[key],reported['absolute'][key])
        for key in contrasts:compare(contrasts[key],reported['contrasts'][key])
        for a,b in zip(details,reported['lineage_position_contrasts'],strict=True):compare(a,b)
        assert output['coverage_by_lineage']==reported['coverage_by_lineage']
        output.update(status='verified',decisions_checked=decisions,independent_arithmetic_matches=True)
    write(directory/('audit.json.gz' if replay else 'summary.json.gz'),output)
    return output
