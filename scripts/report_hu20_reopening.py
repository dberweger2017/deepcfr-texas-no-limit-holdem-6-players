"""Independent native replay and block aggregation for the frozen cap A/B."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from random import Random
from statistics import mean, stdev
from time import time

from scipy.stats import t

from scripts.evaluate_hu20_reopening import ATTACKS, Target
from scripts.hu20_reopening_common import case_view
from scripts.play_robustness import replay_row
from scripts.tp20_common import seal
from scripts.train_hu20 import write_json, rss
from scripts.run_tp20_campaign import swap_bytes
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import choices, information_key
from src.blueprint.solver import _seed
from src.blueprint.windowed import _hash
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.diagnostics.robustness import ReactiveAttack


def estimate(values, level=.95):
    n=len(values);mu=mean(values) if n else None
    half=t.ppf((1+level)/2,n-1)*stdev(values)/n**.5 if n>1 else None
    return {'blocks':n,'bb_hand':None if mu is None else mu/100,'bb100':mu,
            'buyins20_per100':None if mu is None else mu/20,
            'level':level,'interval':None if half is None else [mu-half,mu+half]}


def paired(groups,seeds,attacker,index=3,level=.975):
    pairs=[(groups.get((f'A-{s}-{index}',attacker),{}),groups.get((f'B-{s}-{index}',attacker),{})) for s in seeds]
    if not all(a and b for a,b in pairs):return {'status':'unavailable','reason':'Missing a paired target'}
    common=sorted(set.intersection(*(set(g) for pair in pairs for g in pair)))
    values=[mean([b[k]-a[k] for a,b in pairs]) for k in common]
    return {'status':'available','B_minus_A':estimate(values,level),
            'A_absolute':estimate([mean([a[k] for a,b in pairs]) for k in common],level),
            'B_absolute':estimate([mean([b[k] for a,b in pairs]) for k in common],level),
            'per_seed':{str(s):estimate([b[k]-a[k] for k in common],level) for s,(a,b) in zip(seeds,pairs)}}


def verify_phase(folder):
    path=folder/'checksums.json'
    if not path.exists():return 0
    count=0
    for name,h in json.loads(path.read_text()).items():
        if _hash(folder/name)!=h:raise ValueError(f'Changed artifact: {folder/name}')
        count+=1
    return count


def verify_hand(row,spec,source):
    n=2;rot=row['rotation'];ids=tuple(f'player-{(s-rot)%n}' for s in range(n))
    hand=Hand.start(Table(ids,(2000,)*n,button=row['button']),
        hand_id=f"robustness-{row['phase']}-{n}-{row['block']}",seed=row['deal_seed'])
    rng=Random(stream_seed(row['root_seed'],'test','action',2,row['block'],0))
    old_off=own_off=False
    for item in row['actions']:
        if item['seat']!=hand.actor:raise ValueError('Replay actor')
        v=hand.observe(hand.actor);action=Action(ActionKind(item['kind']),item['raise_to'])
        old=choices(v,free_fold=False);own=choices(v,raise_cap=source.raise_cap,free_fold=False)
        old_member=action in [c.action for c in old];own_member=action in [c.action for c in own]
        if item['on_original_cap2_menu']!=old_member or item['on_target_menu']!=own_member:
            raise ValueError('Dual action-contract telemetry')
        if item['preceding_original_off_menu']!=old_off or item['preceding_target_off_menu']!=own_off:
            raise ValueError('Dual preceding-history telemetry')
        if item['logical_player']==0:
            menu,p,hit=source.distribution(v)
            if rng.choices(menu,weights=p,k=1)[0].action!=action:raise ValueError('Frozen target/action RNG replay')
            if item['target_trained']!=hit:raise ValueError('Trained/fallback flag')
            key=information_key(v,menu,schema=spec['abstraction'])
            if item['target_key']!=key:raise ValueError('Target-specific public key')
            if hasattr(source,'visits') and item['target_visits']!=source.visits.get(key,0):raise ValueError('Visit density')
        elif row['attacker'] in [a[0] for a in ATTACKS]:
            _,rule,contract=next(a for a in ATTACKS if a[0]==row['attacker'])
            if ReactiveAttack(rule,contract).choose_action(v)!=action:raise ValueError('Changed fixed reactive attack')
        elif row['attacker']=='LBR-original-cap2' and not old_member:
            raise ValueError('LBR used an expanded attacker action')
        hand=hand.apply(action);old_off|=not old_member;own_off|=not own_member
    if digest([repr(e) for e in hand.events])!=row['event_digest']:raise ValueError('Native event digest')
    if row['status']=='complete':
        net=[p.stack-2000 for p in hand.observe(0).players]
        if not hand.finished or sum(net)!=0 or net!=row['net_chips_by_seat'] or net[rot]!=row['target_chips']:
            raise ValueError('Native chip settlement')
    return hand


def report(root,out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time();plan=json.loads((root/'frozen-plan.json').read_text())
    campaign=json.loads((root/'campaign.json').read_text());verified=0;pref_replays=0
    old_checks=root/'preflight-checksums.json'
    if old_checks.exists():
        for name,h in json.loads(old_checks.read_text()).items():
            renamed={'campaign.json':'preflight-campaign-record.json','resources.jsonl':'preflight-resources.jsonl'}
            f=root/renamed.get(name,name)
            if _hash(f)!=h:raise ValueError(f'Changed original preflight artifact: {name}')
            verified+=1
    independent_deals=set();fixture=Path(plan['independent_path']);fixture_count=0
    if (root/'independent-manifest.json').exists():
        manifest=json.loads((root/'independent-manifest.json').read_text())
        if _hash(fixture)!=manifest['sha256']:raise ValueError('Independent observation hash')
    with gzip.open(fixture,'rt') as saved:
        for line in saved:
            r=json.loads(line);case_view(r);independent_deals.add(r['seed']);fixture_count+=1
    for folder in sorted((root/'preflight').iterdir()):
        if not folder.is_dir():continue
        verified+=verify_phase(folder)
        for f in folder.glob('*-timing-hands.jsonl.gz'):
            with gzip.open(f,'rt') as saved:
                for line in saved:
                    r=json.loads(line)
                    if r['target_chips'] is not None or r['net_chips_by_seat'] is not None:raise ValueError('Preflight opened profit')
                    replay_row(r);pref_replays+=1
    training=[];training_deals=set();preflight_deals=set()
    for folder in sorted((root/'preflight').iterdir()):
        r=json.loads((folder/'result.json').read_text())
        for i in range(1,r['completed_iterations']+3):
            for s in (0,1):preflight_deals.add(_seed(r['seed'],i,s,0,'deal'))
    for seed in plan['training_seeds']:
        for arm in ('A','B'):
            folder=root/'training'/f'{arm}-{seed}'
            if not (folder/'result.json').exists():continue
            verified+=verify_phase(folder);r=json.loads((folder/'result.json').read_text());training.append(r)
            if r['initial_entries']!=0 or r['initial_iteration']!=0:raise ValueError('Nonfresh training lineage')
            for i in range(1,r['iterations']+2):
                for s in (0,1):training_deals.add(_seed(seed,i,s,0,'deal'))
    evalroot=root/'evaluation'
    if not (evalroot/'models.json').exists():
        result={'status':'incomplete','training':training,'campaign':campaign,
                'preflight_replayed_hands':pref_replays,'verified_phase_files':verified,
                'hands':0,'evaluation':'not started after training obstruction','primary':'unavailable'}
        write_json(out/'results.json',result);seal(out);return result
    verified+=verify_phase(evalroot)
    specs={s['name']:s for s in json.loads((evalroot/'models.json').read_text())}
    groups=defaultdict(lambda:defaultdict(dict));roles=defaultdict(lambda:defaultdict(list));telemetry=defaultdict(Counter)
    visit_hist=defaultdict(Counter);lbr=defaultdict(Counter);failures=[];cases=[];count=0;deals=set();coupling={};source=None;current=None
    with gzip.open(evalroot/'hands.jsonl.gz','rt') as saved:
        for line in saved:
            row=json.loads(line);count+=1;name=row['policy'];attack=row['attacker'];key=(name,attack);spec=specs[name]
            if row['deal_seed']!=stream_seed(row['root_seed'],'test','deal',2,row['block']) or row['button']!=row['block']%2:
                raise ValueError('Frozen schedule')
            ck=(attack,row['block'],row['rotation']);cv=(row['deal_seed'],row['button'])
            if ck in coupling and coupling[ck]!=cv:raise ValueError('Unpaired arms/milestones/references')
            coupling[ck]=cv;deals.add(row['deal_seed'])
            if name!=current:source=Target(spec);current=name
            verify_hand(row,spec,source)
            if row['status']!='complete':failures.append({k:v for k,v in row.items() if k!='actions'});continue
            b=row['block'];rot=row['rotation']
            if rot in groups[key][b]:raise ValueError('Duplicate rotation')
            groups[key][b][rot]=row['target_chips']
            role='button_small_blind' if rot==row['button'] else 'big_blind'
            roles[(key,role)][b].append(row['target_chips'])
            for a in row['actions']:
                if a['logical_player']==0:
                    telemetry[key][(a['street'],a['kind'],'trained' if a['target_trained'] else 'fallback',a['preceding_original_off_menu'],a['preceding_target_off_menu'])]+=1
                    if a['target_visits'] is not None:visit_hist[key][(a['street'],a['target_visits'])]+=1
                if 'lbr' in a:
                    d=a['lbr'];lbr[key]['decisions']+=1;lbr[key]['complete']+=d['completed'];lbr[key]['over_soft_budget']+=d['over_soft_budget']
                    lbr[key]['max_seconds']=max(lbr[key]['max_seconds'],d['seconds']);lbr[key]['samples']+=d['samples'];lbr[key]['zero_likelihood_cumulative']+=d['zero_likelihood_events']
            if name==f"B-{plan['training_seeds'][0]}-3" and attack=='Pressure-native' and len(cases)<12 and any(a['street_raises']>2 for a in row['actions']):cases.append(row)
    blocks={k:{b:mean(v.values()) for b,v in bs.items() if set(v)=={0,1}} for k,bs in groups.items()}
    prior=set();prior_files=0
    for dirname in plan['prior_roots']:
        folder=Path(dirname)
        for f in folder.rglob('*hands.jsonl*'):
            opener=gzip.open if f.suffix=='.gz' else open
            with opener(f,'rt') as old:
                for line in old:
                    r=json.loads(line)
                    if 'deal_seed' in r:prior.add(r['deal_seed'])
            prior_files+=1
    if deals&(prior|training_deals|preflight_deals|independent_deals) or independent_deals&(prior|training_deals|preflight_deals):
        raise ValueError('Opened/training/preflight/independent deal overlap')
    primary={a:paired(blocks,plan['training_seeds'],a,level=.975) for a in ('Pressure-native','LBR-original-cap2')}
    primary_roles={role:{a:paired({k:{b:mean(xs) for b,xs in roles[(k,role)].items()} for k in blocks},plan['training_seeds'],a,level=.975) for a in ('Pressure-native','LBR-original-cap2')} for role in ('button_small_blind','big_blind')}
    effects=[]
    for index in range(4):
        for a in (*[x[0] for x in ATTACKS],*plan['secondary_opponents']):
            effects.append({'milestone':index,'attacker':a,**paired(blocks,plan['training_seeds'],a,index,.95)})
    results=[]
    for k,bs in blocks.items():
        results.append({'policy':k[0],'attacker':k[1],'absolute_target':estimate(list(bs.values())),
            'absolute_attacker':estimate([-v for v in bs.values()]),
            'roles':{role:estimate([mean(v) for v in roles[(k,role)].values()]) for role in ('button_small_blind','big_blind')},
            'decisions':[{'coordinates':list(c),'count':n} for c,n in sorted(telemetry[k].items())],
            'decision_weighted_visits':[{'street':s,'visits':v,'decisions':n} for (s,v),n in sorted(visit_hist[k].items())],
            'lbr':dict(lbr[k])})
    attempts=json.loads((evalroot/'attempts.json').read_text());expected=(24+len(plan['reference_policies']))*5*plan['cheap_blocks']*2+6*plan['lbr_blocks']*2+6*6*plan['secondary_panel_blocks']*2
    complete=len(training)==6 and all(r['status']=='complete' for r in training) and not failures and count==expected and all(a['status']=='complete' for a in attempts)
    resources=[json.loads(l) for l in (root/'resources.jsonl').read_text().splitlines()]
    prefrows=[json.loads(f.read_text()) for f in root.glob('preflight/*/result.json')]
    ev=json.loads((evalroot/'result.json').read_text())
    peak=max([r['rss_bytes'] for r in resources]+[r['peak_rss_bytes'] for r in [*prefrows,*training,ev]]+[rss()])
    swap=[swap_bytes(r['swap']) for r in resources];swap=[v for v in swap if v is not None]
    resource_checks={'rss_within_limit':peak<=10.5*1024**3,'disk_within_limit':min(r['free_disk_bytes'] for r in resources)>=8*1024**3,
        'swap_within_limit':not swap or max(swap)-swap_bytes(campaign['swap_baseline'])<=.5*1024**3,
        'within_original_deadline':time()<campaign['deadline']}
    complete=complete and all(resource_checks.values())
    native=primary['Pressure-native'];safe=primary['LBR-original-cap2']
    improved=native.get('B_minus_A',{}).get('interval')
    safeguarded=safe.get('B_minus_A',{}).get('interval')
    conclusion={'pressure_gain_established':bool(complete and improved and improved[0]>0),
        'lbr_noninferiority_established':bool(complete and safeguarded and safeguarded[0]>-plan['noninferiority_margin_bb100']),
        'margin_bb100':plan['noninferiority_margin_bb100'],'no_promotion':True}
    result={'status':'complete' if complete else 'incomplete','hands':count,'expected_hands':expected,
        'preflight_replayed_hands':pref_replays,'native_replayed_hands':count,'verified_phase_files':verified,
        'primary':primary,'primary_roles':primary_roles,'exploratory_effects':effects,'per_policy':results,'training':training,'attempts':attempts,
        'failures':failures,'conclusion':conclusion,'prior_deals':len(prior),'prior_hand_files':prior_files,
        'opened_deal_overlap':0,'training_deal_overlap':0,'preflight_deal_overlap':0,
        'independent_observations_replayed':fixture_count,'independent_deal_overlap':0,
        'resources':{'checks':resource_checks,'peak_rss_bytes':peak,'minimum_free_disk_bytes':min(r['free_disk_bytes'] for r in resources),
          'swap_baseline':campaign['swap_baseline'],'last_swap':resources[-1]['swap'],'elapsed_since_preflight':time()-campaign['started']},
        'plan_digest':digest(plan),'report_seconds':time()-start,'finished':time()}
    write_json(out/'results.json',result);write_json(out/'fixed-order-replay-cases.json',cases);seal(out)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=report(a.root,a.out);print(json.dumps({'status':r['status'],'hands':r['hands'],'primary':r.get('primary')}));return 0

if __name__=='__main__':raise SystemExit(main())
