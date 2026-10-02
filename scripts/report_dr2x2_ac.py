"""Independent sealed-record replay, saved-policy audit and matched-block tables."""
import argparse
from collections import Counter, defaultdict
import csv
import gc
import gzip
import json
from math import fsum, sqrt, isclose
from pathlib import Path
from random import Random
import shutil
import tarfile
import time

from scipy.stats import t
from scripts.dr2x2_eval_control import verify_closed_archive
from scripts.evaluate_dr2x2_ac import open_average
from scripts.evaluate_hu20_river import root_hand
from scripts.hu20_platform_pilot import write, peak_rss
from scripts.evaluate_hu20_stackoff import swap_bytes
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import information_key, Choice
from src.blueprint.river_cfr import profile_quality
from src.blueprint.river_game import river_root_history
from src.diagnostics.history_river import CommonRiverGame, common_ranges, policy_profile, LAW, PROJECTION
from src.diagnostics.saved_hu20 import load_saved, file_hash
from src.diagnostics.stackoff_tails import snapshot, hand_tails
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind

SEEDS=(2026093001,2026093002,2026093003)
ROLES=('button','big_blind')


def interval(values):
    """Chip100 is one BB; chips/hand numerically equals BB/100."""
    n=len(values)
    if n<2:raise ValueError('Insufficient independent blocks')
    total=fsum(values);mu=total/n
    variance=fsum((v-mu)**2 for v in values)/(n-1)
    half=float(t.ppf(.975,n-1))*sqrt(variance/n)
    return {'blocks':n,'bb_per_100':mu,'ci95_low':mu-half,'ci95_high':mu+half,
            'block_chip_sum':total,'sample_variance_chips':variance}


def block_values(series,stage,panel,terms,seeds=SEEDS,role='overall'):
    keys=[(stage,panel,s,cell,readout) for s in seeds for cell,readout,_ in terms]
    if not all(k in series for k in keys):raise ValueError('Missing fixed lineage/readout')
    blocks=set(series[keys[0]])
    if any(set(series[k])!=blocks for k in keys):raise ValueError('Mismatched paired blocks')
    positions=ROLES if role=='overall' else (role,)
    return [fsum(weight*series[(stage,panel,s,cell,readout)][b][r]
                 for s in seeds for cell,readout,weight in terms for r in positions)/(len(seeds)*len(positions))
            for b in sorted(blocks)]


def csv_rows(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def replay_saved(row,source,visits,spec,readout,guard):
    rot=row['rotation'];ids=tuple(f'player-{(seat-rot)%2}' for seat in range(2))
    native=Hand.start(Table(ids,(2000,2000),button=row['button']),
                     hand_id=f"robustness-{row['phase']}-2-{row['block']}",seed=row['deal_seed'])
    rng=Random(stream_seed(row['root_seed'],'test','action',2,row['block'],0))
    for index,item in enumerate(row['actions']):
        if native.actor!=item['seat'] or item['index']!=index or int(native.actor!=rot)!=item['logical_player']:
            raise ValueError('Native actor/rotation differs')
        view=native.observe(native.actor);observed=item['observation']
        action=Action(ActionKind(item['kind']),item['raise_to'])
        if item['logical_player']==0:
            menu,p,hit=source.distribution(view);key=information_key(view,menu,schema=spec['abstraction'])
            expected=snapshot(view,menu,p,hit,visits.get(key,0));expected.update(logical_player=0,key=key,
                readout_mass_status='missing' if not hit else 'zero_mass' if key in getattr(source,'zero_mass',()) else 'positive_mass' if readout=='average' else 'current')
            if rng.choices(menu,weights=p,k=1)[0].action!=action:raise ValueError('Target action RNG differs')
        else:
            menu=tuple(Choice(m['name'],Action(ActionKind(m['kind']),m['raise_to'])) for m in observed['menu'])
            for choice in menu:view.legal_actions.validate(choice.action)
            expected=snapshot(view,menu);expected['logical_player']=1
        if json.loads(json.dumps(expected))!=observed:raise ValueError('Saved-policy observation/key/visits/probability differs')
        native=native.apply(action)
    net=[p.stack-2000 for p in native.observe(0).players]
    if (not native.finished or sum(net) or net!=row['net_chips_by_seat'] or net[rot]!=row['target_chips']
            or digest([repr(e) for e in native.events])!=row['event_digest']):
        raise ValueError('Independent native settlement/events differ')
    if hand_tails(row)!=row['tails']:raise ValueError('Independent tail arithmetic differs')


def report(root,plan,inputs,out):
    final=json.loads((root/'operator-finished.json').read_text());jobs=json.loads((root/'pods.json').read_text())
    if (final['status']!='evaluation-complete' or final['remaining_ids'] or len(jobs)!=3
            or {j['seed'] for j in jobs}!=set(SEEDS) or any(j['status']!='retrieved-complete' or not j.get('terminated') for j in jobs)):
        raise ValueError('All three closed verified lineages required before aggregate')
    out.mkdir(parents=True,exist_ok=False);started=time.time();before=swap_bytes();last=0
    def guard():
        nonlocal last
        now=time.time()
        if now-last<1:return
        last=now
        if peak_rss()>6*2**30 or swap_bytes()-before>.5*2**30 or shutil.disk_usage(out).free<8*2**30:
            raise RuntimeError('M1 report resource guard; retain partials')
    series={};coupling={};counts=defaultdict(Counter);parts=defaultdict(Counter);coverage=defaultdict(Counter)
    raw_hands=0;raw_actions=0;archives=[];river=[];quality_audits=0
    definitions={p['name']:p for p in plan['panels']};plan_sha=digest(plan)
    for job in sorted(jobs,key=lambda j:j['seed']):
        guard();archive_path=Path(job['archive_path'])
        if file_hash(archive_path)!=job['archive_sha256'] or archive_path.stat().st_size!=job['archive_bytes']:
            raise ValueError('Closed archive transport changed')
        checked=verify_closed_archive(archive_path);archives.append({'seed':job['seed'],'path':str(archive_path),'sha256':job['archive_sha256'],'bytes':job['archive_bytes'],**checked})
        with tarfile.open(archive_path,'r:') as tar:
            prefix='results/dr2x2-eval/'
            def read(name):return json.load(tar.extractfile(prefix+name))
            summary=read('evaluation/summary.json');parity=read('linux-parity.json');capacity=read('linux-capacity.json');worker=read('finished.json')
            if (summary['status']!='complete' or summary['closed_hands']!=55296 or summary['plan_sha256']!=plan_sha
                    or len(summary['tasks'])!=6 or not parity['passed'] or not capacity['passed'] or worker['source']!='6ce14e1513ce1b6f452b59930bf8f2034ad80115'):
                raise ValueError('Frozen source/work/parity/capacity proof differs')
            selected=[s for s in plan['models'] if s['seed']==job['seed']]
            for spec in selected:
                for readout in ('current','average'):
                    guard();resolved={**spec,**{k:str(inputs/spec[k]) for k in ('path','checkpoint_path','average_path')}}
                    source,visits=load_saved(resolved,Path('/'),guard,expected_schema=spec['abstraction']) if readout=='current' else open_average(resolved,guard)
                    if source.description['iteration']!=spec['iteration']:raise ValueError('Pinned model iteration differs')
                    for task in [r for r in summary['tasks'] if r['cell']==spec['cell'] and r['readout']==readout]:
                        filename=task['file'];stage=task['stage'];seen=set()
                        if stage not in ('primary','readout') or (stage=='primary' and readout!='current'):raise ValueError('Unexpected stage')
                        with gzip.GzipFile(fileobj=tar.extractfile(prefix+'evaluation/'+filename)) as stream:
                            for line in stream:
                                guard();row=json.loads(line);panel=definitions[row['panel']];block=row['block'];rot=row['rotation'];coordinate=(row['panel'],block,rot)
                                blocks=panel['blocks'] if stage=='primary' else panel['readout_blocks'];root_seed=panel['primary_root'] if stage=='primary' else panel['readout_root']
                                expected_hash=spec['sha256'] if readout=='current' else spec['average_sha256']
                                if (coordinate in seen or not 0<=block<blocks or rot not in (0,1) or row['cell']!=spec['cell'] or row['readout']!=readout
                                        or row['stage']!=stage or row['seed']!=spec['seed'] or row['policy']!=spec['name'] or row['policy_sha256']!=expected_hash
                                        or row['checkpoint_sha256']!=spec['checkpoint_sha256'] or row['status']!='complete' or not row['native_replay_verified']
                                        or row['root_seed']!=root_seed or row['button']!=block%2 or row['deal_seed']!=stream_seed(root_seed,'test','deal',2,block)):
                                    raise ValueError('Unexpected raw coordinate/model/deal')
                                seen.add(coordinate);paired=(stage,*coordinate);law=(row['deal_seed'],row['button'],row['root_seed'])
                                if coupling.setdefault(paired,law)!=law:raise ValueError('Unpaired saved lineages/readouts')
                                replay_saved(row,source,visits,spec,readout,guard)
                                role='button' if rot==row['button'] else 'big_blind';key=(stage,row['panel'],spec['seed'],spec['cell'],readout)
                                series.setdefault(key,{}).setdefault(block,{})[role]=row['target_chips']
                                for position in ('overall',role):
                                    group=(*key,position);counts[group].update(row['tails']['counts']);counts[group]['large_calls']+=row['large_calls'];counts[group]['allin_calls']+=row['allin_calls']
                                    part=row['tails']['first_large_raise_response'];parts[(*group,part)]['hands']+=1;parts[(*group,part)]['target_chips']+=row['target_chips']
                                    for action in row['actions']:
                                        observed=action['observation']
                                        if action['logical_player']==0:
                                            c=coverage[(*group,observed['street'])];v=observed['visits'];c.update(decisions=1,zero=v==0,below10=v<10,below100=v<100)
                                            c[observed['readout_mass_status']]+=1
                                        if 'lbr' in action:
                                            telemetry=action['lbr'];c=counts[group];c.update(lbr_decisions=1,lbr_completed=telemetry['completed'],lbr_limited=not telemetry['completed'],lbr_over_soft=telemetry['over_soft_budget'],lbr_samples=telemetry['samples'],lbr_zero_likelihood_events=telemetry['zero_likelihood_events'])
                                            if telemetry['requested_samples']!=plan['chance_samples']:raise ValueError('LBR K differs')
                                raw_hands+=1;raw_actions+=len(row['actions'])
                                if raw_hands%1024==0:write(out/'progress.json',{'hands':raw_hands,'seed':spec['seed'],'cell':spec['cell'],'readout':readout,'seconds':time.time()-started,'peak_rss_bytes':peak_rss()})
                        expected=sum(2*(p['blocks'] if stage=='primary' else p['readout_blocks']) for p in plan['panels'])
                        if len(seen)!=expected or task['hands']!=expected:raise ValueError('Raw task count differs')
                    saved=read(f"evaluation/river-{spec['cell']}-{spec['seed']}-{readout}.json")
                    if saved['law']!=LAW or saved['projection']!=PROJECTION or len(saved['records'])!=3:raise ValueError('Common river law differs')
                    for case,item in zip(plan['river_roots'],saved['records'],strict=True):
                        guard();history=river_root_history(root_hand(case['seed'],case['raise_to']).events);ranges=common_ranges(history)
                        game=CommonRiverGame(history,ranges,raise_cap=plan['river_raise_cap'],max_public_nodes=plan['max_public_nodes'])
                        profile,telemetry=policy_profile(game,source,guard);quality=profile_quality(game,profile)
                        if (item['case']!=case or item['range_sha256']!=digest(ranges) or item['public_nodes']!=len(game.nodes)
                                or telemetry!=item['projection_telemetry'] or any(not isclose(quality[k],item['quality'][k],rel_tol=1e-10,abs_tol=1e-10) if isinstance(quality[k],float) else any(not isclose(a,b,rel_tol=1e-10,abs_tol=1e-10) for a,b in zip(quality[k],item['quality'][k],strict=True)) for k in quality)):
                            raise ValueError('Independent restricted profile/BR differs')
                        river.append({'seed':spec['seed'],'cell':spec['cell'],'readout':readout,'root':case['seed'],'raise_to':case['raise_to'],'range_sha256':item['range_sha256'],'public_nodes':len(game.nodes),**telemetry,**{k:v for k,v in quality.items() if not isinstance(v,list)},'profile_value_seat0':quality['profile_values_bb'][0],'profile_value_seat1':quality['profile_values_bb'][1],'br_gain_seat0':quality['best_response_gains_bb'][0],'br_gain_seat1':quality['best_response_gains_bb'][1]})
                        quality_audits+=1;del game,profile;gc.collect()
                    del source,visits;gc.collect()
    if raw_hands!=165888 or quality_audits!=36:raise ValueError('Frozen total work incomplete')
    absolute=[];changes=[]
    contrasts={'C_current_minus_A_current':[('C','current',1),('A','current',-1)],'C_average_minus_A_average':[('C','average',1),('A','average',-1)],'A_average_minus_current':[('A','average',1),('A','current',-1)],'C_average_minus_current':[('C','average',1),('C','current',-1)]}
    for stage in ('primary','readout'):
        for panel in definitions:
            for seeds in (SEEDS,*[(s,) for s in SEEDS]):
                seed='aggregate' if len(seeds)==3 else seeds[0]
                for role in ('overall',*ROLES):
                    for cell in ('A','C'):
                        for readout in (('current',) if stage=='primary' else ('current','average')):
                            absolute.append({'stage':stage,'panel':panel,'seed':seed,'cell':cell,'readout':readout,'position':role,**interval(block_values(series,stage,panel,[(cell,readout,1)],seeds,role))})
                    for name,terms in contrasts.items():
                        if stage=='primary' and name!='C_current_minus_A_current':continue
                        changes.append({'stage':stage,'panel':panel,'seed':seed,'contrast':name,'position':role,**interval(block_values(series,stage,panel,terms,seeds,role))})
    tail_rows=[dict(zip(('stage','panel','seed','cell','readout','position'),key),**value) for key,value in sorted(counts.items())]
    visit_rows=[dict(zip(('stage','panel','seed','cell','readout','position','street'),key),**value,zero_fraction=value['zero']/value['decisions'],below10_fraction=value['below10']/value['decisions'],below100_fraction=value['below100']/value['decisions']) for key,value in sorted(coverage.items())]
    partitions=[dict(zip(('stage','panel','seed','cell','readout','position','first_large_response'),key),**value) for key,value in sorted(parts.items())]
    for name,rows in [('absolute',absolute),('paired-changes',changes),('tails',tail_rows),('decision-visits',visit_rows),('whole-hand-partitions',partitions),('restricted-river',river)]:csv_rows(out/(name+'.csv'),rows)
    result={'status':'verified','hands':raw_hands,'actions':raw_actions,'restricted_quality_records':quality_audits,'native_replay_and_saved_policy_audited':True,'plan_sha256':plan_sha,'source':'6ce14e1513ce1b6f452b59930bf8f2034ad80115','archives':archives,'rental_closeout':final,'seconds':time.time()-started,'peak_rss_bytes':peak_rss(),'swap_growth_bytes':swap_bytes()-before,'primary':next(x for x in changes if x['stage']=='primary' and x['panel']=='lbr' and x['seed']=='aggregate' and x['position']=='overall'),'absolute':absolute,'changes':changes,'tails':tail_rows,'visits':visit_rows,'partitions':partitions,'river':river,'interpretation':'Primary95% paired block interval conditional on saved three lineages; secondary unadjusted exploratory. Restricted river quality is not full game exploitability; no unique causal mechanism, promotion or factorial interaction.'}
    write(out/'summary.json',result);write(out/'manifest.json',{'files':{p.name:{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=report(a.root,json.loads(a.plan.read_text()),a.inputs,a.out)
    print(json.dumps({k:result[k] for k in ('status','hands','restricted_quality_records','seconds','primary')}),flush=True)
