"""Frozen v1/v2 paired hands through the existing evaluator and native replay."""
import argparse
from collections import Counter
import gc
import gzip
import json
from pathlib import Path
from time import perf_counter,time

from scripts.evaluate_hu20_stackoff import opponent
from scripts.evaluate_robustness import play
from scripts.hu20_platform_pilot import write
from scripts.play_robustness import replay_row
from scripts.train_hu20_cards_v2 import fingerprint
from src.arena.report import estimate
from src.arena.schedule import stream_seed
from src.blueprint.abstraction import HU20_CARD_V2_SCHEMA,HU20_UNCAPPED_SCHEMA
from src.blueprint.artifact import HU20_UNCAPPED_FORMAT
from src.diagnostics.cached_lbr import SharedProbabilityCache
from src.diagnostics.exact_ranker import RankedCachedLocalBestResponse
from src.diagnostics.robustness import LBRConfig
from src.diagnostics.saved_hu20 import load_saved,file_hash
from src.diagnostics.stackoff_tails import RecordingOpponent,RecordingTarget,attach_snapshots,hand_tails


def hand(source,visits,spec,panel,block,rotation,evaluation,resource_only=False,failure_sink=None):
    config=LBRConfig(evaluation['chance_samples'],evaluation['lbr_seconds'])
    rival=(RankedCachedLocalBestResponse(source,stream_seed(panel['root'],'test','opponent',2,block,1),
                 SharedProbabilityCache(source),config) if panel['rule']=='lbr'
           else opponent(panel,source,block,evaluation))
    observations=[];rows=[]
    target=RecordingTarget(source,visits,observations);wrapped=RecordingOpponent(rival,observations)
    def emit(row):
        row.update(panel=panel['name'],version=spec['version'],seed=spec['seed'])
        if row['status']!='complete' and failure_sink is not None:
            attach_snapshots(row,observations[:len(row['actions'])])
            failure_sink(row)
            rows.append(row)
            return
        if not resource_only:
            attach_snapshots(row,observations[:len(row['actions'])]);replay_row(row)
            row['native_replay_verified']=True;row['tails']=hand_tails(row)
        rows.append(row)
    play(target,spec,(panel['rule'],),panel['contract'],block,rotation,panel['root'],'card-v2-ab-v1',
         config,emit,resource_only=resource_only,opponent_policies={1:wrapped})
    return rows[0]


def specs(plan,seed,training,baseline):
    original=next(s for s in baseline['models'] if s['seed']==seed and s['milestone']==100000000)
    original={**original,'version':'v1','abstraction':HU20_UNCAPPED_SCHEMA}
    summary=json.loads((training/'summary.json').read_text())
    if summary['status']!='complete' or summary['completed_nodes']<plan['nodes_per_seed']:
        raise ValueError('Incomplete training cannot enter the final comparison')
    cp=summary['checkpoint_milestones'][-1]
    new={'name':f'v2-B-{seed}-100M','seed':seed,'milestone':100000000,'players':2,'version':'v2',
         'abstraction':HU20_CARD_V2_SCHEMA,'format':HU20_UNCAPPED_FORMAT,
         'path':'current.json.gz','sha256':summary['current']['sha256'],
         'checkpoint_path':cp['path'],'checkpoint_sha256':cp['checkpoint']['sha256']}
    return [(original,Path('/workspace/baseline')),(new,training)]


def evaluate(plan,seed,training,baseline,out,deadline):
    out.mkdir(parents=True,exist_ok=False);e=plan['evaluation'];started=time();timings=[];status='complete'
    model_specs=specs(plan,seed,training,baseline);admitted=[dict(p) for p in e['panels']]
    # Outcome-blind pilot first. Its validation deals are disjoint; it returns
    # no chip outcomes and cannot select models, actions or hand counts.
    for spec,inputs in model_specs:
        source,visits=load_saved(spec,inputs,expected_schema=spec['abstraction'])
        for index,p in enumerate(admitted):
            times=[];panel={**p,'root':e['timing_root']+index}
            for block in range(e['timing_blocks']):
                for rotation in (0,1):
                    begin=perf_counter();row=hand(source,visits,spec,panel,block,rotation,e,True)
                    if row['target_chips'] is not None:raise ValueError('Timing pilot exposed outcomes')
                    times.append(perf_counter()-begin)
            timings.append({'version':spec['version'],'panel':p['name'],'hands':len(times),'max_seconds':max(times),'seconds':sum(times)})
        del source,visits;gc.collect()
    projection={p['name']:sum(t['max_seconds'] for t in timings if t['panel']==p['name'])*2*p['blocks']*e['projection_multiplier'] for p in admitted}
    cheap=sum(v for k,v in projection.items() if k!='lbr')
    include_lbr=cheap+projection['lbr']<=e['maximum_seconds_per_seed']
    admission={'timings':timings,'projection_seconds':projection,'include_lbr':include_lbr,
               'rule':'1.25 × sum of per-arm maximum hand times × paired hand count <=1800s/seed',
               'outcome_blind':True,'cheap_projection_seconds':cheap}
    write(out/'admission.json',admission)
    if cheap>e['maximum_seconds_per_seed']:raise RuntimeError('Frozen cheap-panel runtime gate failed; do not shrink hands')
    admitted=[p for p in admitted if p['name']!='lbr' or include_lbr]
    actual_deadline=min(deadline,time()+e['maximum_seconds_per_seed']);counts=Counter();tails={};returns={}
    with gzip.open(out/'hands.jsonl.gz','wt') as handle:
        try:
            for spec,inputs in model_specs:
                source,visits=load_saved(spec,inputs,expected_schema=spec['abstraction'])
                write(out/(spec['version']+'-input.json'),{'spec':spec,'description':source.description,
                      'entries':len(visits),'visits':sum(visits.values()),'mean_visits_per_key':sum(visits.values())/len(visits)})
                for index,p in enumerate(e['panels']):
                    if p not in admitted:continue
                    panel={**p,'root':e['root']+index}
                    key=(spec['version'],p['name']);tails[key]=Counter();returns[key]={}
                    for block in range(p['blocks']):
                        for rotation in (0,1):
                            if time()>=actual_deadline:raise TimeoutError('Frozen evaluation cutoff')
                            row=hand(source,visits,spec,panel,block,rotation,e)
                            handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');handle.flush()
                            tails[key].update(row['tails']['counts'])
                            group=row['tails']['first_large_raise_response']
                            tails[key]['first_large_'+group+'_hands']+=1
                            tails[key]['first_large_'+group+'_whole_hand_chips']+=row['target_chips']
                            returns[key][(block,rotation)]=row['target_chips'];counts[spec['version']]+=1
                        write(out/'progress.json',{'version':spec['version'],'panel':p['name'],'block':block,'hands':dict(counts),'time':time()})
                del source,visits;gc.collect()
        except Exception as exc:
            status='failed';failure=f'{type(exc).__name__}: {exc}'
    summary={'status':status,'seed':seed,'hands':dict(counts),'wall_seconds':time()-started,
             'admission':admission,'panels':[],'failure':locals().get('failure')}
    if status=='complete':
        for p in admitted:
            a=returns[('v1',p['name'])];b=returns[('v2',p['name'])];blocks=p['blocks']
            row={'panel':p['name'],'blocks':blocks,'hands_per_arm':blocks*2,'per_arm':{},'positions':{}}
            for version,data in [('v1',a),('v2',b)]:
                row['per_arm'][version]={'bb_per_100':estimate([(data[(i,0)]+data[(i,1)])/2 for i in range(blocks)]),
                                        'tails':dict(tails[(version,p['name'])])}
            row['paired_v2_minus_v1_bb_per_100']=estimate([sum(b[(i,r)]-a[(i,r)] for r in (0,1))/2 for i in range(blocks)])
            for position in ('button','big_blind'):
                coordinates=[(i,i%2 if position=='button' else 1-i%2) for i in range(blocks)]
                row['positions'][position]={'paired_v2_minus_v1_bb_per_100':estimate([b[c]-a[c] for c in coordinates]),
                                           'v1':estimate([a[c] for c in coordinates]),'v2':estimate([b[c] for c in coordinates])}
            summary['panels'].append(row)
    write(out/'summary.json',summary);write(out/'manifest.json',{p.name:fingerprint(p) if p.name.endswith('.gz') else
        {'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in out.iterdir() if p.is_file()})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','training','baseline-plan','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    result=evaluate(json.loads(a.plan.read_text()),a.seed,a.training,json.loads(a.baseline_plan.read_text()),a.out,a.deadline)
    print(json.dumps({'status':result['status'],'hands':result['hands']}),flush=True);raise SystemExit(result['status']!='complete')
