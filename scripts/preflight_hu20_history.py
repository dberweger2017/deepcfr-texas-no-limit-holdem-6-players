"""Unpaid from-zero history-density comparison; no poker strength outcomes."""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import resource
import shutil
import subprocess
import sys
from time import monotonic

from scripts.hu20_platform_pilot import environment

from src.blueprint.abstraction import choices, information_key
from src.blueprint.artifact import export_policy, save_training
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Hand, Table


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(canonical(value)+b'\n');temp.replace(path)


def file_hash(path):
    h=sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform=='darwin' else value*1024


def distribution(hist):
    count=sum(hist.values())
    if not count:return {'count':0,'median':None,'fraction_zero':None,'fraction_lt10':None,'fraction_lt100':None,'histogram':{}}
    rank=(count-1)//2;total=0;median=None
    for visits,n in sorted(hist.items()):
        total+=n
        if total>rank:median=visits;break
    return {'count':count,'median':median,'fraction_zero':hist.get(0,0)/count,
            'fraction_lt10':sum(n for v,n in hist.items() if v<10)/count,
            'fraction_lt100':sum(n for v,n in hist.items() if v<100)/count,
            'histogram':{str(v):n for v,n in sorted(hist.items())}}


def corpus(plan,out):
    out.mkdir(parents=True,exist_ok=False)
    rows=[];digest=sha256();traces=sha256()
    for block in range(plan['probe']['blocks']):
        for button in (0,1):
            label=f"dr2x2-probe/{plan['probe']['root_seed']}/{block}/{button}"
            deal=int.from_bytes(sha256((label+'/deal').encode()).digest()[:8],'big')
            rng=Random(int.from_bytes(sha256((label+'/actions').encode()).digest()[:8],'big'))
            hand=Hand.start(Table(('player-0','player-1'),(2000,2000),button),hand_id=label,seed=deal)
            while not hand.finished:
                view=hand.observe(hand.actor);menu=choices(view,raise_cap=None,free_fold=False)
                row={'block':block,'button':button,'actor':view.actor,'street':view.street.value,
                     'keys':{name:information_key(view,menu,schema=schema) for name,schema in plan['schemas'].items()},
                     'menu':[item.name for item in menu]}
                rows.append(row);digest.update(canonical(row)+b'\n')
                action=menu[rng.randrange(len(menu))].action
                traces.update(canonical({'hand':label,'street':view.street.value,'actor':view.actor,'action':asdict(action)})+b'\n')
                hand=hand.apply(action)
    with gzip.GzipFile(filename='',fileobj=(out/'decisions.jsonl.gz').open('wb'),mode='wb',mtime=0) as target:
        for row in rows:target.write(canonical(row)+b'\n')
    result={'plan_sha256':sha256(canonical(plan)).hexdigest(),'decisions':len(rows),'counts_by_street':dict(Counter(r['street'] for r in rows)),
            'decisions_sha256':file_hash(out/'decisions.jsonl.gz'),'canonical_rows_sha256':digest.hexdigest(),
            'native_action_trace_sha256':traces.hexdigest(),'strength_outcomes_inspected':False}
    write(out/'manifest.json',result);return result


def snapshot(trainer,streets,probes,cell):
    stored={street:Counter() for street in ('preflop','flop','turn','river')}
    for key,node in trainer.nodes.items():stored[streets[key]][node.visits]+=1
    encountered={street:Counter() for street in stored};unique={street:{} for street in stored}
    for row in probes:
        key=row['keys'][cell];node=trainer.nodes.get(key);visits=node.visits if node else 0
        encountered[row['street']][visits]+=1;unique[row['street']][key]=visits
    return {'stored':{s:distribution(h) for s,h in stored.items()},
            'common_encounters':{s:distribution(h) for s,h in encountered.items()},
            'common_unique_keys':{s:distribution(Counter(keys.values())) for s,keys in unique.items()}}


def worker(plan,seed,cell,probe_dir,out):
    import src.blueprint.solver as solver
    out.mkdir(parents=True,exist_ok=False);start=monotonic();deadline=start+plan['limits']['worker_seconds']
    runtime=environment()
    engine=json.loads(runtime['engine_origin'])['vcs_info']['commit_id']
    if sys.version_info[:3]!=(3,11,14) or engine!='5db20e3d5d6862b32a7402035c1340b622d3b005':
        raise ValueError('Worker requires pinned Python 3.11.14 and engine5db20e3')
    probe_manifest=json.loads((probe_dir/'manifest.json').read_text())
    if file_hash(probe_dir/'decisions.jsonl.gz')!=probe_manifest['decisions_sha256']:raise ValueError('Probe hash changed')
    if sha256(canonical(plan)).hexdigest()!=probe_manifest['plan_sha256']:raise ValueError('Probe plan changed')
    with gzip.open(probe_dir/'decisions.jsonl.gz','rt') as source:probes=[json.loads(line) for line in source]
    streets={};original=solver.information_key
    def record(view,menu,**kwargs):
        key=original(view,menu,**kwargs)
        street=streets.setdefault(key,view.street.value)
        if street!=view.street.value:raise ValueError('Information key aliases streets')
        return key
    solver.information_key=record
    config=PilotConfig(**plan['trainer'],seed=seed,abstraction=plan['schemas'][cell])
    trainer=BlueprintTrainer(Table(('player-0','player-1'),(2000,2000)),config)
    result={'status':'running','seed':seed,'cell':cell,'config':asdict(config),'plan_sha256':sha256(canonical(plan)).hexdigest(),
            'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'environment':runtime,'milestones':[],'failure':None,'strength_outcomes_inspected':False}
    write(out/'attempt.json',result);total=0;seconds=0;index=0
    try:
        with (out/'iterations.jsonl').open('w') as log:
            while total<plan['milestones'][-1]:
                if monotonic()>=deadline or rss()>plan['limits']['max_rss_gib']*2**30 or shutil.disk_usage(out).free<plan['limits']['min_free_gib']*2**30:
                    raise RuntimeError('Worker time/RSS/disk guard')
                step=trainer.step(workers=1,cancelled=lambda:monotonic()>=deadline)
                total+=step.nodes;seconds+=step.elapsed_seconds
                row=asdict(step);row.pop('updated_keys');row['completed_nodes']=total
                log.write(json.dumps(row,sort_keys=True)+'\n')
                if total>=plan['milestones'][index]:
                    target=plan['milestones'][index];t=monotonic();cp=out/f'checkpoint-{target}.json.gz';h=save_training(trainer,cp);save_s=monotonic()-t
                    density=snapshot(trainer,streets,probes,cell)
                    result['milestones'].append({'requested_nodes':target,'completed_nodes':total,'overshoot':total-target,'iteration':trainer.iteration,
                        'entries':len(trainer.nodes),'training_seconds':seconds,'nodes_per_training_second':total/seconds,'wall_seconds':monotonic()-start,
                        'peak_rss_bytes':rss(),'checkpoint_sha256':h,'checkpoint_bytes':cp.stat().st_size,'save_seconds':save_s,'density':density})
                    index+=1;write(out/'progress.json',result)
            t=monotonic();exp=out/'current.json.gz';h=export_policy(trainer,exp)
            result.update(status='complete',export_sha256=h,export_bytes=exp.stat().st_size,export_seconds=monotonic()-t)
    except Exception as exc:
        result.update(status='failed',failure=f'{type(exc).__name__}: {exc}',discarded_work=trainer.last_attempt_work,
                      partial_checkpoint_sha256=save_training(trainer,out/'partial-last-completed.json.gz'))
    finally:solver.information_key=original
    result.update(completed_nodes=total,iterations=trainer.iteration,peak_rss_bytes=rss(),wall_seconds=monotonic()-start)
    write(out/'result.json',result);return result


def passes(full,compressed,gate):
    if full['count']==0 or compressed['count']!=full['count']:return False
    reduction=gate['low_visit_fraction_reduction_pp']/100
    return (compressed['median']>=max(gate['minimum_median'],gate['median_ratio']*full['median'])
            and all(full[field]-compressed[field]>=reduction for field in ('fraction_lt10','fraction_lt100')))


def report(plan,root):
    rows=[];pooled={cell:Counter() for cell in plan['schemas']};all_complete=True
    for seed in plan['seeds']:
        pair={}
        for cell in plan['schemas']:
            result=json.loads((root/f'{cell}-{seed}'/'result.json').read_text());all_complete &= result['status']=='complete'
            if result['status']=='complete':
                pair[cell]=result['milestones'][-1]['density']['common_encounters']['river']
                pooled[cell].update({int(v):n for v,n in pair[cell]['histogram'].items()})
        rows.append({'seed':seed,'passed':len(pair)==2 and passes(pair['full'],pair['compressed'],plan['gate']),'river':pair})
    aggregate={cell:distribution(hist) for cell,hist in pooled.items()}
    passed=all_complete and sum(row['passed'] for row in rows)>=plan['gate']['minimum_passing_seeds'] and passes(aggregate['full'],aggregate['compressed'],plan['gate'])
    result={'status':'density-pass' if passed else 'density-fail','paid_training_authorized_by_density':passed,'all_workers_complete':all_complete,
            'seed_results':rows,'aggregate_river':aggregate,'gate':plan['gate'],'strength_outcomes_inspected':False,
            'limitations':'Density gate only; correctness/recovery/public-separation, D prefix, live quote and owner budget remain separate prerequisites.'}
    write(root/'density-gate.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('corpus','worker','report'));p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--probe',type=Path);p.add_argument('--seed',type=int);p.add_argument('--cell',choices=('full','compressed'))
    a=p.parse_args();plan=json.loads(a.plan.read_text())
    result=corpus(plan,a.out) if a.phase=='corpus' else report(plan,a.out) if a.phase=='report' else worker(plan,a.seed,a.cell,a.probe,a.out)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','failure','decisions','completed_nodes','peak_rss_bytes','wall_seconds')}));return result.get('status')=='failed'


if __name__=='__main__':raise SystemExit(main())
