"""Frozen from-zero v2 lineage; complete iterations and streaming artifact hashes."""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil
from time import perf_counter, time

from scripts.hu20_platform_pilot import canonical, environment, write
from src.blueprint.artifact import export_policy, save_training
from src.blueprint.hu20_river import peak_rss
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.diagnostics.saved_hu20 import file_hash
from src.game.hand import Table


def fingerprint(path):
    # Full exports can be gigabytes. Hash streams rather than materializing both
    # compressed and decompressed copies alongside the already resident table.
    h=sha256();size=0
    with gzip.open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk);size+=len(chunk)
    return {'sha256':file_hash(path),'bytes':path.stat().st_size,
            'uncompressed_sha256':h.hexdigest(),'uncompressed_bytes':size}


def run(plan,seed,out,deadline):
    if seed not in plan['seeds']:
        raise ValueError('Undeclared training seed')
    if file_hash(Path('src/blueprint/cards_v2.py'))!=plan['descriptor_sha256']:
        raise ValueError('Frozen descriptor changed')
    out.mkdir(parents=True,exist_ok=False)
    trainer=BlueprintTrainer(Table(('player-0','player-1'),(2000,2000)),PilotConfig(**plan['config'],seed=seed))
    completed=0;seconds=0;coverage=Counter();created=Counter();visits=Counter();started=time()
    result={'status':'running','environment':environment(),'plan_sha256':sha256(canonical(plan)).hexdigest(),
            'seed':seed,'from_zero':True,'checkpoint_milestones':[],'config':asdict(trainer.config)}
    write(out/'attempt.json',result)
    def guard():
        if time()>=deadline:raise TimeoutError('Frozen workload cutoff')
        if peak_rss()>plan['limits']['max_owned_rss_gib']*2**30:raise MemoryError('Frozen RSS ceiling')
        if shutil.disk_usage(out).free<plan['limits']['min_free_disk_gib']*2**30:raise OSError('Frozen disk floor')
    def progress():
        return {'completed_nodes':completed,'iteration':trainer.iteration,'entries':len(trainer.nodes),
                'training_seconds':seconds,'nodes_per_second':completed/seconds if seconds else None,
                'peak_rss_bytes':peak_rss(),'coverage':dict(coverage),'entries_by_street':dict(created),
                'traverser_visits_by_street':dict(visits),'time':time()}
    target=plan['nodes_per_seed'];cadence=plan['checkpoint_every_nodes'];next_save=cadence
    try:
        with (out/'iterations.jsonl').open('x') as log:
            while completed<target:
                guard();begin=perf_counter()
                r=trainer.step(workers=1,cancelled=lambda:time()>=deadline)
                seconds+=perf_counter()-begin;completed+=r.nodes
                coverage.update(r.coverage);created.update(r.new_entries_by_street);visits.update(r.traverser_visits_by_street)
                row=asdict(r);row.pop('updated_keys');row['completed_nodes']=completed
                log.write(json.dumps(row,sort_keys=True)+'\n')
                if trainer.iteration%100==0:log.flush();write(out/'progress.json',progress())
                if completed>=next_save or completed>=target:
                    guard();path=out/f'checkpoint-{next_save}.json.gz';save_started=time();before=perf_counter();save_training(trainer,path)
                    save_seconds=perf_counter()-before;save_finished=time();before=perf_counter();artifact=fingerprint(path)
                    record={**progress(),'checkpoint':artifact,'path':path.name,'save_seconds':save_seconds,
                            'save_started':save_started,'save_finished':save_finished,'hash_seconds':perf_counter()-before}
                    result['checkpoint_milestones'].append(record);write(out/'checkpoint-milestones.json',result['checkpoint_milestones'])
                    next_save+=cadence
        guard();result['export_started']=time();before=perf_counter();export_policy(trainer,out/'current.json.gz')
        result['export_seconds']=perf_counter()-before;result['export_finished']=time()
        result.update(status='complete',**progress(),overshoot_nodes=completed-target,
                      current=fingerprint(out/'current.json.gz'),visits=sum(n.visits for n in trainer.nodes.values()),
                      mean_visits_per_key=sum(n.visits for n in trainer.nodes.values())/len(trainer.nodes))
    except Exception as exc:
        result.update(status='failed',failure=f'{type(exc).__name__}: {exc}',**progress(),
                      discarded_nodes=trainer.last_attempt_nodes,discarded_work=trainer.last_attempt_work)
        # Retain the last complete in-memory state if headroom permits. Previously
        # committed milestone files remain immutable even when this save fails.
        try:
            if time()<deadline and shutil.disk_usage(out).free>=plan['limits']['min_free_disk_gib']*2**30:
                save_training(trainer,out/'partial.json.gz');result['partial']=fingerprint(out/'partial.json.gz')
        except Exception as error:result['partial_save_error']=type(error).__name__
    finally:
        result.update(wall_seconds=time()-started,peak_rss_bytes=peak_rss());write(out/'summary.json',result)
        write(out/'manifest.json',{p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in out.iterdir() if p.is_file()})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    r=run(json.loads(a.plan.read_text()),a.seed,a.out,a.deadline)
    print(json.dumps({k:r[k] for k in ('status','completed_nodes','entries','wall_seconds','peak_rss_bytes')}),flush=True)
    raise SystemExit(r['status']!='complete')
