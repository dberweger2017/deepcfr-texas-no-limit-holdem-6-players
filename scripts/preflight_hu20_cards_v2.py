"""From-zero representation resource prefix; unchanged single-worker trainer."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from time import monotonic,perf_counter

from scripts.hu20_platform_pilot import environment,write,fingerprint
from src.blueprint.artifact import save_training,export_policy
from src.blueprint.hu20_river import peak_rss
from src.blueprint.solver import BlueprintTrainer,PilotConfig
from src.diagnostics.saved_hu20 import file_hash
from src.game.hand import Table


def run(plan,out):
    out.mkdir(parents=True,exist_ok=False);start=perf_counter();deadline=monotonic()+plan['max_seconds']
    trainer=BlueprintTrainer(Table(('player-0','player-1'),(2000,2000)),PilotConfig(**plan['config']))
    completed=0;training_seconds=0;new_by_street=Counter();visits_by_street=Counter();coverage=Counter();milestones=[];failure=None
    env=environment()
    write(out/'attempt.json',{'plan':plan,'environment':env,'from_zero':True})
    try:
        with (out/'iterations.jsonl').open('x') as log:
            for target in plan['milestones']:
                while completed<target:
                    if monotonic()>=deadline or peak_rss()>plan['max_rss_gib']*2**30 or shutil.disk_usage(out).free<plan['min_free_gib']*2**30:
                        raise RuntimeError('Resource preflight time/RSS/disk guard')
                    before=perf_counter();r=trainer.step(workers=1,cancelled=lambda:monotonic()>=deadline);training_seconds+=perf_counter()-before
                    completed+=r.nodes;new_by_street.update(r.new_entries_by_street);visits_by_street.update(r.traverser_visits_by_street);coverage.update(r.coverage)
                    data=asdict(r);data.pop('updated_keys');data['completed_nodes']=completed;log.write(json.dumps(data,sort_keys=True)+'\n')
                record={'target':target,'nodes':completed,'iteration':trainer.iteration,'entries':len(trainer.nodes),
                    'new_entries_by_street':dict(new_by_street),'visits_by_street':dict(visits_by_street),'coverage':dict(coverage),
                    'training_seconds':training_seconds,'nodes_per_second':completed/training_seconds,
                    'peak_rss_bytes':peak_rss(),'visits':sum(n.visits for n in trainer.nodes.values()),
                    'mean_visits_per_key':sum(n.visits for n in trainer.nodes.values())/len(trainer.nodes)}
                milestones.append(record);write(out/'progress.json',record);print(json.dumps(record),flush=True)
        before=perf_counter();save_training(trainer,out/'checkpoint.json.gz');checkpoint_seconds=perf_counter()-before
        before=perf_counter();export_policy(trainer,out/'current.json.gz');export_seconds=perf_counter()-before
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}';save_training(trainer,out/'partial.json.gz')
    result={'status':'failed' if failure else 'complete','failure':failure,'plan':plan,'environment':env,
        'milestones':milestones,'nodes':completed,'entries':len(trainer.nodes),'seconds':perf_counter()-start,'peak_rss_bytes':peak_rss()}
    if not failure:result.update(checkpoint_seconds=checkpoint_seconds,export_seconds=export_seconds,
        checkpoint=fingerprint(out/'checkpoint.json.gz'),current=fingerprint(out/'current.json.gz'))
    write(out/'summary.json',result)
    write(out/'manifest.json',{p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in out.iterdir()})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=run(json.loads(a.plan.read_text()),a.out);print(json.dumps({k:r[k] for k in ('status','failure','nodes','entries','seconds','peak_rss_bytes')}))
    if r['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
