"""Train one fresh cap A/B arm to a frozen common node budget, with atomic milestones."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from time import perf_counter, time

from scripts.hu20_reopening_common import independent_density
from scripts.preflight_hu20_reopening import configuration
from scripts.tp20_common import append, guard, interruptible, seal
from scripts.train_hu20 import rss, system, write_json
from src.arena.schedule import digest
from src.blueprint.artifact import export_policy, save_training
from src.blueprint.solver import BlueprintTrainer
from src.game.hand import Table


def run(plan,seed,arm,out,deadline):
    if out.exists():raise FileExistsError(out)
    if seed not in plan['training_seeds'] or plan['training_nodes'] not in plan['training_nodes_options']:
        raise ValueError('Training requires the frozen fresh-seed/work choice')
    out.mkdir(parents=True);interruptible();started=time()
    config=configuration(plan,seed,arm)
    trainer=BlueprintTrainer(Table(('player-0','player-1'),(2000,2000)),config)
    result={'status':'incomplete','seed':seed,'arm':arm,'initial_entries':0,'initial_iteration':0,
            'config':asdict(config),'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'plan_digest':digest(plan),'started':started,'requested_nodes':plan['training_nodes'],
            'milestones':[],'failure':None,'swap_before':system(['sysctl','vm.swapusage'])}
    write_json(out/'manifest.json',result)
    total=0;marker=0;last_saved=0;updates=Counter();new=Counter();revisited=Counter()
    targets=[int(plan['training_nodes']*f) for f in plan['checkpoint_fractions']]
    save_training(trainer,out/'last-completed.json.gz')
    try:
        while total<plan['training_nodes']:
            guard(plan,out,deadline)
            t=perf_counter();report=trainer.step(cancelled=lambda:time()>=deadline)
            total+=report.nodes;updates.update(report.traverser_visits_by_street)
            new.update(report.new_entries_by_street);revisited.update(report.revisited_keys_by_street)
            append(out/'iterations.jsonl',{**asdict(report),'updated_keys':None,
                'complete_outer_seconds':perf_counter()-t,'completed_nodes':total,'rss_bytes':rss()})
            while marker<len(targets) and total>=targets[marker]:
                guard(plan,out,deadline);t=perf_counter()
                cp=out/f'checkpoint-{marker}.json.gz';policy=out/f'current-{marker}.json.gz'
                row={'index':marker,'requested_nodes':targets[marker],'completed_nodes':total,
                     'overshoot_nodes':total-targets[marker],'iteration':trainer.iteration,
                     'checkpoint_sha256':save_training(trainer,cp),
                     'policy_sha256':export_policy(trainer,policy),'entries':len(trainer.nodes)}
                row['checkpoint_export_seconds']=perf_counter()-t;t=perf_counter()
                row['independent']=independent_density(trainer,Path(plan['independent_path']))
                row['independent_seconds']=perf_counter()-t
                append(out/'milestones.jsonl',row);result['milestones'].append(row)
                write_json(out/'progress.json',{**result,'completed_nodes':total,'iteration':trainer.iteration})
                marker+=1
            if total>=last_saved+1000000:
                t=perf_counter();h=save_training(trainer,out/'last-completed.json.gz');last_saved=total
                append(out/'checkpoint-overhead.jsonl',{'nodes':total,'iteration':trainer.iteration,'sha256':h,'seconds':perf_counter()-t})
        if marker!=4:raise ValueError('Missing fixed milestone')
        result['status']='complete'
    except Exception as exc:
        result.update(failure=f'{type(exc).__name__}: {exc}',discarded_nodes=trainer.last_attempt_nodes,
                      discarded_work=trainer.last_attempt_work,failed_iteration=trainer.iteration+1)
    result['last_checkpoint_sha256']=save_training(trainer,out/'last-completed.json.gz')
    result.update(completed_nodes=total,overshoot_nodes=max(0,total-plan['training_nodes']),
        iterations=trainer.iteration,entries=len(trainer.nodes),updates_by_street=dict(updates),
        new_entries_by_street=dict(new),revisited_keys_by_street=dict(revisited),
        elapsed_seconds=time()-started,peak_rss_bytes=rss(),swap_after=system(['sysctl','vm.swapusage']))
    write_json(out/'result.json',result);seal(out);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--arm',choices=('A','B'),required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();r=run(json.loads(a.plan.read_text()),a.seed,a.arm,a.out,a.deadline)
    print(json.dumps(r,sort_keys=True));return r['status']!='complete'

if __name__=='__main__':raise SystemExit(main())
