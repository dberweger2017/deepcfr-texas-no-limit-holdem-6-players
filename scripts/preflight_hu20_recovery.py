"""Outcome-free M4 deployment, production milestone and deterministic resume check."""

import argparse
from dataclasses import asdict
import gc
import gzip
import json
from pathlib import Path
from time import perf_counter, time

from scripts.hu20_scaling_runtime import validate_inputs
from scripts.hu20_scaling_common import acquire, parent_trainer, specification, check
from scripts.evaluate_hu20_reopening import Target
from scripts.evaluate_robustness import play
from scripts.train_hu20_scaling import run
from scripts.train_hu20 import write_json, rss, system
from scripts.tp20_common import seal
from src.blueprint.artifact import load_training, save_training
from src.diagnostics.robustness import LBRConfig


def preflight(plan,out):
    acquire(out);started=time();deadline=plan['preflight_deadline'];before=system(['sysctl','vm.swapusage'])
    verified=validate_inputs(plan,parse=True)
    record={'started':started,'runtime_inputs':verified,'lineages':[],'timing_panels':[]}
    for seed in plan['training_seeds']:
        parent=plan['parents'][str(seed)];total=parent['completed_nodes']
        short={**plan,'milestones':([40000000] if seed==2026093001 else [])+[total+100000],
               'training_total_nodes':total+100000}
        t=perf_counter();r=run(short,parent,out/f'production-{seed}',deadline)
        if r['status']!='complete':raise ValueError('Production milestone preflight failed')
        trainer=parent_trainer(parent,plan['limits']['max_entries']);t=perf_counter()
        first=[asdict(trainer.step()) for _ in range(4)]
        elapsed=perf_counter()-t
        h1=save_training(trainer,out/f'next-original-{seed}.gz');del trainer;gc.collect()
        trainer=parent_trainer(parent,plan['limits']['max_entries'])
        second=[asdict(trainer.step()) for _ in range(4)]
        h2=save_training(trainer,out/f'next-reloaded-{seed}.gz')
        if h1!=h2 or [(x['iteration'],x['nodes']) for x in first]!=[(x['iteration'],x['nodes']) for x in second]:
            raise ValueError('Retained-state next-iteration determinism')
        del trainer;gc.collect()
        m=r['milestones'][-1];cp=Path(m['checkpoint_path']);policy=Path(m['policy_path'])
        spec=specification(seed,short['training_total_nodes'],r['completed_iterations'],cp,policy,m['checkpoint_sha256'],m['policy_sha256'])
        t=perf_counter();source=Target(spec);load_seconds=perf_counter()-t
        record['lineages'].append({'seed':seed,'production_result':r,'next_sha256':h1,
               'deterministic_replayed_nodes':sum(x['nodes'] for x in first+second),
               'next_four_seconds':elapsed,'next_four_nodes':sum(x['nodes'] for x in first),
               'verified_reload_seconds':load_seconds})
        if seed==plan['training_seeds'][0]:
            with gzip.open(out/'resource-hands.jsonl.gz','wt') as saved:
                for rule,contract in [('pressure','native'),('lbr','menu')]:
                    t=perf_counter();count=0
                    def emit(row):
                        nonlocal count
                        assert row['target_chips'] is None
                        saved.write(json.dumps(row,sort_keys=True)+'\n');count+=1
                    for b in range(8):
                        for rot in (0,1):
                            check(plan,out,deadline,before)
                            play(source,spec,(rule,),contract,b,rot,plan['recovery_preflight_root'],
                                 'recovery-resource-only',LBRConfig(4,5),emit,resource_only=True)
                    record['timing_panels'].append({'rule':rule,'hands':count,'seconds':perf_counter()-t})
        del source;gc.collect();check(plan,out,deadline,before)
    record.update(status='complete',finished=time(),peak_rss_bytes=rss(),swap_before=before,swap_after=system(['sysctl','vm.swapusage']))
    write_json(out/'result.json',record);seal(out);return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=preflight(json.loads(a.plan.read_text()),a.out);print(json.dumps({'status':r['status'],'lineages':len(r['lineages']),'seconds':r['finished']-r['started']}))
