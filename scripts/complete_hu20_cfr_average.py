"""Finish only unrecorded frozen coordinates after a terminated M1 worker.

Retain the truncated gzip and every completed record. This bounded repair uses
the original source commit time as a conservative, earlier-than-launch deadline.
It is not a general recovery or experiment extension framework.
"""
import argparse
import gc
import gzip
import json
from pathlib import Path
import resource
import shutil
import subprocess
from time import time,perf_counter

from scripts.evaluate_hu20_cfr_average import play,summarize
from src.arena.catalog import Checkpoint
from src.arena.schedule import digest
from src.blueprint.artifact import FrozenBlueprint,HU20_UNCAPPED_FORMAT
from src.diagnostics.cfr_average import DiagnosticAverage
from src.diagnostics.saved_hu20 import file_hash


def surviving_rows(path):
    rows=[];problem=None
    try:
        with gzip.open(path,'rt') as f:
            for line in f:
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:
                    problem='unterminated final JSON record';break
    except (EOFError,OSError) as exc:problem=str(exc)
    return rows,problem


def complete(plan,original,inputs,averages,out,original_source):
    for name in ('scripts/evaluate_hu20_cfr_average.py','src/diagnostics/cfr_average.py','src/blueprint/artifact.py','src/diagnostics/exact_ranker.py'):
        if Path(name).read_bytes()!=subprocess.check_output(['git','show',original_source+':'+name]):raise ValueError('Frozen gameplay source changed')
    anchor=int(subprocess.check_output(['git','show','-s','--format=%ct',original_source],text=True));deadline=anchor+plan['max_seconds']
    out.mkdir(parents=True,exist_ok=False);start=perf_counter();rows=[];receipts=[];loaded=[];new_hands=0;failure=None
    def guard():
        if time()>=deadline:raise TimeoutError('Original conservative execution window expired')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>6*1024**3:raise MemoryError('M1 process peak guard')
    try:
        guard()
        for spec in plan['models']:
            path=original/(spec['name']+'.hands.jsonl.gz');old,problem=surviving_rows(path)
            expected={(p['name'],b,r) for p in plan['panels'] for b in range(p['blocks']) for r in (0,1)}
            seen={(r['panel'],r['block'],r['rotation']) for r in old}
            if len(seen)!=len(old) or not seen<=expected or any(r['policy']!=spec['name'] or r['status']!='complete' for r in old):raise ValueError('Unexpected original records')
            receipts.append({'name':path.name,'bytes':path.stat().st_size,'sha256':file_hash(path),'surviving_hands':len(old),'gzip_problem':problem})
            missing=expected-seen;destination=out/path.name
            if not missing:
                if problem:raise ValueError('A complete file has a damaged stream')
                shutil.copyfile(path,destination);rows.extend(old);continue
            if problem:shutil.copyfile(path,out/(path.name+'.interrupted'))
            model_path=(inputs if spec['strategy']=='current' else averages)/spec['path']
            if model_path.stat().st_size!=spec['bytes'] or file_hash(model_path)!=spec['sha256']:raise ValueError('Recovery input bytes differ')
            began=perf_counter();source=(FrozenBlueprint(Checkpoint(spec['name'],str(model_path),spec['sha256'],HU20_UNCAPPED_FORMAT),model_path)
                    if spec['strategy']=='current' else DiagnosticAverage(model_path,spec['sha256']))
            if source.description['training_seed']!=spec['seed'] or source.description['iteration']!=spec['iteration']:raise ValueError('Recovery lineage differs')
            loaded.append({'model':spec,'description':source.description,'load_seconds':perf_counter()-began})
            with gzip.open(destination,'wt') as f:
                for row in old:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                rows.extend(old)
                for panel in plan['panels']:
                    for block in range(panel['blocks']):
                        for rotation in (0,1):
                            if (panel['name'],block,rotation) not in missing:continue
                            guard();row=play(source,spec,panel,plan['root'],block,rotation,guard)
                            rows.append(row);new_hands+=1;f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush()
                            if new_hands%16==0:print(json.dumps({'new_hands':new_hands,'seconds':perf_counter()-start}),flush=True)
            del source;gc.collect()
    except Exception as exc:failure=f'{type(exc).__name__}: {exc}'
    result={'status':'incomplete' if failure else 'complete','failure':failure,'source':original_source,
            'plan':plan,'plan_sha256':digest(plan),'hands':len(rows),'loaded':loaded,
            'recovery':{'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'original_files':receipts,'new_hands':new_hands,'retained_original_hands':len(rows)-new_hands,
                'deadline_anchor_source_commit':anchor,'original_conservative_deadline':deadline,
                'seconds':perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'original_worker_peak_rss_unavailable':True,'reason':'tool turn interruption terminated worker before summary publication'},
            'seconds':time()-anchor,'peak_rss_bytes':None}
    if not failure:result.update(summarize(rows))
    (out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'manifest.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in out.iterdir()},indent=2,sort_keys=True)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','original','inputs','averages','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--original-source',required=True);a=p.parse_args()
    result=complete(json.loads(a.plan.read_text()),a.original,a.inputs,a.averages,a.out,a.original_source)
    print(json.dumps({k:result[k] for k in ('status','failure','hands','seconds')}))
    if result['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
