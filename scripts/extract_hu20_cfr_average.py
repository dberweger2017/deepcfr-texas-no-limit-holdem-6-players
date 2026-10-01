"""Extract/audit all three frozen B500M retained accumulators on M1."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from src.diagnostics.cfr_average import extract, audit
from src.diagnostics.saved_hu20 import file_hash


def run(plan,inputs,out):
    out.mkdir(parents=True,exist_ok=False);start=perf_counter();records=[];failure=None
    try:
        for spec in plan['models']:
            checkpoint=inputs/spec['checkpoint_path'];current=inputs/spec['path']
            if checkpoint.stat().st_size!=spec['checkpoint_bytes'] or current.stat().st_size!=spec['bytes']:
                raise ValueError('Source length differs before extraction')
            begun=perf_counter();path=out/(spec['name']+'.average.jsonl.gz')
            exported=extract(checkpoint,spec,path)
            checked=audit(checkpoint,current,path,spec,exported['sha256'])
            row={'model':spec,'export':exported,'audit':checked,'seconds':perf_counter()-begun,'path':path.name}
            records.append(row);print(json.dumps({'seed':spec['seed'],'audit':checked,'seconds':row['seconds']}),flush=True)
    except Exception as exc:failure=f'{type(exc).__name__}: {exc}'
    result={'status':'incomplete' if failure else 'complete','failure':failure,'records':records,'plan':plan,
        'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'seconds':perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'manifest.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in out.iterdir()},indent=2,sort_keys=True)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=run(json.loads(a.plan.read_text()),a.inputs,a.out)
    if result['status']!='complete':raise SystemExit(result['failure'])


if __name__=='__main__':main()
