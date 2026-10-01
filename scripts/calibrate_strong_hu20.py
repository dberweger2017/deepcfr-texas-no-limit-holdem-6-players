"""Run the declared heuristic controls on M1; preserve every native replay."""
import argparse
import gzip
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from src.arena.schedule import digest
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.strong_evaluation import calibration_gates, match, summarize_matches


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=('pilot','development','confirmation'),required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    c=config['calibration'];root=202610040100 if a.stage=='pilot' else c[a.stage+'_root'];blocks=2 if a.stage=='pilot' else c[a.stage+'_blocks']
    start=perf_counter();rows=[];failure=None
    try:
        with gzip.open(a.out/'hands.jsonl.gz','wt') as f:
            for mode in c['modes']:
                for control in c['controls']:
                    for block in range(blocks):
                        for rotation in (0,1):
                            row=match(config,mode,root,block,rotation,control=control)
                            rows.append(row);f.write(json.dumps(row,sort_keys=True)+'\n');f.flush()
                    print(json.dumps({'mode':mode,'control':control,'hands':len(rows),'seconds':perf_counter()-start}),flush=True)
    except Exception as exc:failure=f'{type(exc).__name__}: {exc}'
    result={'stage':a.stage,'root':root,'blocks_per_control':blocks,'hands':len(rows),'failure':failure,
        'status':'incomplete' if failure else 'complete','config_sha256':digest(config),'config':config,
        'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'seconds':perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'panels':summarize_matches(rows) if not failure else []}
    if a.stage=='confirmation' and not failure:result['gates']=calibration_gates(result['panels'])
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (a.out/'manifest.json').write_text(json.dumps({f.name:{'bytes':f.stat().st_size,'sha256':file_hash(f)} for f in a.out.iterdir()},indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('stage','status','hands','seconds','peak_rss_bytes','failure')}))
    if failure:raise SystemExit(1)


if __name__=='__main__':main()
