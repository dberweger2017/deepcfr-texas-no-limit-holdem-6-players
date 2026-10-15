"""Fixed native key-growth study; each subprocess has the qualified on-host guard."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.startswith('{')]


def main(phase, out):
    out=out.resolve();out.mkdir(exist_ok=False,parents=True)
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(), 'commit the measured source'
    binary=Path('native/hu20-trainer/target/release/examples/nplayer_size').resolve()
    legacy=binary.with_name('hu_size_reference')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    admission={'source':source,'native_tree':subprocess.check_output(['git','rev-parse','HEAD:native'],text=True).strip(),'phase':phase,'binary_sha256':sha(binary),'legacy_binary_sha256':sha(legacy),'seed':2026101520 if phase=='pilots' else 2026101521}
    if phase=='main':
        pilots=json.loads((out.parent/'pilots/result.json').read_text())
        assert pilots['status']=='passed' and max(pilots['quotes_seconds'].values())<600, pilots
        admission['pilot_admission']=pilots
    (out/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    deadline=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=3600 if phase=='main' else 300)
    results=[]
    configs=[(3,20,3),(3,100,3)] if phase=='pilots' else [(2,20,3),(2,100,3),(3,20,3),(3,100,3),(3,20,0),(3,100,0)]
    target=100_000 if phase=='pilots' else 20_000_000
    cap=12_000_000
    def run(name, command, seconds):
        env={**os.environ,'RUST_TEST_THREADS':'1','RAYON_NUM_THREADS':'4','CARGO_BUILD_JOBS':'4','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MKL_CBWR':'AVX2,STRICT'}
        wrapper=[sys.executable,'scripts/machine_qualification/watchdog.py','--output',str(out/name),'--seconds',str(seconds),'--deadline-utc',deadline.isoformat(),'--max-rss-gib','16','--min-disk-gib','32','--','env',*[f'{k}={env[k]}' for k in ('RUST_TEST_THREADS','RAYON_NUM_THREADS','CARGO_BUILD_JOBS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','MKL_CBWR')],*map(str,command)]
        code=subprocess.call(wrapper,env=env)
        rows=records(out/name/'output.log')
        result={'name':name,'guard':json.loads((out/name/'result.json').read_text()),'records':rows}
        results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        capacity_terminal = phase=='main' and code==3 and result['guard']['stop_reason'] is None and rows and rows[-1].get('stop_reason')=='capacity'
        if code and not capacity_terminal: raise SystemExit(f'{name} stopped with {code}; retained, no automatic extension')
        assert rows[-1]['stop_reason'] in ('nodes','capacity') if phase=='main' else rows[-1]['stop_reason']=='nodes',rows[-1]
        return result
    for n,bb,scope in configs:
        name=f'n{n}-{bb}bb-scope{scope}'
        run(name,[binary,n,bb,scope,admission['seed'],target,cap,110 if phase=='pilots' else 550,Path('../tables').resolve()],120 if phase=='pilots' else 600)
    if phase=='pilots':
        quotes={r['name']:2*r['records'][-1]['training_seconds']/r['records'][-1]['nodes']*20_000_000+max(30,r['guard']['elapsed_seconds']-r['records'][-1]['training_seconds']) for r in results}
        summary={'status':'passed' if max(quotes.values())<600 else 'not-admitted','quotes_seconds':quotes,'results':len(results)}
    else:
        for bb in (20,100):
            run(f'production-hu{bb}',[legacy,bb,admission['seed'],target,cap,500,Path('../tables').resolve(),out/f'production-hu{bb}.gz'],600)
        comparisons=[]
        for bb in (20,100):
            full=next(r for r in results if r['name']==f'n3-{bb}bb-scope3')['records'][-1]
            pre=next(r for r in results if r['name']==f'n3-{bb}bb-scope0')['records'][-1]
            matched=full['nodes']==pre['nodes']
            if matched: assert full['fingerprints']==pre['fingerprints'],(full,pre)
            comparisons.append({'stack_bb':bb,'matched_endpoint':matched,'fingerprints_equal':full['fingerprints']==pre['fingerprints'] if matched else None})
        summary={'status':'passed','results':len(results),'capacity_terminals':[r['name'] for r in results if r['records'][-1]['stop_reason']=='capacity'],'scope_comparisons':comparisons}
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    if summary['status']!='passed':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['pilots','main'],required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.phase,a.out)
