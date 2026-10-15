"""Execute one frozen contrast, replay all hands and reproduce before any readout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.analyze_nplayer_arena import sha
from scripts.check_nplayer_policy_replay import check
from scripts.nplayer_phase2_matrix import matrix


def run(manifest_path, case_name, binaries, tables, training, out):
    manifest=json.loads(manifest_path.read_text())
    if manifest['format']!='nplayer-phase2-evaluation-v1' or manifest['cases']!=matrix():
        raise ValueError('manifest does not match the predeclared matrix')
    stop=os.environ.get('NPLAYER_STOP_FILE')
    if not stop or Path(stop).exists(): raise RuntimeError('live VRM supervisor required')
    case=next(c for c in manifest['cases'] if c['name']==case_name)
    binary=binaries/'nplayer_arena'
    if sha(binary)!=manifest['arena_binary_sha256']: raise ValueError('harness binary pin mismatch')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('source checkout must be clean')
    out.mkdir(parents=True,exist_ok=False)
    (out/'admission.json').write_text(json.dumps({'manifest_sha256':sha(manifest_path),'case':case,
        'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},indent=2)+'\n')
    def spec(identity):
        if identity.startswith('bot='): return identity
        model=manifest['models'][identity]
        if model['strategy']!='average' or model['last_average_street']!=3:
            raise ValueError('final play requires full-street averages')
        return f'policy={training/model["member"]}@{model["sha256"]}'
    a,b=spec(case['singleton']),spec(case['pair'])
    traces=[];operations=[]
    for repeat in range(2):
        if Path(stop).exists(): raise RuntimeError('supervisor stop; no continuation')
        trace=out/f'play-{repeat}.jsonl'
        command=[str(binary),str(tables),str(case['blocks']),str(case['root']),str(trace),a,b]
        tick=time.monotonic()
        with (out/f'play-{repeat}.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        operation={'command':command,'seconds':time.monotonic()-tick,'returncode':result.returncode}
        operations.append(operation)
        (out/'operations.json').write_text(json.dumps(operations,indent=2)+'\n')
        if result.returncode or Path(stop).exists(): raise RuntimeError('play stopped/failed; retain partial')
        operation['report']=json.loads((out/f'play-{repeat}.log').read_text())
        traces.append(trace)
    if sha(traces[0])!=sha(traces[1]): raise ValueError('complete fixed-seed reproduction failed')
    if Path(stop).exists(): raise RuntimeError('supervisor stop before replay')
    tick=time.monotonic();replay=check(traces[0]);replay['elapsed_seconds']=time.monotonic()-tick
    if replay['hands']!=case['blocks']*6: raise ValueError('replay count mismatch')
    (out/'replay.json').write_text(json.dumps(replay,indent=2)+'\n')
    result={'status':'passed','case':case,'trace_sha256':replay['trace_sha256'],
        'hands':replay['hands'],'operations':operations,'replay_seconds':replay['elapsed_seconds'],
        'readout':'withheld until every final contrast passes'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--case',required=True)
    p.add_argument('--binaries',required=True,type=Path)
    p.add_argument('--tables',required=True,type=Path)
    p.add_argument('--training',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();run(a.manifest,a.case,a.binaries.resolve(),a.tables.resolve(),a.training.resolve(),a.out.resolve())
