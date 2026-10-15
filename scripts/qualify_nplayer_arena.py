"""Fixed cost/correctness qualification with PR291's smoke policy; no strength gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.analyze_nplayer_arena import sha
from scripts.check_nplayer_policy_replay import check

SMOKE = Path('/home/dberweger/nplayer-size-smoke-20261015/smoke/scoped-average.gz')
SMOKE_SHA = 'fcbe8bd5733572fe9f9046840d6e277a0b11ce0ac96936546df0815be218485e'


def run(binary, tables, out):
    stop = os.environ.get('NPLAYER_STOP_FILE')
    if not stop:
        raise RuntimeError('Run inside the on-host VRM supervisor')
    if sha(SMOKE) != SMOKE_SHA:
        raise ValueError('smoke input SHA mismatch')
    out.mkdir(parents=True, exist_ok=False)
    model = f'policy={SMOKE}@{SMOKE_SHA}'
    cases = [('self', model, model)]
    for bot in ('random', 'check-call', 'tight-aggressive', 'pot-pressure'):
        cases.extend([(bot+'-singleton', model, 'bot='+bot), (bot+'-pair', 'bot='+bot, model)])
    started = time.monotonic()
    records = []
    for index, (name, a, b) in enumerate(cases):
        root = 2026101551 + index
        traces = []
        runs = []
        for repeat in range(2):
            if Path(stop).exists(): raise RuntimeError('supervisor requested stop')
            trace = out/f'{name}-{repeat}.jsonl'
            command = [str(binary), str(tables), '64', str(root), str(trace), a, b]
            tick = time.monotonic()
            result = subprocess.run(command, capture_output=True, text=True)
            record = {'command': command, 'seconds': time.monotonic()-tick,
                      'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
            (out/f'{name}-{repeat}.command.json').write_text(json.dumps(record, indent=2)+'\n')
            if result.returncode: raise RuntimeError(f'play failed: {name}-{repeat}')
            traces.append(trace)
            runs.append(json.loads(result.stdout))
        if sha(traces[0]) != sha(traces[1]): raise ValueError('fixed-seed reproduction failed')
        tick = time.monotonic()
        replay = check(traces[0])
        replay_seconds = time.monotonic()-tick
        (out/f'{name}.replay.json').write_text(json.dumps(replay, indent=2)+'\n')
        records.append({'name': name, 'root': root, 'blocks':64, 'runs':runs,
                        'replay_seconds':replay_seconds, 'trace_bytes':traces[0].stat().st_size})
        # Keep strength readout separate: these observations estimate cost only.
        (out/'costs.json').write_text(json.dumps({'status':'partial','cases':records},indent=2)+'\n')
    result = {'status':'passed','seconds':time.monotonic()-started,'cases':records,
              'source': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'binary_sha256':sha(binary),'smoke_sha256':SMOKE_SHA}
    (out/'costs.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',required=True,type=Path)
    p.add_argument('--tables',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();run(a.binary,a.tables,a.out)
