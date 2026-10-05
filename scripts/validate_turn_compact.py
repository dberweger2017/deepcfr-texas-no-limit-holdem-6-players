"""Cross-check external compact turn losses against independent Python locks.

Preparation, projection, and comparison use separate workers so the deliberately
verbose fixture profile is released before any locked solver is launched.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import numpy as np
from src.diagnostics.flop_check import atomic_json,compile_tree,fixture_root,line_key
from src.diagnostics.flop_check_analysis import blueprint_locks,projection
from src.diagnostics.flop_check_runtime import run_tool
from src.diagnostics.turn_check import export
from src.game.types import Street
from scripts.validate_flop_check import small_ranges,response_rows

METRICS=('e_bp','e_v1proj','e_v1proj_line','e_eq50','e_eq200')


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False)
    root=fixture_root(a.kind,street=Street.TURN)
    spec=json.loads(a.plan.read_text())['policies'][0]
    compact=export(root,spec,a.inputs,a.out/'export')
    request,_=compile_tree(root);ranges=small_ranges(request['board'])
    request.update(mode='solve',memory_budget_bytes=1024**3,
        ranges=[[{'hand':list(h),'weight':w} for h,w in ranges[s]] for s in request['seat_map']],
        compact_path=str(compact.resolve()),compact_kind='all',dump_path=str((a.out/'profile.jsonl').resolve()),
        max_iterations=2000,progress_every=25,target_pct_pot=.01,seconds=180)
    path=a.out/'request.json';atomic_json(path,request)
    result=run_tool(a.binary,path,a.out/'compact',memory_bytes=1024**3,threads=2,seconds=600,job_memory_bytes=4*1024**3)
    if result['status']!='completed':raise RuntimeError(result['failure'])


def project(a):
    request=json.loads((a.out/'request.json').read_text())
    data=json.loads((a.out/'export/compact.json').read_text())
    records=response_rows(a.out/'profile.jsonl')
    templates={line_key(n['line']):n['template'] for n in request['nodes'] if not n['terminal']}
    tables={'tables':data['tables'],'node_tables':data['node_tables']}
    hands={tuple(sorted(h)):i for i,h in enumerate(data['holdings'])};boards={tuple(b):i for i,b in enumerate(data['boards'])}
    for name in ('compact_path','compact_kind','dump_path'):request.pop(name)
    request.update(max_iterations=1,seconds=120,memory_budget_bytes=2*1024**3)
    for metric in METRICS:
        if metric=='e_bp':locks=blueprint_locks(records,request,tables)
        elif metric.startswith('e_eq'):
            k=int(metric[4:])
            bucket=lambda row,h:data['labels'][str(k)][boards[tuple(row['board'])]][hands[tuple(sorted(h))]]
            locks=projection(records,templates,bucket=bucket)
        else:locks=projection(records,templates,per_line=metric.endswith('_line'))
        for target in (0,1):
            atomic_json(a.out/f'{metric}-{target}.json',dict(request,locks=[r for r in locks if r['player']==target]))
        del locks
    atomic_json(a.out/'projection.json',{'aliased_public_nodes':len(data['node_tables'])-len(data['tables'])})


def compare(a):
    rows=response_rows(a.out/'compact/response.jsonl');reference=rows[-1]
    values={(r['metric'],r['target_solver_seat']):r['gain_bb'] for r in rows if r['event']=='compact_metric'}
    checks=[]
    for metric in METRICS:
        for target in (0,1):
            path=a.out/f'{metric}-{target}.json'
            runtime=run_tool(a.binary,path,a.out/f'{metric}-{target}',memory_bytes=2*1024**3,
                             threads=2,seconds=180,job_memory_bytes=4*1024**3)
            if runtime['status']!='completed':raise RuntimeError(runtime['failure'])
            final=response_rows(a.out/f'{metric}-{target}/response.jsonl')[-1]
            gain=(final['mes_ev_chips'][1-target]-reference['current_ev_chips'][1-target])/100
            error=abs(values[metric,target]-gain)
            checks.append({'metric':metric,'target':target,'compact_gain_bb':values[metric,target],
                           'python_gain_bb':gain,'error_bb':error,'passed':error<3e-5})
    summary={'gate':'compact-locks','passed':all(c['passed'] for c in checks),'checks':checks,
             'equilibrium_residual_pct_pot':reference['exploitability_pct_pot'],
             'both_blueprint_ev':next(r for r in rows if r['event']=='both_blueprint_ev'),
             'fixture':a.kind+' turn; four artificial holdings per seat; not main evidence',
             **json.loads((a.out/'projection.json').read_text())}
    atomic_json(a.out/('result.json' if summary['passed'] else 'failure.json'),summary)
    if not summary['passed']:raise ValueError('Compact lock comparison failed')


def main():
    child=None
    def stop(signum,frame):
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=5)
        # A worker interrupted in run_tool closes its owned solver group.
        raise SystemExit('Turn validation stopped by resource guard')
    signal.signal(signal.SIGTERM,stop)
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','plan','inputs','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--kind',choices=('tiny-spr','limped'),default='tiny-spr')
    p.add_argument('--phase',choices=('prepare','project','compare'),help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.phase:return {'prepare':prepare,'project':project,'compare':compare}[a.phase](a)
    for phase in ('prepare','project','compare'):
        child=subprocess.Popen([sys.executable,'-m','scripts.validate_turn_compact',*sys.argv[1:],'--phase',phase])
        try:
            if child.wait()!=0:raise RuntimeError(f'Compact fixture {phase} failed')
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=5)
        child=None
if __name__=='__main__':main()
