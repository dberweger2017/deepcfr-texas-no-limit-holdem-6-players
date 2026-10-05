"""Time the complete compact pipeline on declared, excluded preflight roots."""
import argparse
import json
from pathlib import Path
import signal
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.flop_check_runtime import run_tool
from scripts.validate_flop_check import response_rows


def main():
    def stop(signum,frame):raise SystemExit('Cost pilot stopped by resource guard')
    signal.signal(signal.SIGTERM,stop)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--budget-gib',type=float,default=4)
    p.add_argument('--swap-baseline',type=int,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((a.preflight/'result.json').read_text())
    if not all(g['passed'] for g in prior['gates']):raise ValueError('Prior real-export gates did not pass')
    request=json.loads((a.preflight/'equilibrium.json').read_text())
    request.update(memory_budget_bytes=int(a.budget_gib*1024**3),compact_path=str((a.preflight/'export/compact.json').resolve()),
                   compact_kind='all',secondary_ranges=request['ranges'])
    path=a.out/'request.json';atomic_json(path,request)
    runtime=run_tool(a.binary,path,a.out/'solver',memory_bytes=request['memory_budget_bytes'],threads=2,seconds=1200,
                     initial_swap=a.swap_baseline)
    if runtime['status']!='completed':raise RuntimeError(runtime['failure'])
    rows=response_rows(a.out/'solver/response.jsonl');final=rows[-1]
    gates=[r for r in rows if r.get('gate')=='V1']
    exact=next(r['current_ev_chips'] for r in rows if r['event']=='both_blueprint_ev')
    original=json.loads((a.preflight/'both-locked.json').read_text())
    value=exact[original['seat_map'].index(0)]/100;mc=next(g['native_mc'] for g in prior['gates'] if g['gate']=='V4')
    gates.append({'gate':'V4-current-tool','passed':mc['ci95'][0]<=value<=mc['ci95'][1],
                  'solver_ev_bb':value,'native_mc':mc,'source':'identical frozen source/root/ranges; retained independent 20,000-deal native sample'})
    primary={(r['metric'],r['target_solver_seat']):r['gain_bb'] for r in rows if r['event']=='compact_metric'}
    errors=[abs(r['gain_bb']-primary[r['metric'],r['target_solver_seat']]) for r in rows if r['event']=='compact_secondary_metric']
    gates.append({'gate':'secondary-identity','passed':len(errors)==10 and max(errors)<3e-5,
                  'samples':len(errors),'maximum_error_bb':max(errors) if errors else None})
    gates.append({'gate':'V5','passed':final['exploitability_pct_pot']<=.2,'exploitability_pct_pot':final['exploitability_pct_pot']})
    progress=[r for r in rows if r['event']=='progress']
    result={'purpose':'complete cost preflight; root excluded from main corpus','kind':prior['kind'],'gates':gates,
            'runtime':runtime,'equilibrium':final,'equilibrium_seconds':progress[-1]['elapsed_seconds'],
            'postsolve_seconds':final['elapsed_seconds']-progress[-1]['elapsed_seconds'],
            'main_values_measured':False,'metrics_recorded_in_retained_raw_response':True}
    atomic_json(a.out/('result.json' if all(g['passed'] for g in gates) else 'failure.json'),result)
    if not all(g['passed'] for g in gates):raise ValueError('Complete cost preflight gate failed')
if __name__=='__main__':main()
