"""Real-export turn preflight: native payoffs, both-policy EV, and solve timing.

No target-only losses are measured here; the main protocol remains unfrozen.
"""
import argparse
import gc
import json
from pathlib import Path
from random import Random
import signal
from time import monotonic
import numpy as np
from scripts.prepare_flop_check import load_policy
from scripts.validate_flop_check import monte_carlo,native_line,response_rows
from src.diagnostics.flop_check import atomic_json,fixture_root
from src.diagnostics.flop_check_runtime import append,run_tool
from src.diagnostics.turn_check import export
from src.game.observation import HandFinished,replay
from src.game.types import Street
from src.blueprint.search import DECK


def validate(binary,plan,inputs,prepared,out,kind,*,budget=4*1024**3,baseline=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    spec=json.loads(Path(plan).read_text())['policies'][0]
    request=json.loads((Path(prepared)/(kind+'.json')).read_text())
    root=fixture_root(kind,street=Street.TURN);view=replay(root,0,())
    started=monotonic();compact=export(root,spec,inputs,out/'export')
    export_seconds=monotonic()-started;gc.collect()
    request.update(mode='solve',compress=True,memory_budget_bytes=budget,max_iterations=1,
                   progress_every=1,seconds=120,compact_kind='bp-ev',compact_path=str(compact.resolve()))
    rng=Random(202610020220+('limped','min-raised','3-bet').index(kind))
    ranges={s:tuple((tuple(h['hand']),h['weight']) for h in request['ranges'][p])
            for p,s in enumerate(request['seat_map'])}
    terminals=[n for n in request['nodes'] if n['terminal']];expected=[];queries=[];types={}
    for _ in range(300):
        while True:
            hands={s:rng.choice(ranges[s])[0] for s in (0,1)}
            if not set(hands[0])&set(hands[1]):break
        available=[c for c in DECK
                   if c not in (*view.board,*hands[0],*hands[1])]
        board=view.board+(rng.choice(available),);node=rng.choice(terminals)
        hand=native_line(root,board,hands,node['line'])
        finish=next(e for e in hand.events if isinstance(e,HandFinished))
        expected.append([finish.stacks[s]-view.players[s].stack-view.pot/2 for s in request['seat_map']])
        queries.append({'line':node['line'],'board':list(board),'hands':[hands[s] for s in request['seat_map']]})
        name='showdown' if finish.showdown else 'fold';types[name]=types.get(name,0)+1
    request['terminal_queries']=queries;path=out/'both-locked.json';atomic_json(path,request)
    runtime=run_tool(binary,path,out/'both-locked',memory_bytes=budget,threads=2,seconds=600,initial_swap=baseline)
    if runtime['status']!='completed':raise RuntimeError(runtime['failure'])
    rows=response_rows(out/'both-locked/response.jsonl')
    v1=next(r for r in rows if r.get('gate')=='V1')
    actual=[r['payoff_chips'] for r in rows if r['event']=='payoff_query']
    if len(actual)!=len(expected):raise ValueError('Missing native payoff queries')
    error=np.abs(np.asarray(actual)-expected)
    v2={'gate':'V2','passed':bool(np.all(np.rint(actual)==expected) and error.max()<.01),
        'samples':len(actual),'terminal_types':types,'integer_chip_mismatches':int(np.count_nonzero(np.rint(actual)!=expected)),
        'maximum_float_chip_error':float(error.max())}
    gates=[v1,v2];atomic_json(out/'gates.json',{'gates':gates})
    if not all(g['passed'] for g in gates):raise ValueError('Turn tree/payoff gate failed')
    source=load_policy(spec,inputs);started=monotonic()
    mc=monte_carlo(root,ranges,source,seed=202610020230+('limped','min-raised','3-bet').index(kind))
    mc_seconds=monotonic()-started;del source;gc.collect()
    value=next(r for r in rows if r['event']=='both_blueprint_ev')['current_ev_chips'][request['seat_map'].index(0)]/100
    v4={'gate':'V4','passed':mc['ci95'][0]<=value<=mc['ci95'][1],
        'solver_ev_bb':value,'native_mc':mc,'real_export':True,'positive_holdings':[len(r) for r in request['ranges']]}
    gates.append(v4);atomic_json(out/'gates.json',{'gates':gates})
    if not v4['passed']:
        atomic_json(out/'failure.json',{'gates':gates});raise ValueError('Real-export V4 failed')
    request.pop('compact_path');request.pop('compact_kind');request.pop('terminal_queries')
    request.update(max_iterations=10000,progress_every=25,target_pct_pot=.2,seconds=900)
    path=out/'equilibrium.json';atomic_json(path,request)
    eq_runtime=run_tool(binary,path,out/'equilibrium',memory_bytes=budget,threads=2,seconds=1050,initial_swap=baseline)
    if eq_runtime['status']!='completed':raise RuntimeError(eq_runtime['failure'])
    eq_rows=response_rows(out/'equilibrium/response.jsonl');final=eq_rows[-1]
    gates.append({'gate':'V5','passed':final['exploitability_pct_pot']<=.2,
                  'exploitability_pct_pot':final['exploitability_pct_pot']})
    result={'kind':kind,'gates':gates,'source':spec,'fallback':'native; 16-bit compressed',
            'export_seconds':export_seconds,'mc_seconds':mc_seconds,'bp_runtime':runtime,
            'equilibrium_runtime':eq_runtime,'equilibrium':final,'main_values_measured':False}
    atomic_json(out/('result.json' if all(g['passed'] for g in gates) else 'failure.json'),result)
    return result


def main():
    signal.signal(signal.SIGTERM,lambda signum,frame:(_ for _ in ()).throw(SystemExit('Resource guard stopped owned work')))
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','plan','inputs','prepared','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--kind',choices=('limped','min-raised','3-bet'),required=True)
    p.add_argument('--budget-gib',type=float,default=4);p.add_argument('--swap-baseline',type=int)
    a=p.parse_args()
    result=validate(a.binary,a.plan,a.inputs,a.prepared,a.out,a.kind,budget=int(a.budget_gib*1024**3),baseline=a.swap_baseline)
    if not all(g['passed'] for g in result['gates']):raise SystemExit('Turn validation failed')
if __name__=='__main__':main()
