"""River-native check of responder-only empirical-range EV reweighting."""
import argparse
import json
import signal
from pathlib import Path
from scripts.prepare_flop_check import load_policy
from scripts.validate_flop_check import small_ranges,locks_for_river,response_rows
from src.blueprint.hu20_river import HU20RiverGame
from src.blueprint.river_cfr import profile_quality
from src.diagnostics.flop_check import atomic_json,compile_tree,fixture_root
from src.diagnostics.flop_check_runtime import run_tool
from src.game.types import Street


def validate(binary,plan,inputs,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    source=load_policy(json.loads(Path(plan).read_text())['policies'][0],inputs);checks=[]
    for kind in ('limped','3-bet'):
        root=fixture_root(kind,street=Street.RIVER);request,histories=compile_tree(root)
        ranges=small_ranges(request['board']);locks,profile=locks_for_river(request,histories,ranges,source)
        changed={s:tuple((h,(4-i)/10) for i,(h,w) in enumerate(ranges[s])) for s in (0,1)}
        qualities={p:profile_quality(HU20RiverGame(root,ranges|{request['seat_map'][p]:changed[request['seat_map'][p]]},raise_cap=None),profile) for p in (0,1)}
        request.update(mode='solve',memory_budget_bytes=512*1024**2,
            ranges=[[{'hand':list(h),'weight':w} for h,w in ranges[s]] for s in request['seat_map']],
            secondary_ranges=[[{'hand':list(h),'weight':w} for h,w in changed[s]] for s in request['seat_map']],
            max_iterations=1,progress_every=1,seconds=120)
        for target in (None,0,1):
            path=out/f'{kind}-{target}.json';atomic_json(path,dict(request,locks=locks if target is None else [r for r in locks if r['player']==target]))
            runtime=run_tool(binary,path,out/f'{kind}-{target}',memory_bytes=512*1024**2,threads=2,seconds=180,job_memory_bytes=4*1024**3)
            if runtime['status']!='completed':raise RuntimeError(runtime['failure'])
            rows=response_rows(out/f'{kind}-{target}/response.jsonl')
            for p in ((0,1) if target is None else (1-target,)):
                quality=qualities[p];seat=request['seat_map'][p]
                event='secondary_reference' if target is None else 'secondary_mes'
                value=next(r['value'] for r in rows if r['event']==event and r['responder']==p)
                expected=quality['profile_values_bb'][seat]+(0 if target is None else quality['best_response_gains_bb'][seat])
                actual=value['ev_chips']/100;error=abs(expected-actual)
                checks.append({'kind':kind,'target':target,'responder':p,'event':event,'expected_bb':expected,'actual_bb':actual,
                               'error_bb':error,'retained_fraction':value['retained_fraction'],
                               'passed':error<3e-5 and abs(value['retained_fraction']-1)<1e-8})
                if not checks[-1]['passed']:
                    atomic_json(out/'failure.json',{'gate':'secondary-native','checks':checks});raise ValueError('Reweighting validation failed')
    result={'gate':'secondary-native','passed':True,'checks':checks,'source':source.description,
            'fixture':'river; four artificial holdings per seat; bias only responder range, preserve target range'}
    atomic_json(out/'result.json',result);return result


def main():
    def stop(signum,frame):raise SystemExit("Secondary fixture stopped by resource guard")
    signal.signal(signal.SIGTERM,stop)
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','plan','inputs','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();validate(a.binary,a.plan,a.inputs,a.out)
if __name__=='__main__':main()
