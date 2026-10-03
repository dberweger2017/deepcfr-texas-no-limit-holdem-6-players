"""Memory-only native/3/2 turn admission; no main values or training."""
import argparse
import json
import math
from pathlib import Path
import sys
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.flop_check_runtime import machine_snapshot,run_tool,append
from scripts.preflight_flop_check import prepare_guarded,heartbeat


def preflight(binary,plan,inputs,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    before=machine_snapshot();atomic_json(out/'machine-before.json',before)
    budget=min(6*1024**3,math.floor(before['reclaimable_bytes']*.8/1024**3)*1024**3)
    if budget<2*1024**3:raise MemoryError('Insufficient measured turn headroom')
    baseline=before['swap_used_bytes'];records=[];admitted={}
    atomic_json(out/'admission.json',{'budget_bytes':budget,'swap_baseline_bytes':baseline,'threads':2,'nice':10})
    for cap in (None,3,2):
        label='native' if cap is None else f'cap{cap}'
        prepared=out/f'requests-{label}'
        command=[sys.executable,'-m','scripts.prepare_flop_check','--plan',str(plan),'--inputs',str(inputs),
                 '--out',str(prepared),'--memory-gib',str(budget/1024**3),'--street','turn']
        if cap is not None:command+=['--raise-cap',str(cap)]
        heartbeat(out,f'{label} turn preparation',len(records),9)
        prepare_guarded(command,out,budget,baseline)
        manifest=json.loads((prepared/'manifest.json').read_text())
        for item in manifest['records']:
            if item['kind'] in admitted:continue
            if item['status']!='prepared':records.append(item);continue
            path=prepared/item['request'];run=out/f'{label}-{item["kind"]}'
            runtime=run_tool(binary,path,run,memory_bytes=budget,threads=2,seconds=600,initial_swap=baseline)
            if runtime['status']!='completed':raise RuntimeError(runtime['failure'])
            rows=[json.loads(t) for t in (run/'response.jsonl').read_text().splitlines()]
            for row in rows:append(out/'progress.jsonl',dict(row,spot=item['kind'],stage='turn-preflight',tree=label))
            memory=next((r for r in rows if r['event']=='memory'),None)
            fits=bool(memory and memory['compressed_bytes']<=budget)
            record=dict(item,tree=label,memory=memory,completion=rows[-1],runtime=runtime,fits=fits,
                        removed_reach_audit='not needed' if cap is None else 'pending; not admitted')
            records.append(record)
            if fits:admitted[item['kind']]=record
            atomic_json(out/'partial.json',{'records':records,'admission_budget_bytes':budget})
        if len(admitted)==3:break
    result={'records':records,'admission_budget_bytes':budget,'swap_baseline_bytes':baseline,
            'all_three_memory_fit':len(admitted)==3,'three_bet_fits':'3-bet' in admitted,
            'main_run_admitted':False,'admitted_memory_only':list(admitted),
            'status':'memory preflight complete; validation/timing pending' if len(admitted)==3
                     else 'resource-blocked; stop before main run', 'after':machine_snapshot()}
    atomic_json(out/'result.json',result);heartbeat(out,result['status'],len(records),len(records))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','plan','inputs','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();preflight(args.binary,args.plan,args.inputs,args.out)
if __name__=='__main__':main()
