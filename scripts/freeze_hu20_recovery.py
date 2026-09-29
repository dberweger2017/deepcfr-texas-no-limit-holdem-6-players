"""Freeze M4-only phase costs from timing, before confirmation outcomes."""

import argparse
import json
from math import ceil
from pathlib import Path
from time import time

from scripts.train_hu20 import write_json,system


def freeze(path):
    plan=json.loads(path.read_text());root=Path(plan['root'])
    if plan['status']!='staging' or (root/'campaign.json').exists():raise ValueError('Recovery already frozen or launched')
    pref=json.loads((root/'preflight/result.json').read_text())
    if pref['status']!='complete':raise ValueError('Required deployed preflight incomplete')
    original=Path(plan['original_root']);old=json.loads((original/'preflight-m4/result.json').read_text())
    memory=json.loads((original/'memory-3M/result.json').read_text())
    remaining=sum(plan['training_total_nodes']-s['completed_nodes'] for s in plan['parents'].values())
    # The one-million-node prior timing and completed 20M prefix are more representative than four iterations.
    node_rate=max(old['complete_outer_seconds']/old['completed_additional_nodes'],
                  max(sum(json.loads(line)['complete_outer_seconds'] for line in
                      (root/'preflight'/f'production-{seed}'/'iterations.jsonl').read_text().splitlines())/
                      next(x['production_result']['additional_nodes'] for x in pref['lineages'] if x['seed']==seed)
                      for seed in plan['training_seeds']))
    lbr=max(x['seconds']/x['hands'] for x in pref['timing_panels'] if x['rule']=='lbr')
    lbr=max(lbr,next(x['seconds']/x['hands'] for x in old['timing_panels'] if x['rule']=='lbr'))
    cp=memory['checkpoint_seconds'];export=memory['export_seconds'];load=memory['verified_reload_seconds']
    training=remaining*node_rate*1.2+ceil(remaining/10000000)*cp+9*(cp+export+5)
    primary=6*plan['lbr_blocks']*2*lbr*1.25+6*plan['cheap_blocks']*2*.003+12*load+600
    diagnostics=(plan['expected_confirmation_hands']-6*(plan['lbr_blocks']+plan['cheap_blocks'])*2)*.003+30*load+900
    estimate=training+primary+diagnostics+plan['report_reserve_seconds']+plan['audit_reserve_seconds']
    plan['resource_forecast']={'remaining_nodes':remaining,'seconds_per_node':node_rate,
       'lbr_seconds_per_hand':lbr,'training_seconds':training,'primary_seconds':primary,
       'diagnostic_seconds':diagnostics,'total_remaining_seconds':estimate,
       'available_seconds':plan['deadline']-time(),'all_phases_conservatively_fit':time()+estimate<plan['deadline'],
       'qualification':'Forecast only; unchanged work/counts, sequential M4; diagnostic admission is resource-only and fixed before outcomes'}
    plan['status']='frozen';plan['frozen_at']=time();plan['swap_baselines']={'m4':system(['sysctl','vm.swapusage'])}
    write_json(path,plan);write_json(root/'frozen-plan.json',plan);write_json(root/'resource-forecast.json',plan['resource_forecast'])
    print(json.dumps(plan['resource_forecast']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args();freeze(a.plan)
