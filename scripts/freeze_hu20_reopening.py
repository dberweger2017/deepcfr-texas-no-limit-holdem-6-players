"""Choose one common work/count from outcome-free costs, before main returns."""

import argparse
import json
from pathlib import Path
from time import time

from scipy.stats import t

from scripts.train_hu20 import write_json
from src.blueprint.windowed import _hash


def choose(summary,remaining):
    rows=summary['runs']
    if len(rows)!=6 or any(r['status']!='complete' for r in rows):raise ValueError('Six feasible resource runs required')
    pernode=max(r['complete_outer_cost']['sum_seconds']/r['completed_nodes'] for r in rows)
    keyrate=max(r['entries']/r['completed_nodes'] for r in rows)
    bytes_per_key=max(r['peak_rss_bytes']/r['entries'] for r in rows)
    # Uniform and short-trained tails are both retained; use the slower mean.
    lbr_hand=max(x['seconds']/x['hands'] for r in rows for x in r['timing_panels'] if x['rule']==['lbr'])
    proposals=[]
    for nodes in (20000000,10000000,5000000):
        for count in (2048,1024,512):
            entries=nodes*keyrate*1.25
            rss=entries*bytes_per_key+128*1024**2
            training=pernode*nodes*6*2  # Explicit conservative growth allowance.
            lbr=lbr_hand*count*2*6*1.25
            # Export/density, cheap suite and independent audit/report allowances.
            reserve=max(7200,lbr+1800+1000+900+300)
            seconds=training+reserve
            feasible=seconds<remaining and entries<4000000 and rss<10.5*1024**3
            row={'nodes':nodes,'lbr_blocks':count,'training_seconds':training,
                 'evaluation_report_reserve_seconds':reserve,'forecast_seconds':seconds,
                 'forecast_entries_per_arm':entries,'forecast_peak_rss_bytes':rss,'feasible':feasible}
            proposals.append(row)
            if feasible:return {'choice':row,'proposals':proposals,'remaining_seconds':remaining,
                'slowest_outer_seconds_per_node':pernode,'slowest_lbr_seconds_per_hand':lbr_hand,
                'growth_allowance':2,'lbr_allowance':1.25,'entry_allowance':1.25,
                'scope':'Resource forecast, not a guarantee of future tails or strength'}
    return {'choice':None,'proposals':proposals,'remaining_seconds':remaining}


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();summary=json.loads(a.summary.read_text())
    decision=choose(summary,summary['campaign']['deadline']-time())
    plan=json.loads(a.plan.read_text());choice=decision['choice']
    if choice is None:raise RuntimeError('No fixed-budget combination fits the remaining original deadline')
    plan.update(training_nodes=choice['nodes'],lbr_blocks=choice['lbr_blocks'],
        evaluation_report_reserve_seconds=int(choice['evaluation_report_reserve_seconds']+1),
        preflight_plan_digest=summary['campaign']['plan_digest'],resource_summary_sha256=_hash(a.summary),
        resource_decision=decision)
    # Prior exploratory checkpoint contrast is a variance scale, not A/B outcomes.
    # Its 512-block 95% half-width implies ~39 BB/100 at 97.5% here at n=512.
    old_sd=(34.954446817027794-(-33.03387390036112))/2/t.ppf(.975,511)*512**.5
    n=choice['lbr_blocks'];plan['anticipated_lbr_precision']={
        'source':'#114 paired 20M-minus-10M exploratory variance; not an A/B power guarantee',
        'block_sd_bb100':old_sd,'half_width_97_5_at_chosen_count':float(t.ppf(.9875,n-1)*old_sd/n**.5),
        'noninferiority_margin_bb100':10,'qualification':'May be too imprecise to establish the margin; do not widen it'}
    write_json(a.out,plan);print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
