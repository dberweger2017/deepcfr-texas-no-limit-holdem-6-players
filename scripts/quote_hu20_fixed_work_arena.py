"""Price fixed work using censored stage-4 requests and six-thread CPU replays.

This produces a financial forecast, never an action deadline or rental request.
Stage-4 wall times are not mislabelled as measured process CPU consumption.
"""
import argparse
import json
from math import ceil, floor
from pathlib import Path
from time import time

from src.arena.schedule import digest

THREADS = 6
WORK_PROTOCOL = 'hu20-fixed50-no-fallback-v1'
RESERVES = {'setup_build':1800,'actual_host_parity':900,'replay_verification':3600,
            'retrieval_hash_verification':900,'shutdown':600,'retention_archive':300}


def admitted_workers(quota_cpus, ram_bytes):
    # No latency target or fixed spare-CPU rule. Memory includes a family cap
    # and a pod-level reserve for the controller/SSH/build tools.
    return max(0,min(floor(quota_cpus/THREADS),floor((ram_bytes-2*1024**3)/(9*1024**3))))


def work_estimate(stage4, replays):
    if not stage4 or not replays:raise ValueError('Retained timing evidence is required')
    if any(r['threads']!=6 for r in stage4+replays):raise ValueError('Six-thread evidence required')
    if any(r['cpu_seconds']<=0 or r['cpu_seconds']>6.2*r['wall_seconds']+.1
           or r['completion']['iterations']!=50 or r['completion']['status']!='play_complete'
           for r in replays):raise ValueError('CPU replay failed fixed-work/thread checks')
    statuses={r['status'] for r in stage4}
    if statuses!={'completed','failure'}:raise ValueError('Preserve completed and censored strata')
    if any(r['status']=='failure' and r['cause']!='timeout' for r in stage4):
        raise ValueError('Non-timeout defects need owner review')
    if len({r['request_sha256'] for r in replays})!=len(replays):raise ValueError('Duplicate timing replay')
    weighted=[]
    for status in sorted(statuses):
        original=[r for r in stage4 if r['status']==status]
        sample=[r for r in replays if r['status']==status]
        wanted=sorted(original,key=lambda r:r['request_sha256'])[:12]
        if [r['request_sha256'] for r in sample]!=[r['request_sha256'] for r in wanted]:
            raise ValueError('Timing sample differs from predeclared hash ordering')
        weight=len(original)/len(stage4)/len(sample)
        weighted.extend((r['cpu_seconds'],weight) for r in sample)
    cumulative=0
    for cpu,weight in sorted(weighted):
        cumulative+=weight
        if cumulative>=.95:p95=cpu;break
    ratios=[r['solver_seconds']/(r['cpu_seconds']/THREADS) for r in replays
            if r['status']=='completed' and r['cpu_seconds']>1]
    if not ratios:raise ValueError('Matched completed requests needed for host scaling')
    return {'mean_reference_cpu_seconds':sum(cpu*w for cpu,w in weighted),
            'p95_reference_cpu_seconds':p95,'reference_cpu_method':'native child user+system seconds on M4, RUSAGE_CHILDREN delta, six-thread environment',
            'stage4_host_cost_scale':max(ratios),'scale_matched_requests':len(ratios),
            'scale_definition':'max(stage4 native elapsed / (same-request M4 six-thread CPU seconds / 6))',
            'stage4_requests':len(stage4),'censored_requests':sum(r['status']=='failure' for r in stage4),
            'replayed_requests':len(replays),'stage4_sha256':digest(stage4),'replays_sha256':digest(replays),
            'limitations':'24 requests from an early stopped panel; not all-panel timing or a completion-time bound; actual-host recomputation required'}


def quote(stage4,replays,offers,*,now=None):
    now=time() if now is None else now
    work=work_estimate(stage4,replays)
    # #148's outcome-free 17,100-call estimate is not a scientific solve cap.
    # The same count is priced with all-call p95 and 1.5 overall contingency.
    expected_calls=17100;maximum_calls=expected_calls
    workers=6;pods=2
    historical_cap_charge=.6518542516195319+.01481
    pilot_allowance=.50;storage_contingency=1.0
    prior=historical_cap_charge+pilot_allowance+storage_contingency
    forecasts=[]
    for offer in offers:
        if not 0<=now-offer['retrieved_at']<3600:raise ValueError('Fresh RunPod MCP offer required')
        if offer['cloud']!='COMMUNITY' or offer['provider']!='RunPod':raise ValueError('Community price required')
        price=offer['compute_hourly_usd'];disk=60*.10/720;scale=work['stage4_host_cost_scale']
        # These are explicit planning allowances for work outside the native
        # process, not fabricated measurements. Part A used 4,979.67 M4 seconds.
        mean_production=(expected_calls*(work['mean_reference_cpu_seconds']*scale/THREADS+1.5)+12000)/workers
        maximum_production=(maximum_calls*(work['p95_reference_cpu_seconds']*scale/THREADS+4)+36000)/workers
        expected_seconds=ceil(mean_production+sum(RESERVES.values()))
        maximum_seconds=ceil(1.5*(maximum_production+sum(RESERVES.values())))
        expected_cost=ceil((pods*expected_seconds/3600*(price+disk)+prior)*100)/100
        maximum_cost=ceil((pods*maximum_seconds/3600*(price+disk)+prior)*100)/100
        forecasts.append({**offer,'pods':pods,'workers_per_pod':3,'total_workers':workers,
            'threads_per_worker':6,'minimum_quota_cpus':18,'minimum_ram_bytes':32*10**9,
            'family_rss_limit_bytes':9*1024**3,'disk_gb_per_pod':60,
            'compute_usd_per_admitted_quota_cpu_hour':price/18,
            'expected_hours':expected_seconds/3600,'p95_headroom_hours':maximum_seconds/3600,
            'expected_cost_usd':expected_cost,'maximum_forecast_cost_usd':maximum_cost,
            'within_hard_ceiling':maximum_cost<=25,
            'stock_is_host_admission':False})
    return {'status':'quote-awaiting-owner-approval','owner_approved':False,'revision':3,
        'work_protocol':WORK_PROTOCOL,'expected_hands':82944,'reused_stage4_hands':0,
        'timing':work,'expected_native_calls':expected_calls,'maximum_native_call_allowance':maximum_calls,
        'non_native_allowances':{'expected_prepare_parse_seconds_per_call':1.5,
          'maximum_prepare_parse_seconds_per_call':4,'expected_other_worker_seconds_total':12000,
          'maximum_other_worker_seconds_total':36000,
          'basis':'explicit allowance for public preparation/parsing, model load, base play and complete four-batch LBR; 1.5x maximum contingency remains separate'},
        'reserves_seconds_per_pod':RESERVES,'headroom_multiplier':1.5,
        'historical_cap_charge_usd':historical_cap_charge,'pilot_allowance_usd':pilot_allowance,
        'storage_contingency_usd':storage_contingency,'hard_ceiling_usd':25,
        'dispatch_stop_usd':21,'closeout_reserve_usd':4,'offers':forecasts,
        'maximum_is_completion_guarantee':False,'wall_time_is_action_deadline':False,
        'admission':'actual quota/RAM/disk, green source CI, exact scientific-output parity, retention, and actual-host cost recomputation before arena',
        'no_paid_allocation_performed':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('stage4','replays','offers','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve previous quote')
    result=quote(*(json.loads(getattr(a,k).read_text()) for k in ('stage4','replays','offers')))
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps([(o['gpu'],o['expected_cost_usd'],o['maximum_forecast_cost_usd']) for o in result['offers']]))


if __name__=='__main__':main()
