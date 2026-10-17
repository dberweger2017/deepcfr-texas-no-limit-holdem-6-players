"""Variance-only sizing from a globally verified, disjoint calibration."""

from collections import Counter
from math import ceil, log2, sqrt
from statistics import mean, stdev
import json

import numpy as np
from scipy.stats import chi2, nct, norm, t

HISTORICAL_SD = 879.3558501872744
# PR283 measured trainer throughput, not search. Never label this a search benchmark.
CYGNUS_TRAINER_PROXY = .41144684401040693


def variance_summary(values):
    if len(values) < 2 or not all(np.isfinite(values)):
        raise ValueError('Insufficient/nonfinite calibration blocks')
    n = len(values)
    sd = stdev(values)
    return dict(blocks=n,sd_bb100=sd,nonzero_blocks=sum(v != 0 for v in values),
        sd_normal_model_ci95=[sqrt((n-1)*sd**2/chi2.ppf(.975,n-1)),
                              sqrt((n-1)*sd**2/chi2.ppf(.025,n-1))],
        sd_normal_model_upper95=sqrt((n-1)*sd**2/chi2.ppf(.05,n-1)),
        variance_uncertainty='chi-square normal-block model; not a heavy-tail guarantee')


def rounded_count(sd):
    raw = max(4096,ceil(((norm.ppf(.975)+norm.ppf(.8))*sd/10)**2))
    return 2**ceil(log2(raw))


def readout(results, plan, out, elapsed, memory):
    panels = {}
    for panel in plan['roots']:
        rows = [r for r in results if r['panel']==panel]
        expected = 128 if panel=='direct' else 32
        if len(rows) != expected:
            raise ValueError('Incomplete fixed calibration')
        values = [r['score_bb100'] for r in rows]
        entry = variance_summary(values)
        times = [r['seconds'] for r in rows]
        searched = [d for r in rows for d in r['search_rows']]
        causes = Counter(str(d['fallback_cause'] or 'unspecified') for d in searched if d['fallback'])
        entry.update(hands=sum(r['hands'] for r in rows),
            block_worker_seconds=dict(mean=mean(times),p95=float(np.percentile(times,95)),max=max(times)),
            primary_worker_seconds_mean=mean(r['primary_seconds'] for r in rows),
            search_worker_seconds_mean=mean(r['search_seconds'] for r in rows),
            searched_decisions=len(searched),
            search_decisions_per_search_arm_hand=len(searched)/(2*len(rows)),
            changed_river_actions=sum(d['changed_action'] for d in searched),
            fallback_decisions=sum(d['fallback'] for d in searched),fallback_causes=dict(causes),
            timeout_decisions=sum(n for cause,n in causes.items() if 'timeout' in (cause or '')),
            variance_precision_warning='sparse/zero pilot variance cannot exclude unobserved tails' if entry['nonzero_blocks'] < 5 else None,
            successful_actual_decisions_missing_ev=sum(not d['fallback'] and d['model_ev_unavailable'] is not None for d in searched))
        panels[panel] = entry
    direct = panels['direct']
    sparse = direct['nonzero_blocks'] < 20 or direct['sd_bb100']==0
    sd = direct['sd_normal_model_upper95']
    plugin = direct['sd_bb100']
    if sparse:
        sd = max(sd,HISTORICAL_SD)
        plugin = max(plugin,HISTORICAL_SD)
    count = rounded_count(sd)
    scripts = 2**ceil(log2(max(256,ceil(count/64))))
    counts = {panel:count if panel=='direct' else scripts for panel in panels}
    # Durations include the primary, independent event replay, fresh player
    # reproduction and full per-hand byte verification under measured contention.
    worker_seconds = sum(counts[p]*panels[p]['block_worker_seconds']['mean'] for p in panels)
    conservative_worker_seconds = sum(counts[p]*(panels[p]['block_worker_seconds']['mean']
        +float(t.ppf(.95,panels[p]['blocks']-1))*stdev([r['seconds'] for r in results if r['panel']==p])/sqrt(panels[p]['blocks'])) for p in panels)
    total_search = sum(counts[p]*panels[p]['search_worker_seconds_mean'] for p in panels)
    setup = max((json.loads(p.read_text())['load_seconds'] for p in (out/'workers').glob('*.json')),default=0)
    m4 = 1.25*(setup+worker_seconds/plan['workers'])
    performance = plan['performance']
    mean_sequential = performance['sequential_mean_seconds']
    mean_calibration = performance['chosen_concurrent_mean_seconds']
    # Undo the measured fixed-spot contention ratio to estimate one M4 worker;
    # expose that calibration workloads can have different tree/CPU distributions.
    one_worker = max(0,worker_seconds-total_search) + total_search*mean_sequential/mean_calibration
    cygnus = 1.25*(setup+one_worker/CYGNUS_TRAINER_PROXY)
    precision = {}
    for panel,entry in panels.items():
        if panel=='direct':continue
        precision[panel] = dict(blocks=scripts,
            nominal95_halfwidth_proxy=float(t.ppf(.975,scripts-1))*entry['sd_normal_model_upper95']/sqrt(scripts),
            seven_way95_halfwidth_proxy=float(t.ppf(1-.05/14,scripts-1))*entry['sd_normal_model_upper95']/sqrt(scripts))
    bytes_now = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    projected_bytes = sum(counts[p]/panels[p]['blocks'] *
        sum(f.stat().st_size for f in (out/'blocks'/p).rglob('*') if f.is_file()) for p in panels)
    primary_hand_count = sum(r['hands'] for r in results)
    return dict(status='verified-calibration-proposal-only',source=plan['source'],
        calibration_blocks=len(results),primary_hands=primary_hand_count,
        all_hands_replayed_and_reproduced=True,native_reproduction_solves=0,
        final_hands=0,realized_mean_effects_withheld=True,panels=panels,
        measured_seconds=elapsed,calibration_budget_seconds=7200,
        peak_family_rss_bytes=memory.peak,minimum_host_available_bytes=memory.minimum_available_bytes,
        memory_target_met=memory.peak <= 16*1024**3-3_000_000_000 and memory.minimum_available_bytes >= 3_000_000_000,
        sparse_tail_planning_floor_applied=sparse,planning_sd_bb100=sd,
        plugin_direct_blocks=rounded_count(plugin),proposed_direct_blocks=count,
        proposed_panel_blocks_per_script=scripts,precision_proxies=precision,
        direct_sensitivity_bb100=10,two_sided_alpha=.05,approximate_target_power=.8,
        normal_model_power_at_proposed_count=float(nct.sf(t.ppf(.975,count-1),count-1,10*sqrt(count)/sd)
            +nct.cdf(-t.ppf(.975,count-1),count-1,10*sqrt(count)/sd)),
        m4_runtime_quote=dict(seconds_with25percent_headroom=m4,workers=plan['workers'],
            basis='measured calibration block costs at the admitted measured worker count; verification/setup included',
            block_cost_upper95_normal_model_seconds_with25percent_headroom=1.25*(setup+conservative_worker_seconds/plan['workers'])),
        cygnus_runtime_quote=dict(seconds_with25percent_headroom_proxy=cygnus,
            sensitivity_range_seconds=[1.25*(setup+one_worker),cygnus],native_threads=6,
            simultaneous_native_searches=1,busy_thread_limit=8,available_only_after_pr301=True,
            search_rate_measured=False,hardware_proxy=CYGNUS_TRAINER_PROXY,
            basis='PR283 trainer throughput ratio, not a search benchmark; conditional estimate only',
            required_before_any_final_hands='post-301 fixed-spot build/parity/cost/thermal qualification and owner acceptance of the resulting quote'),
        evidence_bytes=bytes_now,projected_final_evidence_bytes=projected_bytes,
        evidence_projection_scope='observed fixed 1 percent sampling; rare larger bodies and failures remain an admission risk',
        final_counts_owner_approved=False,no_solver_tuning_from_calibration=True)
