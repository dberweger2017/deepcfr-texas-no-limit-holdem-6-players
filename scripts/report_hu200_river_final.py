"""One prespecified readout after the complete final verification barrier."""

from collections import Counter
from math import sqrt
from statistics import mean, stdev
import json

import numpy as np
from scipy.stats import t

from src.arena.river_block_storage import durable, unpack
from src.arena.river_final_schedule import COUNTS


def interval(values,alpha=.05):
    values=list(values)
    if len(values)<2 or not np.isfinite(values).all():raise ValueError('Invalid final block scores')
    center=mean(values);sd=stdev(values);half=float(t.ppf(1-alpha/2,len(values)-1))*sd/sqrt(len(values))
    return dict(blocks=len(values),mean_bb100=center,sd_bb100=sd,se_bb100=sd/sqrt(len(values)),interval_bb100=[center-half,center+half],alpha=alpha)


def report(out,journal,plan,memory):
    proof=json.loads((out/'all-final-hands-verified.json').read_text())
    if proof['primary_hands']!=319488 or proof['counts']!=COUNTS:
        raise ValueError('No complete final verification barrier')
    scores={p:[] for p in COUNTS};arms={p:{'search':[],'plain':[]} for p in COUNTS}
    telemetry={p:[] for p in COUNTS}
    for panel,blob in journal.connection.execute('SELECT panel,summary FROM blocks ORDER BY panel,number'):
        row=unpack(blob);scores[panel].append(row['score_bb100']);telemetry[panel].extend(row['search_rows'])
        arms[panel]['search'].append(row['search_score_bb100'])
        if panel!='direct':arms[panel]['plain'].append(row['plain_score_bb100'])
    panels={}
    for panel,values in scores.items():
        expected=COUNTS[panel]
        if len(values)!=expected:raise ValueError('Incomplete final scores')
        result=dict(nominal=interval(values),search_arm=interval(arms[panel]['search']))
        if panel!='direct':
            result.update(adjusted_seven_way=interval(values,.05/7),plain_arm=interval(arms[panel]['plain']))
            upper=result['adjusted_seven_way']['interval_bb100'][1]
            result['adjusted_negative_effect']=upper<0
            result['adjusted_material_regression_over10']=upper < -10
            result['material_minus10_not_excluded']=result['adjusted_seven_way']['interval_bb100'][0]<=-10
        rows=telemetry[panel];deltas=[r['model_ev_delta_chips'] for r in rows if r['model_ev_delta_chips'] is not None]
        causes=Counter(r['fallback_cause'] or 'unspecified' for r in rows if r['fallback'])
        times=[r['complete_seconds'] for r in rows]
        native=[r['native_seconds'] for r in rows if r.get('native_seconds') is not None]
        trees=[r['tree_nodes'] for r in rows if r.get('tree_nodes') is not None]
        model=[r['model_ev_chips'] for r in rows if r.get('model_ev_chips') is not None]
        result['telemetry']=dict(searched_decisions=len(rows),changed_actions=sum(r['changed_action'] is True for r in rows),
            fallbacks=sum(causes.values()),fallback_causes=dict(causes),
            timeouts=sum(v for k,v in causes.items() if 'timeout' in k),
            changed_fraction=sum(r['changed_action'] is True for r in rows)/len(rows) if rows else None,
            decisions_per_search_arm_hand=len(rows)/(2*COUNTS[panel]),
            native_seconds=dict(p50=float(np.median(native)),p95=float(np.percentile(native,95)),max=max(native)) if native else None,
            tree_nodes=dict(p50=float(np.median(trees)),p95=float(np.percentile(trees,95)),max=max(trees)) if trees else None,
            model_ev_chips=dict(count=len(model),mean=mean(model),median=float(np.median(model))) if model else None,
            peak_searched_decision_family_rss_bytes=max((r.get('sampled_peak_family_rss_bytes') or 0 for r in rows),default=0),
            successful_missing_ev=sum(not r['fallback'] and r['model_ev_unavailable'] is not None for r in rows),
            seconds=dict(p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95)),max=max(times)) if times else None,
            model_ev_delta_chips=dict(count=len(deltas),mean=mean(deltas),median=float(np.median(deltas)),
                p05=float(np.percentile(deltas,5)),p95=float(np.percentile(deltas,95))) if deltas else None)
        panels[panel]=result
    direct=panels['direct']['nominal']['interval_bb100']
    value=dict(status='final-primary-readout-complete-diagnostics-pending',counts=COUNTS,primary_hands=319488,
        direct_gain_established=direct[0]>0,direct_regression_established=direct[1]<0,
        panels=panels,interval_method='Student t over independent duplicate/paired blocks; panel nominal95 and Bonferroni seven-way95',
        uncertainty_scope='one frozen blend and deal/action randomness; no training-seed population claim or heavy-tail guarantee',
        model_ev_scope='same solved continuation estimates, not realized or causal action gains',
        global_verification=proof,peak_family_rss_bytes=memory.peak,minimum_host_available_bytes=memory.minimum_available_bytes,
        calibration_excluded=True,diagnostics_required=True,merge_owner_decision=True)
    durable(out/'strength-summary.json',value)
    print(json.dumps(dict(status=value['status'],direct=panels['direct']['nominal'],pot_pressure=panels['pot_pressure'])),flush=True)
    return value
