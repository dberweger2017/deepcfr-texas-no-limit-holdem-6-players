"""Paired root bootstrap and predeclared turn-only decision rule."""
from collections import defaultdict
import numpy as np
from src.diagnostics.turn_campaign import METRICS

COLUMNS=METRICS+tuple(n+'_pct_pot' for n in METRICS)+('alias_difference_bb',)


def common_rows(rows,policies):
    grouped=defaultdict(list)
    for row in rows:grouped[row['spot']].append(row)
    names={p['name'] for p in policies};eligible=[];excluded=[]
    for spot,values in sorted(grouped.items()):
        actual={r.get('policy') for r in values}
        if actual!=names or len(values)!=len(names) or not all(r.get('decision_eligible',False) for r in values):
            excluded.append({'spot':spot,'present_policies':sorted(actual),'expected_policies':sorted(names),
                             'reason':'missing/unsupported/nonconverged profile in common six-export intersection'})
        else:eligible.extend(values)
    return eligible,excluded


def intervals(rows,*,seed=202610020207,resamples=2000):
    grouped=defaultdict(list)
    for r in rows:grouped[r['spot']].append(r)
    keys=sorted(grouped)
    if not keys:return {'independent_roots':0,'means':{},'ci95':{},'ratios':{},'reason':'no common eligible roots'}
    values=np.asarray([[np.mean([r[c] for r in grouped[k]]) for c in COLUMNS] for k in keys])
    weights=np.asarray([grouped[k][0]['reach_weight'] for k in keys],dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights<=0).any():raise ValueError('Invalid report input')
    means=np.average(values,axis=0,weights=weights);cells=defaultdict(list)
    for i,k in enumerate(keys):cells[tuple(grouped[k][0]['stratum'])].append(i)
    estimands={'R':('e_v1proj','e_bp'),'alias_cost':('alias_difference_bb','e_bp'),
               'equity_headroom_ratio':('e_eq200','e_v1proj')}
    result={'independent_roots':len(keys),'means':dict(zip(COLUMNS,map(float,means))),
            'ci95':{},'ratios':{},'seed':seed,'resamples':resamples,
            'sampling_strata':{str(k):len(v) for k,v in sorted(cells.items())}}
    for name,(num,den) in estimands.items():
        n,d=means[COLUMNS.index(num)],means[COLUMNS.index(den)]
        result['ratios'][name]={'point':float(n/d) if d>0 else None,'ci95':None}
    if any(len(cell)<2 for cell in cells.values()):
        result['reason']='at least one sampling stratum has fewer than two eligible roots';return result
    rng=np.random.default_rng(seed)
    draws=np.concatenate([np.asarray(cell)[rng.integers(0,len(cell),(resamples,len(cell)))] for _,cell in sorted(cells.items())],axis=1)
    sampled=(values[draws]*weights[draws,None]).sum(axis=1)/weights[draws].sum(axis=1)[:,None]
    for i,c in enumerate(COLUMNS):result['ci95'][c]=np.quantile(sampled[:,i],[.025,.975]).tolist()
    for name,(num,den) in estimands.items():
        n=sampled[:,COLUMNS.index(num)];d=sampled[:,COLUMNS.index(den)];valid=d>0
        result['ratios'][name]['defined_resample_fraction']=float(valid.mean())
        # Do not manufacture a ratio interval by dropping undefined draws.
        if valid.all():result['ratios'][name]['ci95']=np.quantile(n/d,[.025,.975]).tolist()
    return result


def decision(summary,*,turn_fraction,loss_bb_per_hand=.65):
    means=summary['means'];r=summary['ratios'].get('R',{}).get('point')
    bp=means.get('e_bp');share=None if bp is None else turn_fraction*bp/loss_bb_per_hand
    alias=summary['ratios'].get('alias_cost',{}).get('point')
    if summary['independent_roots']<16 or summary.get('reason') or r is None:
        label='insufficient eligible Set B coverage for the frozen decision'
    elif share<.10:label='H3-turn: little measured turn contribution; audit earlier streets, including the flop'
    elif r>=.7 and share>=.25:label='H1-turn heuristic: high feasible-v1 projection loss'
    elif r<=.3:label='H2-turn: a feasible-v1 witness to substantially better turn/river play'
    else:label='mixed turn/river evidence'
    alias_label=('undefined' if alias is None else 'material public-line pooling' if alias>=.25
                 else 'small signed public-line pooling difference' if alias<=.10 else 'intermediate public-line pooling')
    return {'classification':label,'R':r,'alias_cost':alias,'alias_interpretation':alias_label,
            'descriptive_turn_share':share,'live_turn_fraction':turn_fraction,'lbr_loss_bb_per_hand':loss_bb_per_hand,
            'limitation':'conditional turn/river diagnosis; does not classify the original flop hypotheses'}
