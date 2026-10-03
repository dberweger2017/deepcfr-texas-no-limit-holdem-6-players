"""Report a frozen turn campaign without adaptive root or checkpoint selection."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.turn_campaign import METRICS
from src.diagnostics.turn_report import common_rows,intervals,decision


def selected_folds(rows,*,expected_total=None,seed=202610020208,resamples=2000):
    grouped=defaultdict(list)
    for row in rows:
        for node in row['selected_lbr_nodes']:
            identity=(row['spot'],node['source_sha256'],node['source_line'],node['action_index'])
            grouped[identity].append((row,node))
    roots=defaultdict(list);total=0.;covered=0.
    for identity,values in grouped.items():
        weight=1/values[0][0]['inclusion_probability'];total+=weight
        if len(values)!=6 or any(n['fold_bp'] is None or n['fold_eq'] is None for r,n in values):continue
        covered+=weight
        roots[identity[0]].append((weight,np.mean([n['fold_bp'] for r,n in values]),np.mean([n['fold_eq'] for r,n in values])))
    if expected_total is not None:total=expected_total
    if not covered:return {'fold_bp':None,'fold_eq':None,'coverage':0.,'ci95_difference':None,'H0':False,'expected_weight':total,'covered_weight':0.}
    keys=sorted(roots);values=[];weights=[];cells=defaultdict(list)
    for i,key in enumerate(keys):
        items=roots[key];w=np.asarray([x[0] for x in items]);values.append(np.average(np.asarray([x[1:] for x in items]),axis=0,weights=w));weights.append(w.sum())
        cells[tuple(next(r['stratum'] for r in rows if r['spot']==key))].append(i)
    values=np.asarray(values);weights=np.asarray(weights);mean=np.average(values,axis=0,weights=weights);ci=None
    if all(len(c)>=2 for c in cells.values()):
        rng=np.random.default_rng(seed);draws=np.concatenate([np.asarray(c)[rng.integers(0,len(c),(resamples,len(c)))] for _,c in sorted(cells.items())],axis=1)
        sampled=(values[draws]*weights[draws,None]).sum(axis=1)/weights[draws].sum(axis=1)[:,None]
        ci=np.quantile(sampled[:,0]-sampled[:,1],[.025,.975]).tolist()
    return {'fold_bp':float(mean[0]),'fold_eq':float(mean[1]),'difference_points':float(100*(mean[0]-mean[1])),
            'coverage':covered/total,'expected_weight':total,'covered_weight':covered,'independent_roots':len(keys),'ci95_difference':ci,
            'H0':covered/total>=.9 and abs(mean[0]-mean[1])<=.03,'seed':seed,
            'scope':'common six-export, positive-equilibrium-reach selected turn bet nodes; inverse root-inclusion weights'}


def fmt(value):return '—' if value is None else f'{value:.4f}'
def interval(summary,name):
    ci=summary.get('ci95',{}).get(name)
    return fmt(summary.get('means',{}).get(name))+(' ['+', '.join(fmt(v) for v in ci)+']' if ci else ' [CI unavailable]')


def report(protocol,run,out):
    config=json.loads(Path(protocol).read_text());run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    terminal=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else json.loads((run/'failure.json').read_text())
    rows=[];paths={}
    for path in sorted((run/'spots').glob('*/result.json')):
        row=json.loads(path.read_text());paths[row['set'],row['spot'],row['policy']]=path
        rows.append({k:v for k,v in row.items() if k not in ('overfold_groups','fold_nodes')})
    admitted={};exclusions={}
    for group in ('A','B'):
        admitted[group],exclusions[group]=common_rows([r for r in rows if r['set']==group],config['policies'])
        # An interrupted campaign must count never-started frozen roots too.
        corpus_path=Path(protocol).resolve().parents[2]/config['corpus'][group]['path']
        present={r['spot'] for r in rows if r['set']==group}
        for root in json.loads(corpus_path.read_text())['roots']:
            if root['spot'] not in present:
                exclusions[group].append({'spot':root['spot'],'present_policies':[],
                    'expected_policies':[p['name'] for p in config['policies']],
                    'reason':'no completed profile from the frozen campaign'})
    summaries={}
    for group in ('A','B'):
        views={'pooled':admitted[group]}
        for seed in sorted({p['seed'] for p in config['policies']}):views[str(seed)]=[r for r in admitted[group] if r['lineage']==seed]
        for strategy in ('current','stored-average'):views[strategy]=[r for r in admitted[group] if r['strategy']==strategy]
        for name,selected in views.items():summaries[group+'/'+name]=intervals(selected)
        for target in (0,1):
            selected=[]
            for row in admitted[group]:
                metric={r['metric']:r for r in row['target_metrics'] if r['target_solver_seat']==target}
                converted={n:metric[n]['gain_bb'] for n in METRICS}|{n+'_pct_pot':metric[n]['gain_pct_pot'] for n in METRICS}
                converted['alias_difference_bb']=converted['e_v1proj']-converted['e_v1proj_line']
                selected.append(row|converted)
            summaries[group+('/OOP' if target==0 else '/IP')]=intervals(selected)
    # Secondary intervals use only complete empirical support, and compare to a
    # matched primary subset. Partial-support values remain in per-spot evidence.
    secondary=[]
    for row in admitted['B']:
        data=row['secondary_metrics']
        if len(data)!=10 or any(r['gain_bb'] is None or r['best_response'].get('retained_fraction',0)<1-1e-7 for r in data):continue
        converted={n:np.mean([r['gain_bb'] for r in data if r['metric']==n]) for n in METRICS}
        converted.update({n+'_pct_pot':10000*converted[n]/row['root_pot_chips'] for n in METRICS})
        converted['alias_difference_bb']=converted['e_v1proj']-converted['e_v1proj_line'];secondary.append(row|converted)
    sec_common,sec_excluded=common_rows(secondary,config['policies']);sec_summary=intervals(sec_common)
    selected=selected_folds(admitted['A'],expected_total=config['selected_lbr_decision_weight']);classification=decision(summaries['B/pooled'],turn_fraction=config['lbr_live_turn_fraction'])
    if terminal.get('status')!='completed':
        classification['classification']='incomplete frozen campaign; no hypothesis decision'
        selected['H0']=False
    groups=defaultdict(lambda:{'hand_contexts':0,'reach_mass':0.,'fold_bp_mass':0.,'fold_eq_mass':0.,'excess_fold_mass':0.})
    with gzip.open(out/'overfold-groups.jsonl.gz','wt') as target:
        for group in ('A','B'):
            for row in admitted[group]:
                raw=json.loads(paths[group,row['spot'],row['policy']].read_text())
                for item in raw['overfold_groups']:
                    target.write(json.dumps({'set':group,'spot':row['spot'],'policy':row['policy'],**item},sort_keys=True)+'\n')
                    key=(group,item['equity_decile']);totals=groups[key];weight=row['reach_weight']/6
                    totals['hand_contexts']+=item['hands']
                    for name in ('reach_mass','fold_bp_mass','fold_eq_mass','excess_fold_mass'):totals[name]+=weight*item[name]
    gains={'protocol_sha256':file_hash(protocol),'terminal':terminal,'summaries':summaries,'decision':classification,
           'selected_turn_overfold':selected,'secondary_full_support':sec_summary,'secondary_excluded':sec_excluded,
           'secondary_full_support_jobs':len(secondary),'common_exclusions':exclusions,
           'overfold_deciles':[{'set':g,'equity_decile':d,**v} for (g,d),v in sorted(groups.items())],
           'overfold_groups_sha256':file_hash(out/'overfold-groups.jsonl.gz'),'completed_results':len(rows),
           'limitations':['Ranges condition on preflop and flop play; errors before the turn are excluded.',
             'Feasible projections are not abstraction equilibria; a high R is a heuristic, not an abstraction lower bound.',
             'Per-turn equity buckets are more favourable than global blueprint buckets.',
             'Within-root line pooling cannot measure inherited preflop/flop aliases across different turn roots.',
             'Signed alias cost normalizes blueprint loss; projection differences are not a causal loss decomposition.',
             'Secondary baseline reweights the primary equilibrium strategy, without solving a different-range equilibrium.',
             'Common complete-case exclusions can change the represented root population.',
             'Turn/river results do not answer the original flop question.']}
    atomic_json(out/'summary.json',gains)
    text=['# HU20 exact turn check', '',f"Campaign status: **{terminal.get('status','stopped')}**. {len(rows)} / {config['jobs_total']} frozen spot-policy jobs produced atomic results.",
          '', 'Turn/river results do not answer the original flop question. The separate history-alias audit is in draft #146.', '',
          '## Set B values', '', 'Reach-weighted BB, bootstrap 95% intervals over independent roots. All six exports use the same eligible-root intersection.', '',
          '| Policy group | Roots | Blueprint | Full v1 | Per-line v1 | Equity 50 | Equity 200 |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,summary in summaries.items():
        if name.startswith('B/'):
            text.append('| '+name[2:]+' | '+str(summary['independent_roots'])+' | '+' | '.join(interval(summary,n) for n in METRICS)+' |')
    text+=['', '## Frozen decision', '',classification['classification']+'.', '',
           f"R = {fmt(classification['R'])}; signed alias cost = {fmt(classification['alias_cost'])}; descriptive turn share of the 0.65 BB/hand LBR loss = {fmt(classification['descriptive_turn_share'])}.",
           '', 'Ratios use pooled means with paired bootstrap draws, rather than averages of per-root ratios. Full JSON contains BB/%-pot intervals, lineage and position splits, ratio intervals and secondary support coverage.', '',
           '## Selected turn folds', '',f"Blueprint = {fmt(selected['fold_bp'])}; equilibrium = {fmt(selected['fold_eq'])}; common positive-reach coverage = {fmt(selected['coverage'])}.", '',
           'H0-turn criterion met: '+str(selected['H0'])+'. This criterion concerns the sampled turn nodes only.', '',
           '## Overfold by equity decile', '', 'These are hand-contexts with positive blueprint excess-fold probability. The compressed full table retains every v1 key; raw solver responses retain individual holdings.', '',
           '| Set | Decile | Hand-contexts | Reach mass | Excess-fold mass |', '| --- | ---: | ---: | ---: | ---: |']
    for row in gains['overfold_deciles']:text.append(f"| {row['set']} | {row['equity_decile']} | {row['hand_contexts']} | {fmt(row['reach_mass'])} | {fmt(row['excess_fold_mass'])} |")
    text+=['', '## Limits and exclusions', '',*['- '+s for s in gains['limitations']], '',
           f"Common-intersection exclusions: A={len(exclusions['A'])}, B={len(exclusions['B'])}. Full-support secondary jobs: {len(secondary)}.", '',
           'The final repository report must also retain the frozen corpus, resource inventory, preflight failures and main-run failures. No training, promotion, rental or automatic merge.']
    (out/'report.md').write_text('\n'.join(text)+'\n')
    return gains


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','run','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report(a.protocol,a.run,a.out)
if __name__=='__main__':main()
