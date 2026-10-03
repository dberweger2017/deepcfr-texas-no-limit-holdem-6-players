"""Outcome-blind job ordering and compact turn result interpretation."""
from collections import defaultdict
import json
from random import Random
from src.diagnostics.flop_check import decode_descriptor,factored_key,line_key

METRICS=('e_bp','e_v1proj','e_v1proj_line','e_eq50','e_eq200')


def jobs(corpus,policies,seed):
    """Round-robin sets/strata, rotating the policy at every root and round."""
    rng=Random(seed);cells=defaultdict(list)
    for group in ('A','B'):
        for root in corpus[group]['roots']:cells[group,root['kind'],root['button']].append(root)
    for rows in cells.values():rows.sort(key=lambda r:r['spot']);rng.shuffle(rows)
    root_order=[]
    while any(cells.values()):
        for cell in sorted(cells,key=lambda c:(c[1],c[2],c[0])):
            if cells[cell]:root_order.append((cell[0],cells[cell].pop()))
    output=[]
    for round_no in range(len(policies)):
        for index,(group,root) in enumerate(root_order):
            policy=(index+round_no)%len(policies)
            output.append({'set':group,'root':root,'policy_index':policy,
                           'job':f'{group}-{root["spot"]}-{policy}'})
    return output


def secondary_ranges(root,request,stored):
    from scripts.select_turn_check_spots import preflop_line
    from src.diagnostics.turn_check import replay_root
    line=preflop_line(replay_root(root),root['button']);output=[];coverage=[]
    for seat in request['seat_map']:
        position=(seat-root['button'])%2
        source=next((r for r in stored['empirical_preflop_ranges']
                     if r['preflop_line']==line and r['position']==position),None)
        rows=[] if source is None else source['hands']
        surviving=[r for r in rows if not set(r['hand']).intersection(root['board'])]
        mass=sum(r['count'] for r in surviving)
        output.append([{'hand':r['hand'],'weight':r['count']/mass} for r in surviving] if mass else [])
        coverage.append({'position':position,'preflop_samples':sum(r['count'] for r in rows),
                         'board_compatible_samples':mass,'distinct_holdings':len(surviving)})
    return output,coverage


def summarize(job,policy,request,rows,runtime,stored):
    root=job['root'];final=rows[-1]
    if final.get('status')!='solved':raise ValueError('Incomplete compact turn solve')
    v1=next(r for r in rows if r.get('gate')=='V1')
    if not v1['passed']:raise ValueError('Native tree gate failed')
    primary=[r for r in rows if r['event']=='compact_metric']
    if len(primary)!=10:raise ValueError('Missing target-only compact metrics')
    residual=final['exploitability_pct_pot'];metrics={}
    for name in METRICS:
        values=[r['gain_bb'] for r in primary if r['metric']==name]
        if len(values)!=2:raise ValueError('Missing target seat')
        metrics[name]=sum(values)/2
        metrics[name+'_pct_pot']=sum(r['gain_pct_pot'] for r in primary if r['metric']==name)/2
    bp=metrics['e_bp'];difference=metrics['e_v1proj']-metrics['e_v1proj_line']
    # Root ranges are separately normalized; convert raw compatible reach to
    # probability under their blocker-conditioned product law.
    other=request['ranges'][1];total_other=sum(r['weight'] for r in other);marginal=defaultdict(float);same={}
    for row in other:
        a,b=row['hand'];w=row['weight'];marginal[a]+=w;marginal[b]+=w;same[tuple(sorted((a,b)))]=w
    normalizer=sum(row['weight']*(total_other-marginal[row['hand'][0]]-marginal[row['hand'][1]]+same.get(tuple(sorted(row['hand'])),0)) for row in request['ranges'][0])
    if normalizer<=0:raise ValueError('No compatible root range mass')
    aggregation=next(r for r in rows if r['event']=='compact_aggregation')
    folds=aggregation['overfold_nodes'];by_node={r['node']:r for r in folds}
    selected=[]
    if job['set']=='A':
        node_ids={line_key(n['line']):i for i,n in enumerate(request['nodes'])}
        for decision in stored['decisions']:
            if decision['spot']!=root['spot']:continue
            node=node_ids.get(line_key(decision['line']));fold=by_node.get(node)
            if fold is None or fold['target_solver_seat']!=decision['target_solver_seat']:
                raise ValueError('Stored LBR decision missing from native turn fold nodes')
            selected.append({'node':node,'target_solver_seat':fold['target_solver_seat'],
                             'fold_bp':fold['fold_bp'],'fold_eq':fold['fold_eq'],
                             'source_sha256':decision['source_sha256'],'source_line':decision['source_line'],
                             'action_index':decision['action_index'],'reach_mass':fold['reach_mass']/normalizer})
    # The compact report retains decile/key totals; raw response retains all hands.
    groups=defaultdict(lambda:{'hands':0,'reach_mass':0.,'excess_fold_mass':0.,'fold_bp_mass':0.,'fold_eq_mass':0.})
    node_summaries=[]
    for row in folds:
        node=request['nodes'][row['node']]
        for hand in row['excess_fold_hands']:
            key=factored_key(node['template'],decode_descriptor(hand['v1_descriptor_code']))
            group=(row['target_solver_seat'],hand['equity_decile'],key)
            value=groups[group];value['hands']+=1;value['reach_mass']+=hand['reach_mass']/normalizer
            value['excess_fold_mass']+=hand['reach_mass']*hand['excess_fold_probability']/normalizer
            value['fold_bp_mass']+=hand['reach_mass']*hand['fold_bp']/normalizer
            value['fold_eq_mass']+=hand['reach_mass']*hand['fold_eq']/normalizer
        node_summaries.append({k:v for k,v in row.items() if k not in ('excess_fold_hands','reach_mass')}|{'line':node['line'],'reach_mass':row['reach_mass']/normalizer})
    total=sum(r['reach_mass'] for r in folds)
    secondary=[r for r in rows if r['event']=='compact_secondary_metric']
    eq=next(r for r in rows if r['event']=='both_blueprint_ev')
    return {'event':'spot_complete','set':job['set'],'spot':root['spot'],'lineage':policy['seed'],
            'strategy':policy['strategy'],'policy':policy['name'],'kind':root['kind'],'button':root['button'],
            'stratum':[root['kind'],root['button']],'reach_weight':root.get('reach_weight',root['multiplicity']),
            'root_multiplicity':root['multiplicity'],'inclusion_probability':root.get('inclusion_probability',1),
            'decision_eligible':residual<=.5,'target_metrics':primary,**metrics,
            'alias_difference_bb':difference,'alias_cost':difference/bp if bp>0 else None,
            'alias_cost_note':'signed ratio; undefined for nonpositive e_bp; no clipping',
            'equilibrium_residual_pct_pot':residual,'target_met':residual<=.2,
            'fold_bp':sum(r['reach_mass']*(r['fold_bp'] or 0) for r in folds)/total if total else None,
            'fold_eq':sum(r['reach_mass']*(r['fold_eq'] or 0) for r in folds)/total if total else None,
            'fold_conditioning':'common equilibrium joint reach; repeated bet nodes count as decisions',
            'fold_nodes':node_summaries,'selected_lbr_nodes':selected,
            'overfold_groups':[{'target_solver_seat':p,'equity_decile':decile,'v1_key':key,**value}
                               for (p,decile,key),value in sorted(groups.items())],
            'secondary_metrics':secondary,'secondary_coverage':request.get('secondary_coverage'),
            'secondary_baseline':'primary equilibrium strategy reweighted; unsupported empirical holdings disclosed',
            'both_blueprint_ev_chips':eq['current_ev_chips'],'solver_seat_map':request['seat_map'],
            'root_compatible_product_normalizer':normalizer,'root_pot_chips':request['pot'],'memory_estimates':{'uncompressed_bytes':final['uncompressed_bytes'],
            'compressed_bytes':final['compressed_bytes']},'runtime':runtime,'gates':[v1,{'gate':'V5',
            'passed':residual<=.5,'target_met':residual<=.2,'achieved_pct_pot':residual}],
            'source':policy,'fallback':'native; compressed; zero-weight holdings removed'}
