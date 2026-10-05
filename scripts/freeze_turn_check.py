"""Freeze stratified public turn roots using declared cost pilots, without loss values."""
import argparse
import json
from pathlib import Path
from scripts.select_flop_check_spots import select_strata
from src.diagnostics.flop_check import atomic_json,compile_tree,fixture_root
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.turn_check import root_record,replay_root
from src.game.types import Street


def freeze(a):
    populations={g:json.loads(path.read_text()) for g,path in (('A',a.a),('B',a.b))}
    pilot_ids={root_record(fixture_root(k,street=Street.TURN,seed=202610010901))['spot']
               for k in ('limped','min-raised','3-bet')}
    costs={}
    artifacts=a.repo/'docs/reports/hu20-exact-turn-check-artifacts'
    for kind,short in (('limped','limp'),('min-raised','min'),('3-bet','3bet')):
        path=artifacts/f'cost-preflight-{short}.json';data=json.loads(path.read_text())
        request,_=compile_tree(fixture_root(kind,street=Street.TURN,seed=202610010901))
        costs[kind]={'seconds':data['runtime']['elapsed_seconds'],'nodes':len(request['nodes']),
                     'source_sha256':file_hash(path)}
    counts=json.loads(a.counts.read_text())['counts_by_pot']
    selected={};paths={};selection={}
    for group,n,seed in (('A',16,202610020205),('B',32,202610020203)):
        population=[r for r in populations[group]['roots'] if r['spot'] not in pilot_ids]
        roots,strata=select_strata(population,n,seed)
        for root in roots:
            if root_record(replay_root(root))['spot']!=root['spot']:raise ValueError('Native root mismatch')
        result=populations[group]|{'roots':roots,'population_unique_roots':len(population),
            'source_population_sha256':file_hash(a.a if group=='A' else a.b),'strata':strata,
            'selection_seed':seed,'selected_unique_roots':n,'pilot_roots_excluded':sorted(pilot_ids),
            'weights':'root multiplicity / stratum inclusion probability; common roots across all six exports'}
        path=artifacts/f'corpus-{group}.json';atomic_json(path,result)
        selected[group]=roots;paths[group]={'path':str(path.relative_to(a.repo)),'sha256':file_hash(path)}
        selection[group]={'population_roots':len(population),'selected_roots':n,'strata':strata,
                          'selected_occurrences':sum(r['multiplicity'] for r in roots),
                          'represented_occurrence_weight':sum(r['reach_weight'] for r in roots)}
    def estimate(roots):
        seconds=0.
        for r in roots:
            reference=costs['min-raised' if r['kind']=='pot-raised' else r['kind']]
            seconds+=10+reference['seconds']*counts[str(r['pot'])]/reference['nodes']
        return 2*6*seconds
    seconds=estimate(selected['A']+selected['B'])
    if seconds>24*3600:raise ValueError('Declared cost estimate exceeds owner budget')
    evidence={'purpose':'outcome-blind corpus admission; no main root loss values inspected',
        'corpus':paths,'selection':selection,'preflight_costs':costs,'seconds_estimate_with_contingency':seconds,
        'hours_estimate_with_contingency':seconds/3600,'jobs_total':288,
        'formula':'2 contingency * 6 exports * sum(10 fixed export seconds + full pilot seconds * native node count / pilot nodes)',
        'tree_counts_sha256':file_hash(a.counts),'native_nodes_by_pot':counts,
        'all_A_plus_B16_hours':estimate(populations['A']['roots']+select_strata(populations['B']['roots'],16,202610020203)[0])/3600,
        'pilot_overlap':[]}
    atomic_json(artifacts/'corpus-admission.json',evidence)
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('a','b','counts'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repo',type=Path,default=Path.cwd());a=p.parse_args()
    print(json.dumps(freeze(a),sort_keys=True))
if __name__=='__main__':main()
