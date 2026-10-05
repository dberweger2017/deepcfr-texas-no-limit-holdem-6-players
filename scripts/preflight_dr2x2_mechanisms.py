"""Outcome-blind A/C stored-average and common-law river capacity admission."""
import argparse
import gc
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.evaluate_hu20_cards_v2 import hand
from scripts.evaluate_hu20_river import root_hand
from scripts.evaluate_hu20_stackoff import swap_bytes
from scripts.hu20_platform_pilot import peak_rss, write
from src.arena.schedule import digest
from src.blueprint.river_cfr import profile_quality
from src.blueprint.river_game import river_root_history
from src.diagnostics.cfr_average import extract, audit, DiagnosticAverage
from src.diagnostics.history_river import CommonRiverGame, common_ranges, policy_profile, LAW, PROJECTION
from src.diagnostics.saved_hu20 import load_saved, file_hash


def run(plan, out):
    if plan['quality_outputs_permitted'] or plan['law'] != LAW or plan['projection'] != PROJECTION:
        raise ValueError('Mechanism admission must use the frozen blind law/projection')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'plan.json', plan)
    start = perf_counter()
    swap_before = swap_bytes()
    checked_at = -1.0
    def guard():
        nonlocal checked_at
        elapsed = perf_counter()-start
        if elapsed >= plan['limits']['timing_max_seconds']:
            raise TimeoutError('Mechanism timing guard')
        if elapsed-checked_at < 1:
            return
        checked_at = elapsed
        if peak_rss() >= plan['limits']['max_rss_gib']*2**30:
            raise MemoryError('Mechanism RSS guard')
        if shutil.disk_usage(out).free < plan['limits']['min_free_disk_gib']*2**30:
            raise OSError('Mechanism disk guard')
        if swap_bytes()-swap_before > plan['limits']['max_swap_growth_gib']*2**30:
            raise MemoryError('Mechanism swap guard')
    geometry, models = [], []
    failure = None
    first_game = None
    try:
        for case in plan['river_roots']:
            guard()
            began = perf_counter()
            root = river_root_history(root_hand(case['seed'],case['raise_to']).events)
            game = CommonRiverGame(root,common_ranges(root),raise_cap=plan['river_raise_cap'],
                                   max_public_nodes=plan['max_public_nodes'])
            setup = perf_counter()-began
            dummy = {n.id:np.full((len(game.holdings[game.seats.index(n.actor)]),len(n.menu)),1/len(n.menu))
                     for n in game.nodes if n.actor is not None}
            began = perf_counter()
            # A uniform dummy profiles the exact apparatus without candidate quality.
            profile_quality(game,dummy)
            quality_seconds = perf_counter()-began
            geometry.append({'case':case,'public_nodes':len(game.nodes),'root_pot_chips':game.root_pot,
                             'holdings':[len(h) for h in game.holdings],
                             'compatible_joint_deals':int((game.joint>0).sum()),
                             'range_sha256':digest(common_ranges(root)),
                             'setup_seconds':setup,'dummy_quality_seconds':quality_seconds})
            if case['seed']==plan['candidate_projection_timing_root']:
                first_game = game
            del game, dummy
            gc.collect()
        write(out/'geometry.json',geometry)
        for spec in plan['models']:
            guard()
            average_path=out/f"{spec['cell']}-{spec['seed']}.average.jsonl.gz"
            began=perf_counter()
            exported=extract(Path(spec['checkpoint_path']),spec,average_path,expected_schema=spec['abstraction'])
            extraction_seconds=perf_counter()-began
            began=perf_counter()
            checked=audit(Path(spec['checkpoint_path']),Path(spec['path']),average_path,spec,
                          exported['sha256'],expected_schema=spec['abstraction'])
            audit_seconds=perf_counter()-began
            if checked['status']!='verified':raise ValueError('Average audit failed')
            # Do not persist candidate current-average distances in capacity admission.
            verified_entries=checked['all_nodes_verified']
            del checked
            gc.collect()
            projections=[]
            for readout in ('current','average'):
                guard();began=perf_counter()
                if readout=='current':
                    source,visits=load_saved(spec,Path('/'),guard,expected_schema=spec['abstraction'])
                else:
                    source=DiagnosticAverage(average_path,exported['sha256'],expected_schema=spec['abstraction'])
                    visits=source.visits
                loaded=perf_counter()-began
                began=perf_counter()
                profile,telemetry=policy_profile(first_game,source,guard)
                projections.append({'readout':readout,'load_seconds':loaded,
                                    'projection_seconds':perf_counter()-began,'queries':telemetry['queries']})
                del profile,telemetry
                if readout=='average':
                    panels=[]
                    for panel in plan['panels']:
                        seconds=[]
                        for block in range(plan['timing_blocks']):
                            for rotation in (0,1):
                                guard();began=perf_counter()
                                row=hand(source,visits,spec,{**panel,'root':panel['timing_root']},block,rotation,plan,resource_only=True)
                                if row['target_chips'] is not None or row['net_chips_by_seat'] is not None:
                                    raise ValueError('Timing exposed a payoff')
                                seconds.append(perf_counter()-began)
                        panels.append({'panel':panel['name'],'hands':len(seconds),'seconds':sum(seconds),
                                       'max_hand_seconds':max(seconds)})
                del source,visits
                gc.collect()
            result={'cell':spec['cell'],'seed':spec['seed'],'checkpoint_sha256':spec['checkpoint_sha256'],
                    'current_sha256':spec['sha256'],'average_sha256':exported['sha256'],
                    'average_bytes':average_path.stat().st_size,'verified_entries':verified_entries,
                    'extraction_seconds':extraction_seconds,'audit_seconds':audit_seconds,
                    'projections':projections,'average_panels':panels}
            models.append(result)
            write(out/f"{spec['cell']}-{spec['seed']}.json",result)
            write(out/'progress.json',{'completed_models':len(models),'seconds':perf_counter()-start,'peak_rss_bytes':peak_rss()})
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}'
    playing={}
    river=None
    if not failure:
        for panel in plan['panels']:
            playing[panel['name']]=plan['projection_multiplier']*sum(
                next(p['max_hand_seconds'] for p in m['average_panels'] if p['panel']==panel['name'])*2*panel['average_blocks']
                for m in models)
        river=plan['projection_multiplier']*(sum(g['setup_seconds'] for g in geometry)+
              12*sum(g['dummy_quality_seconds'] for g in geometry)+
              3*sum(p['projection_seconds'] for m in models for p in m['projections']))
    summary={'status':'failed-retained' if failure else 'complete','failure':failure,'plan_sha256':digest(plan),
             'outcome_blind':True,'candidate_quality_computed':False,'strength_outcomes_inspected':False,
             'geometry':geometry,'models':models,'average_playing_projection_by_panel_seconds':playing,
             'average_playing_projection_seconds':sum(playing.values())+plan['projection_multiplier']*sum(
                 p['load_seconds'] for m in models for p in m['projections'] if p['readout']=='average'),
             'river_projection_seconds':river,'seconds':perf_counter()-start,'peak_rss_bytes':peak_rss(),
             'swap_growth_bytes':swap_bytes()-swap_before,
             'projection_limitations':'capacity extrapolation, not a guarantee; excludes final replay/tail serialization and Linux/provider differences'}
    write(out/'summary.json',summary)
    write(out/'manifest.json',{p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in out.iterdir() if p.is_file()})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=run(json.loads(args.plan.read_text()),args.out)
    print(json.dumps({'status':result['status'],'seconds':result['seconds']}),flush=True)
    raise SystemExit(result['status']!='complete')
