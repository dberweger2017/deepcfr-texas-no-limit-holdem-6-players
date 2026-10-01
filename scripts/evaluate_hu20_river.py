"""Bounded M1 fixed-sweep HU20 timing and paired playing experiments."""
import argparse
from collections import Counter
import gc
import gzip
import json
from pathlib import Path
from random import Random
import subprocess
from time import monotonic, perf_counter

from src.arena.catalog import Checkpoint
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import choices
from src.blueprint.artifact import FrozenBlueprint
from src.blueprint.hu20_river import (HU20RiverConfig, HU20RiverGame, HU20RiverPlayer,
    RiverProfileCache, peak_rss, public_identity, public_ranges)
from src.blueprint.river_cfr import RiverCFR, profile_quality
from src.blueprint.river_game import river_root_history
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.stackoff_tails import snapshot, hand_tails
from scripts.evaluate_hu20_cfr_average import opponent, summarize
from src.arena.runner import public_events
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street


def root_hand(seed, raise_to=None):
    hand = Hand.start(Table(('seat0','seat1'),(2000,2000),button=seed%2), hand_id='timing',seed=seed)
    raised = False
    while hand.observe(hand.actor).street != Street.RIVER:
        view = hand.observe(hand.actor)
        if view.street == Street.FLOP and raise_to is not None and not raised:
            action = Action(ActionKind.RAISE, raise_to); raised = True
        else:
            action = Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL)
        hand = hand.apply(action)
    return hand


def load(spec, inputs):
    path = inputs/spec['path']
    if path.stat().st_size != spec['bytes'] or file_hash(path) != spec['sha256']:
        raise ValueError('Frozen policy bytes differ before loading')
    source = FrozenBlueprint(Checkpoint(spec['name'],str(path),spec['sha256'],spec['format']),path)
    if (source.description['strategy'] != 'current' or source.description['training_seed'] != spec['seed']
            or source.description['iteration'] != spec['iteration'] or source.raise_cap is not None):
        raise ValueError('B500M current-policy identity differs')
    return source


def curve(source, plan, out, check):
    rows = []
    with (out/'curve.jsonl').open('x') as stream:
        for case in plan['roots']:
            check(); start = perf_counter()
            hand = root_hand(case['seed'],case['raise_to']); root = river_root_history(hand.events)
            ranges, coverage = public_ranges(source,root,check)
            game = HU20RiverGame(root,ranges); solver = RiverCFR(game)
            setup = perf_counter()-start
            for sweeps in plan['sweeps']:
                check(); began = perf_counter()
                result = solver.solve(max_sweeps=sweeps-solver.completed_sweeps,
                    deadline=plan['_deadline'],rss_limit_bytes=plan['rss_limit_bytes'])
                reached = result.completed_sweeps == sweeps
                solve_seconds = perf_counter()-began
                row = {'case':case,'root':public_identity(root),'root_pot_chips':game.root_pot,
                    'range_identity':digest(ranges),'range_coverage':coverage,'public_nodes':len(game.nodes),
                    'requested_sweeps':sweeps,'completed_sweeps':result.completed_sweeps,
                    'milestone_reached':reached,'stop_reason':result.stop_reason,
                    'setup_seconds':setup,'incremental_solve_seconds':solve_seconds,
                    'peak_rss_bytes':peak_rss(),'zero_average_rows':result.zero_average_denominators,
                    'compatible_joint_deals':int((game.joint>0).sum()),
                    'quality':profile_quality(game,result.average) if reached else None}
                rows.append(row);stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush()
                print(json.dumps(row),flush=True)
                if not reached: raise TimeoutError('Timing milestone unmet')
            del game,solver,result;gc.collect()
    return {'curve':rows}


def play(source,spec,panel,root,block,rotation,strategy,config,cache,check,deadline=None):
    deal = stream_seed(root,'test','deal',2,block)
    action_seed = stream_seed(root,'test','action',2,block,0); random = Random(action_seed)
    player = HU20RiverPlayer(source,action_seed,config,cache,deadline) if strategy == 'average' else None
    if player is not None:
        player.random = random  # One coupled action stream, including earlier streets.
    rival = opponent(panel,source,stream_seed(root,'test','opponent',2,block,1))
    hand = Hand.start(Table(('seat0','seat1'),(2000,2000),button=block%2),
        hand_id=f"hu20-river/{panel['name']}/{block}/{rotation}",seed=deal)
    actions = [];coverage = Counter();began = perf_counter()
    for index in range(1000):
        check()
        if hand.finished: break
        view = hand.observe(hand.actor); logical = int(hand.actor != rotation)
        search = not logical and player is not None and view.street == Street.RIVER
        blueprint_trained = None
        if not logical:
            blueprint_trained = source.distribution(view)[2]
            menu,p,trained = player.distribution(view) if search else source.distribution(view)
            action = player.choose_action(view) if search else random.choices(menu,weights=p,k=1)[0].action
            status = 'river_search' if search else 'current' if trained else 'missing'
            coverage[status] += 1; coverage[view.street.value+':'+status] += 1
        else:
            menu = choices(view,raise_cap=None,free_fold=False);p = trained = None
            action = rival.choose_action(view)
        observed = snapshot(view,menu,p,trained,None);observed['logical_player'] = logical
        actions.append({'index':index,'seat':hand.actor,'logical_player':logical,'street':view.street.value,
            'kind':action.kind.value,'raise_to':action.raise_to,'observation':observed,
            'target_key':None,'average_mass_status':None,'river_search':search,
            'blueprint_trained':blueprint_trained})
        view.legal_actions.validate(action);hand = hand.apply(action)
    if not hand.finished: raise RuntimeError('HU20 comparison action limit')
    hand_id = hand.events[0].hand_id
    replay = Hand.start(hand.table,hand_id=hand_id,seed=deal)
    for a in actions:
        if replay.actor != a['seat']:raise ValueError('Replay actor differs')
        replay = replay.apply(Action(ActionKind(a['kind']),a['raise_to']))
    chips = [p.stack-2000 for p in hand.observe(0).players]
    if not replay.finished or sum(chips) or [p.stack-2000 for p in replay.observe(0).players] != chips:
        raise ValueError('Native comparison settlement differs')
    row = {'status':'complete','policy':spec['name'],'seed':spec['seed'],'strategy':strategy,
        'panel':panel['name'],'block':block,'rotation':rotation,'button':block%2,'players':2,
        'deal_seed':deal,'root_seed':root,'hand_id':hand_id,'actions':actions,
        'target_chips':chips[rotation],'net_chips_by_seat':chips,'coverage':dict(coverage),
        'public_events_sha256':digest(public_events(hand.events)),'native_replay_verified':True,
        'search_records':player.records if player else [],'seconds':perf_counter()-began}
    if digest(public_events(replay.events)) != row['public_events_sha256']:raise ValueError('Replay digest differs')
    row['tails'] = hand_tails(row)
    return row


def comparison(plan,inputs,out,check,deadline):
    rows = []; loads = []; cache_counts = {}
    config = HU20RiverConfig(**plan['river_config'])
    for spec in plan['models']:
        check();began = perf_counter();source = load(spec,inputs)
        loads.append({'model':spec,'seconds':perf_counter()-began,'description':source.description})
        cache = RiverProfileCache(source,config.cache_entries)
        with gzip.open(out/(spec['name']+'.hands.jsonl.gz'),'wt') as stream:
            for panel in plan['panels']:
                if panel['rule'] == 'lbr':raise ValueError('LBR is excluded from this experiment')
                for strategy in ('current','average'):
                    for block in range(panel['blocks']):
                        for rotation in (0,1):
                            row = play(source,spec,panel,plan['root'],block,rotation,strategy,config,cache,check,deadline)
                            rows.append(row);stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush()
                    print(json.dumps({'model':spec['name'],'panel':panel['name'],'strategy':strategy,
                        'hands':len(rows),'peak_rss_bytes':peak_rss(),'cache':dict(cache.stats)}),flush=True)
        cache_counts[spec['name']] = dict(cache.stats)
        del source,cache;gc.collect()
    return {'hands':len(rows),'loaded':loads,'comparison':summarize(rows),'cache':cache_counts,
            'strategy_labels':{'current':'B500M direct current','average':'B500M current + river average re-solving'}}


def run(plan,inputs,out):
    out.mkdir(parents=True,exist_ok=False);began = perf_counter();failure = None;extra = {}
    source_sha = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    deadline = monotonic()+plan['max_seconds']
    def check():
        if monotonic() >= deadline:raise TimeoutError('Frozen M1 experiment deadline')
        if peak_rss() >= plan['rss_limit_bytes']:raise MemoryError('Frozen M1 RSS guard')
    try:
        if plan['phase'] == 'curve':
            source = load(plan['models'][0],inputs)
            extra = curve(source,{**plan,'_deadline':deadline},out,check)
        else: extra = comparison(plan,inputs,out,check,deadline)
    except Exception as exc:failure = f'{type(exc).__name__}: {exc}'
    result = {'status':'failed' if failure else 'complete','failure':failure,'source':source_sha,
        'plan':plan,'plan_sha256':digest(plan),'seconds':perf_counter()-began,
        'peak_rss_bytes':peak_rss(),**extra}
    (out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'manifest.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)}
        for p in out.iterdir()},indent=2,sort_keys=True)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('plan','inputs','out'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();result=run(json.loads(a.plan.read_text()),a.inputs,a.out)
    print(json.dumps({k:result[k] for k in ('status','failure','seconds','peak_rss_bytes')}))
    if result['status'] != 'complete':raise SystemExit(1)


if __name__ == '__main__': main()
