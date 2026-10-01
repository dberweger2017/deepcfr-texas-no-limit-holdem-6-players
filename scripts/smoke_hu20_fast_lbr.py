"""Bounded M1 integration comparison, not an evaluation of playing strength."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from random import Random
import resource
import subprocess
from time import perf_counter

from src.arena.catalog import Checkpoint
from src.arena.runner import public_events
from src.arena.schedule import digest, stream_seed
from src.blueprint.artifact import FrozenBlueprint, HU20_UNCAPPED_FORMAT
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.diagnostics.cached_lbr import SharedProbabilityCache
from src.diagnostics.exact_ranker import RankedCachedLocalBestResponse, exact_seven_card
from src.diagnostics.robustness import LBRConfig, LocalBestResponse
from src.diagnostics.saved_hu20 import file_hash
from src.game.hand import Hand, Table
from src.game.showdown import hand_value
from src.game.types import Action, ActionKind

ROOT = 202610030201
POLICY_SHA = '4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf'
CONFIG = LBRConfig(4, 5)
TIMINGS = {'seconds','preparation_seconds','over_soft_budget'}


def run_hand(source, block, rotation, fast, *, root=ROOT, config=CONFIG):
    hand_value.cache_clear(); exact_seven_card.cache_clear()
    deal_seed=stream_seed(root,'test','deal',2,block)
    target_random=Random(stream_seed(root,'test','action','target',block,rotation))
    lbr_seed=stream_seed(root,'test','opponent','lbr',block,rotation)
    cache=SharedProbabilityCache(source)
    lbr=(RankedCachedLocalBestResponse(source,lbr_seed,cache,config) if fast
         else LocalBestResponse(source,lbr_seed,config))
    hand_id=f'lbr-smoke-{block}-{rotation}'
    table=Table(('seat0','seat1'),(2000,2000),button=block%2)
    hand=Hand.start(table,hand_id=hand_id,seed=deal_seed)
    actions=[]; start=perf_counter(); lookups=Counter()
    for _ in range(1000):
        if hand.finished:break
        seat=hand.actor;view=hand.observe(seat)
        entry={'seat':seat,'street':view.street.value}
        if seat==rotation:
            menu,probabilities,trained=source.distribution(view)
            lookups['trained' if trained else 'fallback']+=1
            action=target_random.choices(menu,weights=probabilities,k=1)[0].action
        else:
            action=lbr.choose_action(view)
            entry['lbr']={k:v for k,v in lbr.telemetry[-1].items() if k not in TIMINGS}
            entry['lbr_rng_sha256']=digest(lbr.random.getstate())
            entry['posterior_sha256']=digest({'holdings':lbr.holdings,'weights':lbr.weights.tolist(),
                                             'processed':lbr.processed,'zero':lbr.zero_likelihood})
        view.legal_actions.validate(action)
        entry.update(kind=action.kind.value,raise_to=action.raise_to)
        actions.append(entry);hand=hand.apply(action)
    if not hand.finished:raise RuntimeError('Hand exceeded decision bound')
    seconds=perf_counter()-start
    final=hand.observe(0);chips=[p.stack-2000 for p in final.players]
    replay=Hand.start(table,hand_id=hand_id,seed=deal_seed)
    for entry in actions:
        if replay.actor!=entry['seat']:raise ValueError('Replay actor differs')
        replay=replay.apply(Action(ActionKind(entry['kind']),entry['raise_to']))
    events=digest(public_events(hand.events))
    if not replay.finished or digest(public_events(replay.events))!=events or sum(chips)!=0:
        raise ValueError('Native replay/settlement differs')
    return {'block':block,'rotation':rotation,'button':block%2,'deal_seed':deal_seed,
            'actions':actions,'net_chips':chips,'public_events_sha256':events,'native_replay_verified':True,
            'lbr_rng_state':lbr.random.getstate(),'target_rng_state':target_random.getstate(),
            'target_lookups':dict(lookups),'seconds':seconds,'lbr_telemetry':lbr.telemetry,
            'shared_cache':cache.telemetry() if fast else None}


def compare(native,fast):
    fields=('block','rotation','button','deal_seed','actions','net_chips','public_events_sha256',
            'native_replay_verified','lbr_rng_state','target_rng_state','target_lookups')
    differences=[k for k in fields if native[k]!=fast[k]]
    complete=all(t['completed'] for row in (native,fast) for t in row['lbr_telemetry'])
    return {'identical':not differences,'different_fields':differences,'all_batches_complete':complete}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if file_hash(a.policy)!=POLICY_SHA:raise ValueError('Pinned policy hash mismatch before loading')
    load=perf_counter();source=FrozenBlueprint(Checkpoint('B100M-v04',str(a.policy),POLICY_SHA,HU20_UNCAPPED_FORMAT),a.policy)
    load=perf_counter()-load
    if (source.players!=2 or source.raise_cap is not None or source.abstraction!=HU20_UNCAPPED_SCHEMA
        or source.description['strategy']!='current' or source.description['training_seed']!=2026093001):
        raise ValueError('Pinned policy identity differs')
    result={'version':'hu20-fast-lbr-integration-v1','source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'model':source.description,'game':source.identity,'root':ROOT,'config':asdict(CONFIG),'blocks':8,
            'load_seconds':load,'status':'running','comparisons':[]}
    with (a.out/'hands.jsonl').open('w') as records:
        for block in range(8):
            for rotation in (0,1):
                rows=[run_hand(source,block,rotation,fast) for fast in (False,True)]
                for fast,row in zip((False,True),rows):
                    records.write(json.dumps({'executor':'fast' if fast else 'native',**row},sort_keys=True)+'\n');records.flush()
                check=compare(*rows)
                check.update(block=block,rotation=rotation,native_seconds=rows[0]['seconds'],fast_seconds=rows[1]['seconds'])
                result['comparisons'].append(check)
                (a.out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
                print(json.dumps(check),flush=True)
                if not check['identical'] or not check['all_batches_complete']:
                    result['status']='failed';break
            if result['status']=='failed':break
    if result['status']=='running':result['status']='complete'
    result['native_seconds']=sum(c['native_seconds'] for c in result['comparisons'])
    result['fast_seconds']=sum(c['fast_seconds'] for c in result['comparisons'])
    result['speedup']=result['native_seconds']/result['fast_seconds']
    result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (a.out/'manifest.json').write_text(json.dumps({f.name:{'bytes':f.stat().st_size,'sha256':file_hash(f)} for f in a.out.iterdir()},indent=2,sort_keys=True)+'\n')
    if result['status']!='complete':raise SystemExit('Integration mismatch retained; do not call this a passing smoke')


if __name__=='__main__':main()
