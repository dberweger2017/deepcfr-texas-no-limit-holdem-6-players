"""Outcome-blind replay-only cost; no profit series, mean, interval or label is produced."""
import gzip
import json
from pathlib import Path
import resource
from time import perf_counter
from src.arena.runner import public_events
from src.game.hand import Hand,Table
from src.game.types import Action,ActionKind
from scripts.run_hu20_zero_mass_fallback import ROOT,FAMILIES,write,digest,guard

def measure():
    guard();observations=[]
    for l in (1,2,3):
        path=ROOT/'pilot'/FAMILIES[0]/'run'/f'direct-lineage-{l}.hands.jsonl.gz'
        started=perf_counter();hands=actions=0
        with gzip.open(path,'rt') as f:
            for row in map(json.loads,f):
                hand=Hand.start(Table(('seat0','seat1'),(2000,2000),button=row['button']),hand_id=row['hand_id'],seed=row['deal_seed'])
                for index,a in enumerate(row['actions']):
                    assert hand.actor==a['seat'] and index==a['index'];view=hand.observe(hand.actor)
                    assert a['street']==view.street.value and a['observation']['hole_cards']==list(view.hole_cards)
                    assert a['observation']['board']==list(view.board) and a['observation']['pot']==view.pot
                    action=Action(ActionKind(a['kind']),a['raise_to']);view.legal_actions.validate(action);hand=hand.apply(action);actions+=1
                assert hand.finished
                chips=[p.stack-2000 for p in hand.observe(0).players]
                assert chips==row['net_chips_by_seat'] and sum(chips)==0 and chips[row['rotation']]==row['target_chips']
                assert digest(public_events(hand.events))==row['public_events_sha256'];hands+=1
        observations.append({'lineage':l,'hands':hands,'actions':actions,'seconds':perf_counter()-started})
    result={'measurements':observations,'max_seconds_per_hand':max(o['seconds']/o['hands'] for o in observations),
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'profit_series_or_statistics_produced':False,'scope':'replay-only timing, all full pilot arithmetic still deferred until final freezing'}
    write(ROOT/'replay-cost-only.json.gz',result);print(json.dumps(result,indent=2));guard()

if __name__=='__main__':measure()
