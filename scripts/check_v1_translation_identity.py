"""Fixed translation-on HU100/HU200 match streams for cross-revision identity."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from src.blueprint.abstraction import HU100_SCHEMA, HU200_SCHEMA, choices, information_key
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy
from src.blueprint.solver import HU100_GAME, HU200_GAME
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken
from src.game.types import Action, ActionKind
from src.arena.schedule import canonical
from src.arena.runner import public_events


def stream(stack, out):
    schema,game=(HU100_SCHEMA,HU100_GAME) if stack==10000 else (HU200_SCHEMA,HU200_GAME)
    m=AveragePolicy.__new__(AveragePolicy)
    m.players=2;m.raise_cap=None;m.abstraction=schema;m.game=game;m.entries={};m.zero_mass=set();m.description={}
    m.configure_missing_key_fallback('uniform');m.configure_translation(TranslationOptions())
    # Freeze a card-v1 witness table before any match, all rows deterministically call.
    for i in range(64):
        initial=Hand.start(Table(('a','b'),(stack,stack),button=i%2),hand_id=f'identity-{i}',seed=202610142000+i)
        v=initial.apply(Action(ActionKind.RAISE,550)).observe(1-i%2)
        menu=choices(v,raise_cap=None,free_fold=False)
        index=next(j for j,e in enumerate(v.history) if isinstance(e,ActionTaken))
        key=information_key(v,menu,schema=schema,history_label_overrides={index:'raise-2'})
        m.entries[key]=(tuple(c.name for c in menu),tuple(float(c.name=='call') for c in menu))
    table_hash=hashlib.sha256(canonical(m.entries).encode()).hexdigest()
    out.mkdir(parents=True,exist_ok=False);translated=decisions=0
    with (out/'trace.jsonl').open('x') as trace:
        for i in range(64):
            h=Hand.start(Table(('a','b'),(stack,stack),button=i%2),hand_id=f'identity-{i}',seed=202610142000+i)
            player=m.policy(202610143000+i);own=1-i%2;first=True
            while not h.finished:
                v=h.observe(h.actor)
                if h.actor==own:
                    action=player.choose_action(v);menu,p,known,info=player.last_decision
                    info={k:x for k,x in info.items() if k!='lookup_seconds'}
                    trace.write(canonical(dict(hand=i,seat=own,street=v.street.value,action=asdict(action),
                        menu=[asdict(c) for c in menu],probabilities=p,known=known,translation=info))+'\n')
                    decisions+=1;translated+=int(info['mode']=='translated')
                else:
                    action=Action(ActionKind.RAISE,550) if first else Action(ActionKind.CHECK if ActionKind.CHECK in v.legal_actions.kinds else ActionKind.CALL)
                first=False;h=h.apply(action)
            trace.write(canonical(dict(hand=i,events=public_events(h.events)))+'\n')
    return dict(hands=64,decisions=decisions,translated=translated,table_sha256=table_hash,
        trace_sha256=hashlib.sha256((out/'trace.jsonl').read_bytes()).hexdigest(),translation=m.description['action_translation'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result={str(stack//100):stream(stack,a.out/str(stack//100)) for stack in (10000,20000)}
    (a.out/'identity.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
