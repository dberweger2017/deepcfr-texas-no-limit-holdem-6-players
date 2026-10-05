"""Play and replay HU20 using the verified candidate's native-reopening menu."""

import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random

from scripts.play_hu20 import _display, _label, _table, replay_history
from src.arena.catalog import Checkpoint
from src.arena.runner import public_events
from src.arena.schedule import digest
from src.blueprint.abstraction import choices
from src.blueprint.artifact import FrozenBlueprint, HU20_UNCAPPED_FORMAT
from src.blueprint.solver import HU20_UNCAPPED_GAME
from src.game.hand import Hand


def play(policy, sha256, history, *, seed=2026146001, max_hands=None, input_fn=input, output=print,
         source=None):
    source=source or FrozenBlueprint(Checkpoint('native-reopening-demo',str(policy),sha256,HU20_UNCAPPED_FORMAT),policy)
    base=getattr(source,'blueprint',source)
    if base.game!=HU20_UNCAPPED_GAME or base.description['iteration']<1:
        raise ValueError('Use a verified trained native-reopening HU20 artifact')
    if history.exists():raise FileExistsError('Choose a new history path')
    history.parent.mkdir(parents=True,exist_ok=True)
    deals=Random(seed);bot=Random(seed^0xB071);coverage=Counter();total=0;hands=0
    output(f"HU20 native-reopening | game: {base.game} | model: {sha256}")
    output('Abstract min/pot/conditional-jam sizes, no artificial raise-count cap.')
    with history.open('w') as saved:
        while max_hands is None or hands<max_hands:
            button=hands%2;deal=deals.randrange(2**63);hand_id=f'hu20-native-human-{seed}-{hands}'
            hand=Hand.start(_table(button),hand_id=hand_id,seed=deal);actions=[]
            while not hand.finished:
                view=hand.observe(hand.actor)
                if hand.actor==0:
                    _display(view,output)
                    menu=choices(view,raise_cap=source.raise_cap,free_fold=False)
                    for i,c in enumerate(menu,1):output(f'  {i}. {_label(c.action)}')
                    while True:
                        answer=input_fn('Choose action number: ').strip()
                        if answer.isdigit() and 1<=int(answer)<=len(menu):break
                        output('Choose a listed action number.')
                    action=menu[int(answer)-1].action
                else:
                    from src.blueprint.hu20_turn_search import HU20TurnSearchPolicy
                    menu,p,trained=(source.distribution(view,query_kind="play")
                        if isinstance(source,HU20TurnSearchPolicy) else source.distribution(view))
                    coverage['trained' if trained else 'fallback']+=1
                    action=bot.choices(menu,weights=p,k=1)[0].action;output(f'Bot: {_label(action)}')
                view.legal_actions.validate(action)
                actions.append({'seat':hand.actor,'kind':action.kind.value,'raise_to':action.raise_to})
                hand=hand.apply(action)
            final=hand.observe(0);net=final.players[0].stack-2000
            if final.players[1].stack-2000!=-net:raise ValueError('Chip accounting')
            total+=net;output(f"Board: {' '.join(final.board)}")
            for s,p in enumerate(final.players):
                if p.shown_cards:output(f"{('You','Bot')[s]} showed: {' '.join(p.shown_cards)}")
            output(f'Hand: {net/100:+g} BB | session: {total/100:+g} BB')
            saved.write(json.dumps({'schema':'human-hu20-native-reopening-history-v1',
                'game':base.game,'model_sha256':sha256,'policy_description':source.description,
                'hand_id':hand_id,'button':button,
                'deal_seed':deal,'actions':actions,'human_chips':net,
                'public_events_sha256':digest(public_events(hand.events))},sort_keys=True)+'\n');saved.flush()
            hands+=1
            if max_hands is None and input_fn('Another hand? [Y/n] ').strip().lower() in ('n','no'):break
    return {'hands':hands,'human_chips':total,'lookup_coverage':dict(coverage)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path);p.add_argument('--sha256')
    p.add_argument('--history',type=Path);p.add_argument('--seed',type=int,default=2026146001)
    p.add_argument('--hands',type=int);p.add_argument('--replay',type=Path);a=p.parse_args()
    if a.replay:print(f'Verified {replay_history(a.replay)} hands');return 0
    if a.policy is None or a.sha256 is None or a.history is None:p.error('--policy, --sha256 and new --history required')
    print(json.dumps(play(a.policy,a.sha256,a.history,seed=a.seed,max_hands=a.hands)));return 0

if __name__=='__main__':raise SystemExit(main())
