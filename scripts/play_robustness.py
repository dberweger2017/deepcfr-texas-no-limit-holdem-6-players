"""Run one diagnostic hand or replay its recorded concrete actions."""
import argparse,json
from pathlib import Path
from src.arena.schedule import digest
from src.game.hand import Hand,Table
from src.game.types import Action,ActionKind
from scripts.evaluate_robustness import load,play
from src.diagnostics.robustness import LBRConfig


def replay_row(row):
    n=row['players'];r=row['rotation'];ids=tuple(f'player-{(s-r)%n}' for s in range(n))
    hand=Hand.start(Table(ids,(2000,)*n,button=row['button']),
        hand_id=f"robustness-{row['phase']}-{n}-{row['block']}",seed=row['deal_seed'])
    for item in row['actions']:
        if hand.actor!=item['seat']:raise ValueError('Actor differs on replay')
        hand=hand.apply(Action(ActionKind(item['kind']),item['raise_to']))
    if digest([repr(e) for e in hand.events])!=row['event_digest']:raise ValueError('Events differ on replay')
    if row['status']=='complete':
        net=tuple(v-2000 for v in hand.events[-1].stacks)
        if sum(net)!=0 or not hand.finished:raise ValueError('Invalid settlement')
        if row['net_chips_by_seat'] is not None and list(net)!=list(row['net_chips_by_seat']):raise ValueError('Payoff differs')
    return hand


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path);p.add_argument('--policy');p.add_argument('--players',type=int,default=2)
    p.add_argument('--rule',choices=('pressure','minraise','passive','lbr'),default='pressure');p.add_argument('--contract',choices=('menu','native'),default='menu')
    p.add_argument('--block',type=int,default=0);p.add_argument('--root-seed',type=int,default=2026135001)
    p.add_argument('--history',type=Path);p.add_argument('--replay',type=Path);a=p.parse_args()
    if a.replay:
        row=json.loads(a.replay.read_text());replay_row(row);print('Replay verified');return
    if a.history is None or a.history.exists():raise ValueError('Supply a new history path')
    plan=json.loads(a.plan.read_text());spec=next(s for s in plan['policies'] if s['name']==a.policy and s['players']==a.players)
    rows=[];play(load(spec),spec,(a.rule,)*(a.players-1),a.contract,a.block,0,a.root_seed,'demo',
        LBRConfig(plan['chance_samples'],plan['lbr_seconds']),rows.append)
    a.history.write_text(json.dumps(rows[0],indent=2)+'\n');replay_row(rows[0]);print(json.dumps({'status':rows[0]['status'],'target_chips':rows[0]['target_chips']}))

if __name__=='__main__':main()
