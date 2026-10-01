"""Uniform-card, model-free inspection of the unchanged postflop descriptors."""
from collections import defaultdict
from itertools import combinations
from math import sqrt
from random import Random

import numpy as np

from src.arena.schedule import stream_seed
from src.blueprint.abstraction import _postflop
from src.blueprint.search import DECK
from src.diagnostics.exact_ranker import exact_seven_card
from src.game.showdown import hand_value

ROOT = 202610030301
STREETS = {'flop':3, 'turn':4, 'river':5}


def equity(cards, board, seed, *, samples=512, ranker=exact_seven_card):
    """Uniform compatible pair/runout, not an inferred opponent betting range."""
    if len(cards)!=2 or len(board) not in (3,4,5) or len(set(cards+board))!=len(cards+board):
        raise ValueError('Need distinct own cards and flop/turn/river board')
    if any(c not in DECK for c in cards+board) or type(samples) is not int or samples<1:
        raise ValueError('Invalid cards or sample count')
    available=tuple(c for c in DECK if c not in cards+board)
    random=Random(seed);score=0;count=0
    if len(board)==5:
        worlds=((pair,board) for pair in combinations(available,2))
    else:
        def sampled():
            for _ in range(samples):
                drawn=random.sample(available,2+5-len(board))
                yield tuple(drawn[:2]),board+tuple(drawn[2:])
        worlds=sampled()
    for pair,runout in worlds:
        a,b=ranker(cards+runout),ranker(pair+runout)
        score += 1 if a>b else .5 if a==b else 0
        count+=1
    return {'equity':score/count,'worlds':count,'method':'exact' if len(board)==5 else 'monte_carlo',
            'max_standard_error':0 if len(board)==5 else .5/sqrt(count)}


def sample_rows(*,root=ROOT,boards=64,holdings=16,samples=512):
    if not 1<=boards<=64 or not 1<=holdings<=16 or not 1<=samples<=512:
        raise ValueError('Study exceeds the frozen work bounds')
    for street,length in STREETS.items():
        for board_index in range(boards):
            deal_seed=stream_seed(root,'test','deal',street,board_index)
            random=Random(deal_seed);board=tuple(random.sample(DECK,length))
            available=tuple(c for c in DECK if c not in board);seen=set()
            for holding_index in range(holdings):
                while True:
                    cards=tuple(sorted(random.sample(available,2)))
                    if cards not in seen:break
                seen.add(cards)
                equity_seed=stream_seed(root,'test','opponent',street,board_index,holding_index)
                value=hand_value(cards+board)
                yield {'street':street,'board_index':board_index,'holding_index':holding_index,
                       'deal_seed':deal_seed,'equity_seed':equity_seed,'board':board,'cards':cards,
                       'bucket':_postflop(cards,board),'made_value':value,
                       'plays_board':len(board)==5 and value==hand_value(board),
                       **equity(cards,board,equity_seed,samples=samples)}


def summarize(rows):
    groups=defaultdict(list);totals=defaultdict(int)
    for row in rows:
        groups[row['street'],tuple(row['bucket'])].append(row);totals[row['street']]+=1
    result=[]
    for (street,bucket),items in sorted(groups.items()):
        values=[r['equity'] for r in items]
        p10,median,p90=map(float,np.quantile(values,[.1,.5,.9],method='linear'))
        share=len(items)/totals[street]
        by_board=defaultdict(list)
        for row in items:by_board[row['board_index']].append(row)
        collisions=[]
        for board_index,cluster in sorted(by_board.items()):
            if len(cluster)<2:continue
            ordered=sorted(cluster,key=lambda r:(r['equity'],r['holding_index']))
            low,high=ordered[0],ordered[-1]
            collisions.append({'board_index':board_index,'board':low['board'],
                'equity_spread':high['equity']-low['equity'],
                'different_made_value':low['made_value']!=high['made_value'],
                'low':{k:low[k] for k in ('cards','made_value','equity','plays_board','equity_seed')},
                'high':{k:high[k] for k in ('cards','made_value','equity','plays_board','equity_seed')}})
        collision=max(collisions,key=lambda c:(c['equity_spread'],-c['board_index']),default=None)
        result.append({'street':street,'bucket':bucket,'n':len(items),'share':share,
                       'boards':len(by_board),'p10':p10,'median':median,'p90':p90,
                       'spread':p90-p10,'min':min(values),'max':max(values),
                       'common_spread_score':share*(p90-p10),'largest_same_board_collision':collision})
    return {'version':'hu20-postflop-coarseness-v1','street_totals':dict(totals),'buckets':result,
            'unobserved':'Unmeasured; not proof that a descriptor is impossible',
            'range':'uniform compatible opponent, no betting',
            'quantile_method':'linear','common_min_n':20}


def dashboard(result):
    lines=['# Current HU20 postflop bucket coarseness','',
        'Uniform-card sample; no policies loaded. Shares are within each street. Flop/turn equity is '
        'Monte Carlo against a uniform range, river exact. p90−p10 is a between-holding spread, '
        'not an uncertainty interval. Rows share board clusters.', '',
        '## Common buckets ranked by share × spread (n ≥20)','',
        '| Street | Bucket | n / street | Boards | Share | Equity p10 / median / p90 | Spread |',
        '| --- | --- | ---: | ---: | ---: | --- | ---: |']
    ranked=sorted((b for b in result['buckets'] if b['n']>=20),key=lambda b:(-b['common_spread_score'],b['street'],b['bucket']))
    for b in ranked[:15]:
        q=' / '.join(f'{b[k]:.3f}' for k in ('p10','median','p90'))
        lines.append(f"| {b['street']} | {b['bucket']} | {b['n']}/{result['street_totals'][b['street']]} | {b['boards']} | {100*b['share']:.2f}% | {q} | {b['spread']:.3f} |")
    lines += ['', '## Concrete collisions on the same board','',
              '| Street | Bucket | Board | Low holding / value / equity | High holding / value / equity | Spread |',
              '| --- | --- | --- | --- | --- | ---: |']
    collisions=sorted((b for b in result['buckets'] if b['largest_same_board_collision']),
        key=lambda b:(-b['largest_same_board_collision']['equity_spread'],b['street'],b['bucket']))
    for b in collisions[:15]:
        c=b['largest_same_board_collision'];sides=[]
        for side in ('low','high'):
            r=c[side];sides.append(f"{' '.join(r['cards'])} / {r['made_value']} / {r['equity']:.3f}")
        lines.append(f"| {b['street']} | {b['bucket']} | {' '.join(c['board'])} | {' | '.join(sides)} | {c['equity_spread']:.3f} |")
    lines += ['', '## Every observed bucket','',
              '| Street | Bucket | n | Boards | Share | p10 | Median | p90 | p90−p10 | Min / max |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for b in result['buckets']:
        lines.append(f"| {b['street']} | {b['bucket']} | {b['n']} | {b['boards']} | {100*b['share']:.2f}% | {b['p10']:.3f} | {b['median']:.3f} | {b['p90']:.3f} | {b['spread']:.3f} | {b['min']:.3f} / {b['max']:.3f} |")
    return '\n'.join(lines)+'\n'
