"""Independent small equity, collision, aggregation and deterministic samples."""
from itertools import combinations

import numpy as np
import pytest

from src.blueprint.abstraction import _postflop
from src.blueprint.search import DECK
from src.diagnostics.bucket_coarseness import dashboard, equity, sample_rows, summarize
from src.game.showdown import hand_value


def test_exact_river_tie_and_independent_kicker_reference():
    assert equity(('2d','3h'),('Ac','Kc','Qc','Jc','Tc'),91)=={
        'equity':.5,'worlds':990,'method':'exact','max_standard_error':0}
    board=('Ad','7d','7c','Ah','3s');cards=('Qc','Kc');available=[c for c in DECK if c not in cards+board]
    own=hand_value(cards+board);scores=[]
    for pair in combinations(available,2):
        rival=hand_value(pair+board);scores.append(1 if own>rival else .5 if own==rival else 0)
    assert equity(cards,board,91)['equity']==sum(scores)/990
    weak=('2c','2d')
    assert _postflop(weak,board)==_postflop(cards,board)
    assert hand_value(weak+board)!=own
    assert equity(weak,board,91)['equity']<equity(cards,board,91)['equity']


def test_mc_uses_same_worlds_with_independent_reference():
    cards=('As','2c');board=('Ac','Kd','7h')
    assert equity(cards,board,93,samples=16)==equity(cards,board,93,samples=16,ranker=hand_value)
    assert equity(cards,board+('3s',),93,samples=16)==equity(cards,board+('3s',),93,samples=16,ranker=hand_value)


def test_deterministic_reproduction_and_card_removal():
    rows=list(sample_rows(root=91,boards=2,holdings=2,samples=4))
    assert rows==list(sample_rows(root=91,boards=2,holdings=2,samples=4))
    assert len(rows)==12
    for row in rows:
        assert len(set(row['board']+row['cards']))==len(row['board'])+2
        assert 0<=row['equity']<=1
        assert row['bucket']==_postflop(row['cards'],row['board'])
    result=summarize(rows)
    assert result['street_totals']=={'flop':4,'turn':4,'river':4}
    for street in result['street_totals']:
        assert sum(b['share'] for b in result['buckets'] if b['street']==street)==pytest.approx(1)
    assert 'Every observed bucket' in dashboard(result)


def test_quantiles_and_same_board_only_collisions():
    def row(i,board,value):
        return {'street':'river','bucket':(1,2,0,0,1),'equity':value,'board_index':board,
                'holding_index':i,'board':('Ac','Ad','7c','3d','2s'),'cards':('Kd','Qd'),
                'made_value':(1,14,13-i,7,3),'plays_board':False,'equity_seed':i}
    result=summarize([row(0,0,.1),row(1,0,.4),row(2,1,.95)])
    b=result['buckets'][0]
    assert [b[k] for k in ('p10','median','p90')]==pytest.approx(np.quantile([.1,.4,.95],[.1,.5,.9]))
    assert b['n']==3 and b['share']==1 and b['boards']==2
    c=b['largest_same_board_collision']
    assert c['board_index']==0 and c['equity_spread']==pytest.approx(.3)
    assert c['different_made_value']


def test_invalid_cards_and_work_limits():
    with pytest.raises(ValueError):equity(('Ac','Ac'),('2d','3h','4s'),0)
    with pytest.raises(ValueError):list(sample_rows(boards=65))
