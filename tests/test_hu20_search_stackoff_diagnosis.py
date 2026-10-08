"""Paired attribution must conserve outcomes and keep common states comparable."""

from copy import deepcopy

import pytest

from scripts.diagnose_hu20_search_stackoff import first_difference, interval, summarize


def action(kind, *, logical=0, context='same'):
    return {'seat':logical,'logical_player':logical,'kind':kind,'raise_to':None,
            'observation':{'public_context':context}}


def test_first_difference_includes_shared_decision_and_rejects_opponent_drift():
    b={'actions':[action('check'),action('fold')],'target_chips':0}
    s={'actions':[action('check'),action('call')],'target_chips':-100}
    assert first_difference(b,s)==(b['actions'][1],s['actions'][1])
    s['actions'][1]['observation']['public_context']='different'
    with pytest.raises(ValueError,match='public state'): first_difference(b,s)
    b['actions'][1]=action('fold',logical=1)
    s['actions'][1]=action('call',logical=1)
    with pytest.raises(ValueError,match='Opponent differs'): first_difference(b,s)


def test_identical_actions_require_identical_settlement():
    b={'actions':[action('fold')],'target_chips':-50}
    assert first_difference(b,deepcopy(b)) is None
    s=deepcopy(b); s['target_chips']=50
    with pytest.raises(ValueError,match='settlement'): first_difference(b,s)


def test_family_interval_uses_blocks_and_widens_for_thirteen_panels():
    x=[-20,0,10,-5,5,-10]
    ordinary=interval(x); adjusted=interval(x,13)
    assert ordinary['n']==6
    assert ordinary['mean']==adjusted['mean']
    assert adjusted['interval'][0]<ordinary['interval'][0]
    assert adjusted['interval'][1]>ordinary['interval'][1]


def test_incomplete_or_duplicate_coordinates_are_not_silently_dropped():
    row={'seed':2026093001,'block':0,'rotation':0,'arm':'base'}
    with pytest.raises(ValueError,match='Incomplete'): summarize([row])
    with pytest.raises(ValueError,match='Duplicate'): summarize([row,row])


def test_disjoint_hand_accounting_and_common_spot_denominators():
    rows=[]
    for seed in (2026093001,2026093002,2026093003):
        for block in range(256):
            for rotation in (0,1):
                for arm in ('base','search'):
                    changed=block==0 and rotation==0
                    kind='call' if changed and arm=='search' else 'fold'
                    a=action(kind); a.update(street='turn',index=0)
                    a['observation'].update(call_amount=800,pot=2000,
                        menu=[{'kind':'fold','raise_to':None},{'kind':'call','raise_to':None}],
                        probabilities=[.5,.5])
                    rows.append({'seed':seed,'block':block,'rotation':rotation,'arm':arm,
                                 'deal_seed':block,'button':block%2,'actions':[a],
                                 'target_chips':-100 if changed and arm=='search' else 0})
    summary, contrasts=summarize(rows)
    assert summary['aggregate']['n']==256
    assert summary['aggregate']['mean']==pytest.approx(-50/256)
    assert len(contrasts)==3
    for table in summary['partitions'].values():
        assert sum(r['hands'] for r in table)==1536
        assert sum(r['delta_bb'] for r in table)==-3
        assert sum(r['contribution_bb_per_100'] for r in table)==pytest.approx(-300/1536)
    denominators=[r['value'] for r in summary['common_prefix_actual'] if r['key'][-1]=='opportunities']
    assert denominators==[1536,1536]
