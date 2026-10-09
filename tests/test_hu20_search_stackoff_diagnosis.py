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


def test_hydration_timeout_restarts_exact_unhashed_offset(monkeypatch):
    import hashlib
    import io
    from pathlib import Path
    from scripts import diagnose_hu20_search_stackoff as module
    data=b'first'+b'x'*1024**2+b'last'
    class HydratingFile(io.BytesIO):
        calls=0
        def read(self,n):
            self.calls+=1
            if self.calls==2:
                super().read(17)
                raise TimeoutError('hydrating')
            return super().read(n)
    monkeypatch.setattr(Path,'open',lambda *args,**kwargs:HydratingFile(data))
    monkeypatch.setattr(module,'sleep',lambda _:None)
    assert module.file_hash(Path('unused'))==hashlib.sha256(data).hexdigest()


def test_scientific_request_preserves_science_and_removes_host_admission():
    from scripts.resolve_hu20_search_stackoff import scientific_request
    r={'ranges':[[1],[2]],'threads':6,'max_iterations':50,'memory_budget_bytes':123,
       'requested_memory_budget_bytes':456,'mode':'play','seconds':None,'dump_path':'host/path'}
    assert scientific_request(r)=={'ranges':[[1],[2]],'threads':6,'max_iterations':50,'memory_budget_bytes':456}


def test_exact_frozen_rule_likelihood_integrates_trapping_and_overfolds():
    from dataclasses import replace
    from src.game.hand import Hand, Table
    from src.blueprint.abstraction import choices
    from scripts.resolve_hu20_search_stackoff import frozen_likelihood
    view=Hand.start(Table(('a','b'),(2000,2000)),seed=123,hand_id='fixture').observe(0)
    strong=replace(view,hole_cards=('Ac','Ad'))
    menu={c.name:c.action for c in choices(strong,raise_cap=None,free_fold=False)}
    assert frozen_likelihood(strong,menu['min'])==.35
    assert frozen_likelihood(strong,menu['call'])==.65
    assert frozen_likelihood(strong,menu['fold'])==0
    weak=replace(view,hole_cards=('4c','2d'))
    assert frozen_likelihood(weak,menu['fold'])==1
    medium=replace(view,hole_cards=('Ac','2d'))
    assert frozen_likelihood(medium,menu['call'])==1


def test_frozen_rule_large_call_is_a_price_threshold_not_a_bet_size():
    from dataclasses import replace
    from src.game.hand import Hand, Table
    from src.game.types import Action, ActionKind, Street
    from scripts.resolve_hu20_search_stackoff import frozen_likelihood
    view=Hand.start(Table(('a','b'),(2000,2000)),seed=123,hand_id='fixture').observe(0)
    # Isolate the price branch with concrete cards, without recorded inputs.
    view=replace(view,street=Street.RIVER,board=('Js','2h','6d','5d','9h'),
                 hole_cards=('5c','5s'),
                 legal_actions=replace(view.legal_actions,call_amount=800))
    assert frozen_likelihood(view,Action(ActionKind.CALL))==1
    assert frozen_likelihood(view,Action(ActionKind.FOLD))==0
    medium=replace(view,hole_cards=('Jc','Tc'))
    assert frozen_likelihood(medium,Action(ActionKind.FOLD))==1
    assert frozen_likelihood(medium,Action(ActionKind.CALL))==0


def test_guarded_stages_are_sequential_and_only_replay_or_resolve_recorded_inputs(tmp_path):
    from scripts.run_hu20_search_stackoff_diagnosis import jobs
    for stage in ('restore','analyze','retrieve','resolve','tests'):
        work=jobs(stage,tmp_path,tmp_path/'original.zip',tmp_path/'frozen-tool')
        assert work
        assert all('arena' not in ' '.join(j['command']) for j in work)
    assert [j['name'] for j in jobs('resolve',tmp_path,tmp_path/'original.zip',tmp_path/'tool')]==['resolve','responses']
    with pytest.raises(ValueError,match='Unknown'): jobs('play',tmp_path,tmp_path/'zip',tmp_path/'tool')
