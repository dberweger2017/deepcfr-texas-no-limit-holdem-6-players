"""Synthetic policies test the evaluator without consuming blend calibration hands."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.calibrate_hu200_river_search import schedule
from scripts.quote_hu200_river_strength import rounded_count, variance_summary
from src.arena.policies import CheckCall, make_policy
from src.arena.river_search_calibration import ForbiddenSolver, block_value, make_players, play_hand, view_key
from src.blueprint.abstraction import Choice, HU200_EQUITY_SCHEMA
from src.blueprint.hu20_turn_solver import SolveFailure
from src.blueprint.hu20_turn_solver import PolicyMatrix
from src.blueprint.river_ledger import DecisionLedger, encode_matrix
from src.game.observation import Observation
from src.game.types import Street


class ToyBlueprint:
    abstraction = HU200_EQUITY_SCHEMA
    description = {"fixture": "calibration-wiring"}
    def distribution(self, view):
        assert isinstance(view, Observation)
        return (Choice('pass',CheckCall().choose_action(view)),),(1.,),True

    def policy(self, seed):
        return CheckCall()


def test_real_search_deadline_wiring_preserves_static_check(tmp_path):
    from scripts.pilot_hu200_river_search import CONFIG
    calls=[]
    (tmp_path/'ledger').mkdir()
    _,policy,solver=make_players(ToyBlueprint(),None,'fixture',CONFIG,tmp_path,
        [1,2],None,True,True,lambda:calls.append(True))
    policy._check(None)
    assert calls==[True]
    with pytest.raises(SolveFailure,match='deadline'):
        policy._check(0)
    assert calls==[True,True] and not solver.records


def toy_search(base,binary,identity,config,directory,seeds,opponent,searched,reproduce,check,
               *, corrupt=False):
    policy=SimpleNamespace(played={})
    ledger=DecisionLedger(directory/'ledger',replay=reproduce)
    if corrupt and reproduce:
        path=next((directory/'ledger').glob('*.json'))
        value=json.loads(path.read_text());value['matrix']['probabilities'][0][0]=.9
        path.write_text(json.dumps(value))
    class Player:
        def choose_action(self,view):
            assert isinstance(view,Observation)
            menu,_,_=base.distribution(view);action=menu[0].action
            if view.street==Street.RIVER:
                key=view_key(view)
                matrix=PolicyMatrix(menu,(tuple(sorted(view.hole_cards)),),np.ones((1,1)))
                if reproduce:ledger.read(key)
                else:ledger.write(key,dict(matrix=encode_matrix(matrix)))
                policy.played[view.seat,view.history]=(matrix,)
                self.last_decision=dict(public_history=key,fallback=False,fallback_cause=None,
                    changed_action=False,model_ev_chips=0.,model_ev_delta_chips=0.,
                    model_ev_unavailable=None,search_action=action,blueprint_action=action)
            else:self.last_decision=None
            return action
    rival=make_policy(opponent,seeds[1]) if opponent else base.policy(seeds[1])
    return (Player(),rival),policy,ForbiddenSolver(identity)


def test_complete_hand_replays_and_reproduces_policy_ledgers_without_native(tmp_path):
    result=play_hand(ToyBlueprint(),None,'fixture',None,tmp_path/'search',
        deal=31,seeds=[1,2],button=0,rotation=0,hand_id='opaque',
        opponent='check_call',player_builder=toy_search)
    assert result['verified'] and len(result['search_rows'])==1
    check=json.loads((tmp_path/'search/verified.json').read_text())
    assert check['native_replay_solves']==0
    assert check['independent_replay'] and check['fresh_player_reproduction']
    plain=play_hand(ToyBlueprint(),None,'fixture',None,tmp_path/'plain',
        deal=31,seeds=[1,2],button=0,rotation=0,hand_id='opaque',
        opponent='check_call',searched=False)
    assert plain['target_chips']==result['target_chips']
    first=json.loads((tmp_path/'search/hand.json').read_text())
    other=json.loads((tmp_path/'plain/hand.json').read_text())
    assert first['public_events']==other['public_events']


def test_corrupt_policy_ledger_stops_before_a_verified_result(tmp_path):
    def corrupted(*args,**kwargs):return toy_search(*args,**kwargs,corrupt=True)
    with pytest.raises(ValueError,match='checksum'):
        play_hand(ToyBlueprint(),None,'fixture',None,tmp_path/'search',
            deal=31,seeds=[1,2],button=0,rotation=0,hand_id='opaque',
            opponent='check_call',player_builder=corrupted)
    assert (tmp_path/'search/incomplete.json').exists()
    assert not (tmp_path/'search/verified.json').exists()


def test_block_units_and_disjoint_declared_schedule():
    assert block_value([100,300])==200
    with pytest.raises(ValueError):block_value([100])
    first=schedule('direct',20261017317100)
    second=schedule('pot_pressure',20261017317105)
    assert len(first)==128 and len(second)==32
    assert {b.deal_seeds[0] for b in first}.isdisjoint(b.deal_seeds[0] for b in second)


def test_variance_readout_has_uncertainty_and_no_gain_mean():
    summary=variance_summary([-100,0,100,200])
    assert summary['sd_normal_model_ci95'][0]<summary['sd_bb100']<summary['sd_normal_model_ci95'][1]
    assert summary['sd_normal_model_upper95']>summary['sd_bb100']
    assert not any('mean' in key for key in summary)
    assert rounded_count(0)==4096
    assert rounded_count(879.3558501872744)==65536


def test_final_verification_reuses_complete_hand_and_rejects_changed_evidence(tmp_path):
    from scripts.evaluate_hu200_river_final import verify_hand, recovered_result
    from scripts.evaluate_hu100_direct import action_seed
    item=SimpleNamespace(scenario='fixture-final',index=0,button=0,deal_seeds=(31,))
    root=4127;seeds=[action_seed(root,item,0,r) for r in (0,1)]
    directory=tmp_path/'closed'
    play_hand(ToyBlueprint(),None,'fixture',None,directory,deal=31,seeds=seeds,
        button=0,rotation=0,hand_id='table-0/0/0/0',opponent='check_call',player_builder=toy_search)
    hand=verify_hand(directory,item,root,0,'search')
    recovered=recovered_result(directory,hand)
    assert recovered['recovered_without_native'] and recovered['target_chips']==hand['target_chips']
    assert recovered['primary_seconds'] is None
    path=next((directory/'decisions').glob('*.json'));path.write_text('{}')
    with pytest.raises(ValueError,match='checksum'):verify_hand(directory,item,root,0,'search')
