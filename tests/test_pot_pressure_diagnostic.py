from types import SimpleNamespace
import pytest
from scripts import pot_pressure_diagnostic as diagnostic
from src.arena.schedule import build_schedule
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.blueprint.abstraction import HU100_SCHEMA, choices, information_key


def view(check=False):
    hand=Hand.start(Table(('player-0','player-1'),(10000,10000)),hand_id='diagnostic-fixture',seed=41)
    if check: hand=hand.apply(Action(ActionKind.CALL))
    return hand.observe(hand.actor)


def probe(v,mode='uniform',reason='missing-key',fallback=None):
    menu=choices(v,raise_cap=None,free_fold=False)
    class Player:
        calls=0
        last_decision=(menu,(1/len(menu),)*len(menu),mode=='exact',{'mode':mode,'reason':reason})
        def choose_action(self,view):
            self.calls+=1
            return menu[-1].action
    player=Player(); rows=[]
    model=SimpleNamespace(information_key=lambda v,m:information_key(v,m,schema=HU100_SCHEMA))
    wrapper=diagnostic.Probe(player,model,{},rows.append,fallback)
    return wrapper,player,rows,menu


@pytest.mark.parametrize('fallback,check,expected',[('call',False,ActionKind.CALL),('fold',False,ActionKind.FOLD),('call',True,ActionKind.CHECK),('fold',True,ActionKind.CHECK)])
def test_unresolved_intervention_consumes_original_draw(fallback,check,expected):
    v=view(check); wrapper,player,rows,menu=probe(v,fallback=fallback)
    action=wrapper.choose_action(v)
    assert action.kind==expected and player.calls==1
    assert rows[0]['diagnostic_override']
    assert rows[0]['effective_probabilities']==[float(c.action==action) for c in menu]
    v.legal_actions.validate(action)


@pytest.mark.parametrize('mode,reason,fallback',[('translated','positive-mass-witness','call'),('exact','positive-mass','fold'),('uniform','zero-mass','call'),('uniform','missing-key',None)])
def test_released_and_stored_decisions_unchanged(mode,reason,fallback):
    v=view(); wrapper,player,rows,menu=probe(v,mode,reason,fallback)
    assert wrapper.choose_action(v)==menu[-1].action and player.calls==1
    assert not rows[0]['diagnostic_override']
    assert rows[0]['effective_probabilities']==list(player.last_decision[1])


def test_all_arms_share_the_scripted_schedule():
    from scripts.evaluate_native_hu100_baseline import make_plan as original
    spec={'name':'fixture','path':'fixture.gz','sha256':'0'*64,'format':'holdem-hu100-stored-cfr-average-research-v1','abstraction':HU100_SCHEMA}
    expected=build_schedule(original({'model':spec},'pot_pressure',8,20261013100101))
    assert build_schedule(diagnostic.make_plan(spec,8,20261013100101))==expected
