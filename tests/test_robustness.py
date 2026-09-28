from itertools import combinations
from random import Random

import numpy as np
import pytest

from src.blueprint.abstraction import choices
from src.blueprint.search import DECK, _sample_world
from src.diagnostics.robustness import (LBRConfig,LocalBestResponse,ReactiveAttack,
    checkdown_payoffs,posterior)
from src.game.hand import Hand,Table
from src.game.observation import replay
from src.game.types import Action,ActionKind,Street


class Call:
    def distribution(self,view):
        menu=choices(view,free_fold=False)
        kind=ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL
        return menu,tuple(float(c.action.kind==kind) for c in menu),True


def river(board=("2c","5d","8h","Ts","Jc"),hero=("Ac","Ad"),opponent=("3c","3d")):
    # Native deal order for button zero: seat1 then seat0 in each private round.
    prefix=(opponent[0],hero[0],opponent[1],hero[1])+board
    hand=Hand.from_deck(Table(("h","t"),(2000,2000)),hand_id="fixture",
        deck=prefix+tuple(c for c in DECK if c not in prefix))
    while hand.observe(hand.actor).street!=Street.RIVER:
        v=hand.observe(hand.actor)
        hand=hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in v.legal_actions.kinds else ActionKind.CALL))
    # seat1 checks; hero is next.
    hand=hand.apply(Action(ActionKind.CHECK))
    assert hand.actor==0
    return hand


def finite_lbr(view,pairs,weights,source=Call()):
    lbr=LocalBestResponse(source,19,LBRConfig(1,60))
    lbr.holdings=tuple(pairs); lbr.weights=np.array(weights);lbr.processed=len(view.history)
    return lbr


def checkdown(hand,hero_action):
    hand=hand.apply(hero_action)
    while not hand.finished:
        view=hand.observe(hand.actor)
        hand=hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL))
    assert sum(hand.events[-1].stacks)==4000
    return hand.events[-1].stacks[0]-2000


def test_nonuniform_bayes_from_pre_action_observation():
    root=river();view=root.observe(0)
    pairs=(("3c","3d"),("Qc","Qd"))
    class Target:
        def distribution(self,v):
            assert v.actor==v.seat==1 and not any(getattr(e,"action",None)==Action(ActionKind.RAISE,200) for e in v.history)
            menu=choices(v,free_fold=False)
            p=.2 if v.hole_cards==pairs[0] else .8
            return menu,tuple(p if c.name=="min" else 1-p if c.name=="check" else 0 for c in menu),True
    # Take the public river check back out and instead observe a target raise.
    prior=root.events[:-2]
    opponent=replay(prior,1,pairs[0]);assert opponent.actor==1
    world=_sample_world(opponent,{0:((view.hole_cards,1),)},Random(0))
    raised=world.apply(Action(ActionKind.RAISE,100))
    lbr=finite_lbr(raised.observe(0),pairs,(.75,.25),Target());lbr.processed=len(prior)
    lbr.update(raised.observe(0))
    assert lbr.weights==pytest.approx((3/7,4/7))
    p,z=posterior((.75,.25),(0,0));assert z and p==pytest.approx((.75,.25))


def test_card_updates_and_logged_zero_likelihood():
    root=Hand.start(Table(("h","t"),(2000,2000)),hand_id="x",seed=32)
    root=root.apply(Action(ActionKind.RAISE,200))
    lbr=LocalBestResponse(Call(),1); lbr.update(root.observe(1))
    assert lbr.zero_likelihood and sum(lbr.weights)==pytest.approx(1)
    # Public board collision removal exercised by a full observed history.
    hand=river(); lbr=LocalBestResponse(Call(),2);lbr.update(hand.observe(0))
    assert all(w==0 for pair,w in zip(lbr.holdings,lbr.weights) if set(pair)&set(hand.observe(0).board))


def test_exact_ledger_vs_independent_native_deals_and_bound():
    root=river();view=root.observe(0);pairs=(("3c","3d"),("Qc","Qd"));weights=(.3,.7)
    lbr=finite_lbr(view,pairs,weights)
    selected=lbr.choose_action(view)
    # Independent exhaustive one-step BR: aggregate hidden native settlements
    # before maximizing. Target check/call ends this finite river after one action.
    exact=[]
    for choice in choices(view,free_fold=False):
        vals=[]
        for pair in pairs:
            world=_sample_world(view,{1:((pair,1),)},Random(0))
            payoff=checkdown(world,choice.action)
            outcome=1 # both private holdings lose to the hero's AA on this board
            assert checkdown_payoffs(view,choice.action,[outcome])[0]==payoff
            vals.append(payoff)
        exact.append(sum(w*v for w,v in zip(weights,vals)))
    chosen_value=exact[[c.action for c in choices(view,free_fold=False)].index(selected)]
    assert chosen_value<=max(exact)+1e-12 and chosen_value==max(exact)
    assert chosen_value>view.pot/2 # exploits passive caller


def test_known_zero_value_shared_royal_flush_has_no_spurious_gain():
    root=river(("Tc","Jc","Qc","Kc","Ac"),("2h","3h"),("4h","5h"));view=root.observe(0)
    pairs=(("4h","5h"),("6h","7h"))
    lbr=finite_lbr(view,pairs,(.4,.6));action=lbr.choose_action(view)
    assert lbr.telemetry[-1]["values_chips"]==pytest.approx([0]*len(choices(view,free_fold=False)))
    assert checkdown(root,action)==0


def test_hidden_world_and_future_invariance_full_production():
    root=Hand.start(Table(("h","t"),(2000,2000)),hand_id="safe",seed=32)
    view=root.observe(root.actor)
    pair=next(p for p in combinations(DECK,2) if not set(p)&set(view.hole_cards))
    a=_sample_world(view,{1-view.seat:((pair,1),)},Random(1))
    b=_sample_world(view,{1-view.seat:((pair,1),)},Random(2))
    assert a.observe(view.seat)==b.observe(view.seat)==view
    x,y=LocalBestResponse(Call(),99,LBRConfig(1,60)),LocalBestResponse(Call(),99,LBRConfig(1,60))
    assert x.choose_action(a.observe(view.seat))==y.choose_action(b.observe(view.seat))
    assert x.telemetry[-1]["values_chips"]==y.telemetry[-1]["values_chips"]


def test_reactive_contract_raise_cap():
    hand=Hand.start(Table(("h","t"),(2000,2000)),hand_id="stress",seed=22)
    for _ in range(2):
        hand=hand.apply(ReactiveAttack("minraise","native").choose_action(hand.observe(hand.actor)))
    view=hand.observe(hand.actor)
    assert ReactiveAttack("minraise","native").choose_action(view).kind==ActionKind.RAISE
    assert ReactiveAttack("minraise","menu").choose_action(view).kind==ActionKind.CALL
    for rule in ("pressure","minraise","passive"):
        for contract in ("menu","native"):
            view.legal_actions.validate(ReactiveAttack(rule,contract).choose_action(view))


def test_mixed_hidden_values_aggregate_before_maximum():
    root=river(hero=("Qc","Qd"),opponent=("Ac","Ad"));view=root.observe(0)
    pairs=(("Ac","Ad"),("3c","3d"));weights=(.8,.2)
    lbr=finite_lbr(view,pairs,weights);selected=lbr.choose_action(view)
    expected=[];clairvoyant=[]
    for pair in pairs:
        world=_sample_world(view,{1:((pair,1),)},Random(0))
        clairvoyant.append([checkdown(world,c.action) for c in choices(view,free_fold=False)])
    for k in range(len(choices(view,free_fold=False))):
        expected.append(sum(weights[j]*clairvoyant[j][k] for j in range(2)))
    assert lbr.telemetry[-1]['values_chips']==pytest.approx(expected)
    assert expected[[c.action for c in choices(view,free_fold=False)].index(selected)]==max(expected)
    assert max(expected)<sum(weights[j]*max(clairvoyant[j]) for j in range(2))


def test_real_target_rng_not_consumed_by_hypothetical_queries():
    source=Call();source.real_rng=Random(12);before=source.real_rng.getstate()
    view=river().observe(0);LocalBestResponse(source,13,LBRConfig(1,60)).choose_action(view)
    assert source.real_rng.getstate()==before
