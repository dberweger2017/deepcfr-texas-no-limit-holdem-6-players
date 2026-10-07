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
    other=next(p for p in combinations(DECK,2) if p!=pair and not set(p)&set(view.hole_cards))
    b=_sample_world(view,{1-view.seat:((other,1),)},Random(2))
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


def test_arena_keeps_real_target_policy_and_replays_under_native_stress():
    from scripts.evaluate_robustness import play
    from scripts.play_robustness import replay_row
    rows=[]
    class TracedCall(Call):
        def __init__(self):self.calls=[]
        def distribution(self,view):
            self.calls.append((view.street,view.seat));return super().distribution(view)
    source=TracedCall()
    spec={'players':2,'name':'test'}
    play(source,spec,('pressure',),'native',1,0,2026135001,'demo',LBRConfig(1,60),rows.append)
    hand=replay_row(rows[0]);assert hand.finished and sum(hand.events[-1].stacks)==4000
    assert len(source.calls)==sum(i['logical_player']==0 for i in rows[0]['actions'])
    assert all(i['kind'] in ('check','call') for i in rows[0]['actions'] if i['logical_player']==0)


def test_report_intervals_use_paired_block_averages():
    from scripts.report_robustness import estimate
    # Three identical policy repetitions are a single paired observation per
    # block, not three independent samples. 20 BB conversion is also checked.
    seed_contrasts=np.array([[100,100,100],[-100,-100,-100],[50,50,50]])
    clustered=estimate(seed_contrasts.mean(axis=1))
    pseudorepeated=estimate(seed_contrasts.flatten())
    assert clustered['blocks']==3 and clustered['ci95'][1]>pseudorepeated['ci95'][1]
    assert clustered['buyins20_per100']==pytest.approx(clustered['bb100']/20)


def test_production_report_retains_partial_native_rows(tmp_path,monkeypatch):
    import gzip,json,sys
    from scripts.evaluate_robustness import play,Uniform
    from scripts.evaluate_hu20 import write_json
    from scripts.tp20_common import seal
    from scripts.report_robustness import main
    pre=tmp_path/'preflight';pre.mkdir();write_json(pre/'result.json',{'status':'complete'});seal(pre)
    confirm=tmp_path/'confirmation';confirm.mkdir()
    plan={'policies':[{'name':'uniform','players':2}],'hu_lineups':[['pressure']],
        'tp_lineups':[],'hu_blocks':3,'tp_blocks':3,'lbr_targets':[],'lbr_blocks':64}
    write_json(confirm/'plan.json',plan)
    with gzip.open(pre/'hands.jsonl.gz','wt') as f:pass
    seal(pre)
    rows=[]
    for b in range(3):
        for r in range(2):play(Uniform(),plan['policies'][0],('pressure',),'menu',b,r,2026135001,'confirmation',LBRConfig(1,60),rows.append)
    with gzip.open(confirm/'hands.jsonl.gz','wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    write_json(confirm/'attempts.json',[]);seal(confirm)
    write_json(tmp_path/'campaign.json',{'started':0,'status':'partial test'})
    (tmp_path/'resources.jsonl').write_text(json.dumps({'rss_bytes':1,'free_disk_bytes':10000000000,'swap':'x'})+'\n');seal(tmp_path)
    monkeypatch.setattr(sys,'argv',['report','--root',str(tmp_path),'--out',str(tmp_path/'report')])
    assert main()==True
    report=json.load(open(tmp_path/'report/results.json'))
    assert report['status']=='incomplete' and report['native_replayed_hands']==6
    assert report['results'][0]['target']['blocks']==3


def test_fixed_lbr_finishes_every_batch_despite_clock_advances(monkeypatch):
    import src.diagnostics.robustness as diagnostic
    from dataclasses import replace
    hand=Hand.start(Table(("h","t"),(2000,2000)),hand_id="fixed-batches",seed=42)
    while hand.observe(hand.actor).street!=Street.TURN:
        view=hand.observe(hand.actor)
        hand=hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL))
    view=hand.observe(hand.actor)
    pair=tuple(c for c in DECK if c not in view.hole_cards+view.board)[:2]
    lbr=finite_lbr(view,(pair,),(1.0,))
    lbr.config=LBRConfig(4,None)
    ticks=iter((0,1000,2000,3000,4000,5000,6000))
    monkeypatch.setattr(diagnostic,"perf_counter",lambda:next(ticks))
    lbr.choose_action(view)
    assert lbr.telemetry[-1]["samples"]==4
    assert lbr.telemetry[-1]["completed"] and lbr.telemetry[-1]["over_soft_budget"] is None
