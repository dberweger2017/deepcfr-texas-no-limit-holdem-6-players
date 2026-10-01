"""Legality, public-only assumptions, reproducibility and score accounting."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.blueprint.search import DECK
from src.diagnostics.strong_rollout import AssumedRange, StrongRollout, action_menu, expand_range
from src.game.hand import Hand, Table
from src.game.play import RandomPolicy
from src.game.types import Action, ActionKind

CONFIG=json.loads(Path('configs/diagnostics/strong-rollout-hu20-v1.json').read_text())


def test_explicit_ranges_and_modes():
    assert expand_range(['99+'])=={'99p','TTp','JJp','QQp','KKp','AAp'}
    assert expand_range(['AQs+','AKo'])=={'AQs','AKs','AKo'}
    hand=Hand.start(Table(('a','b'),(2000,2000)),hand_id='range-menu',seed=91)
    restricted=action_menu(hand.observe(0),'restricted');native=action_menu(hand.observe(0),'native')
    assert all(c.action.raise_to!=250 for c in restricted)
    assert any(c.action.raise_to==250 for c in native)
    assert all(c.action.kind!=ActionKind.FOLD for c in action_menu(hand.apply(Action(ActionKind.CALL)).observe(1),'native'))


def same_view_worlds():
    deck=list(DECK);other=list(deck)
    other[0],other[4]=other[4],other[0];other[2],other[5]=other[5],other[2]
    other[6:]=reversed(other[6:])
    table=Table(('a','b'),(2000,2000))
    return [Hand.from_deck(table,hand_id='isolated',deck=tuple(d)) for d in (deck,other)]


def test_hidden_cards_and_future_deck_do_not_change_decision_or_rng():
    hands=same_view_worlds();views=[h.observe(0) for h in hands]
    assert views[0]==views[1]
    policies=[StrongRollout(17,CONFIG) for _ in range(2)]
    assert policies[0].choose_action(views[0])==policies[1].choose_action(views[1])
    assert policies[0].telemetry==policies[1].telemetry
    assert policies[0].random.getstate()==policies[1].random.getstate()
    assert policies[0].belief.random.getstate()==policies[1].belief.random.getstate()
    with pytest.raises(TypeError):policies[0].choose_action(hands[0])


def test_range_likelihood_updates_and_card_removal_are_public():
    hand=Hand.start(Table(('a','b'),(2000,2000)),hand_id='posterior',seed=91)
    belief=AssumedRange(12,CONFIG);before=belief.update(hand.observe(0))
    hand=hand.apply(Action(ActionKind.CALL));hand=hand.apply(Action(ActionKind.RAISE,300))
    after=belief.update(hand.observe(0))
    assert before!=after and sum(w for _,w in after)==pytest.approx(1)
    hand=hand.apply(Action(ActionKind.CALL))
    # On flop, seat 1 acts first. An observer can update without being the actor.
    rows=belief.update(hand.observe(0))
    known=set(hand.observe(0).hole_cards+hand.observe(0).board)
    assert all(not known.intersection(pair) for pair,_ in rows)
    other=Hand.start(Table(('a','b'),(2000,2000)),hand_id='another',seed=92)
    with pytest.raises(ValueError,match='hand-local'):belief.update(other.observe(0))


@pytest.mark.parametrize('mode',['restricted','native'])
def test_complete_hands_repeat_with_legal_actions_and_scores(mode):
    config={**CONFIG,'particles':16,'equity_worlds':8}
    def run():
        hand=Hand.start(Table(('a','b'),(2000,2000)),hand_id='repeat',seed=14)
        rival=RandomPolicy(73);hero=StrongRollout(41,config,mode);actions=[]
        for _ in range(100):
            if hand.finished:break
            view=hand.observe(hand.actor)
            action=hero.choose_action(view) if hand.actor==0 else rival.choose_action(view)
            view.legal_actions.validate(action);actions.append(action);hand=hand.apply(action)
        assert hand.finished and sum(p.stack for p in hand.observe(0).players)==4000
        return actions,hero.telemetry,hero.random.getstate(),hero.chance.getstate()
    assert run()==run()


def test_raise_score_is_conditioned_on_continuations():
    hand=Hand.start(Table(('a','b'),(2000,2000)),hand_id='score',seed=14)
    hand=hand.apply(Action(ActionKind.CALL)).apply(Action(ActionKind.CHECK))
    view=hand.observe(hand.actor);hero=StrongRollout(8,{**CONFIG,'particles':32,'equity_worlds':32})
    hero.choose_action(view);trace=hero.telemetry[-1]
    assert trace['scores'] and trace['range']['effective_sample_size']>0
    raises=[s for s in trace['scores'] if s['raise_to'] is not None]
    assert raises and any(s['continuing_equity']!=trace['equity'] for s in raises)
    own,rival=view.players[view.seat],view.players[1-view.seat]
    for score in raises:
        paid=score['raise_to']-own.street_bet;owed=min(rival.stack,max(0,score['raise_to']-rival.street_bet))
        matched=min(own.contributed+paid,rival.contributed+owed)
        expected=score['estimated_fold']*rival.contributed+(1-score['estimated_fold'])*(2*score['continuing_equity']-1)*matched
        assert score['score_chips']==pytest.approx(expected)
    assert sum(s['probability'] for s in trace['scores'])==pytest.approx(1)


def test_match_replay_and_paired_metric_arithmetic():
    from src.diagnostics.strong_evaluation import match, replay_record, summarize_matches
    config={**CONFIG,'particles':16,'equity_worlds':8}
    rows=[match(config,'native',92,0,r,control='jam') for r in (0,1)]
    for row in rows:assert replay_record(row).finished
    result=summarize_matches(rows)[0]
    assert result['overall']['bb_per_100']==sum(r['target_chips'] for r in rows)/2
    assert result['counts']['hands']==2
    assert 'fallback' not in result['counts']
    with pytest.raises(ValueError,match='Incomplete paired'):summarize_matches(rows[:1])
    with pytest.raises(ValueError,match='Duplicate'):summarize_matches(rows+[rows[0]])
