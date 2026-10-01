"""No actual hidden deal enters sampled-world decision values."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.blueprint.abstraction import choices
from src.blueprint.search import DECK
from src.diagnostics.decision_counterfactual import score_decision, summarize_values
from src.game.hand import Hand, Table
from src.game.observation import Observation
from src.game.types import Action, ActionKind

CONFIG=json.loads(Path('configs/diagnostics/strong-rollout-hu20-v1.json').read_text())


class PassiveTarget:
    def distribution(self,view):
        assert isinstance(view,Observation) and view.actor==view.seat
        menu=choices(view,raise_cap=None,free_fold=False)
        weights=[float(c.action.kind in (ActionKind.CHECK,ActionKind.CALL)) for c in menu]
        return menu,tuple(p/sum(weights) for p in weights),True


def test_selection_and_estimation_use_different_samples():
    # The selected action wins only on selection; held-out evidence must retain its loss.
    result=summarize_values([[0,10],[0,20],[5,-10],[5,-20]],[1,0],2)
    assert result['selected_index']==1 and result['gap_bb']==-20
    assert result['evaluation_policy_mean_bb']==5 and result['gap_ci95_bb'][0]<-20
    with pytest.raises(ValueError):summarize_values([[0,1]], [.5,.5],2)


def test_same_observation_changed_hidden_world_same_values_and_seeds():
    deck=list(DECK);other=list(deck);other[0],other[4]=other[4],other[0];other[2],other[5]=other[5],other[2]
    other[6:]=reversed(other[6:]);table=Table(('a','b'),(2000,2000))
    hands=[Hand.from_deck(table,hand_id='world-boundary',deck=tuple(d)) for d in (deck,other)]
    views=[h.observe(0) for h in hands];assert views[0]==views[1]
    config={**CONFIG,'particles':8,'equity_worlds':4}
    results=[score_decision(v,PassiveTarget(),config,'restricted',91,selection_worlds=2) for v in views]
    assert results[0]==results[1]
    assert results[0]['world_records'] and results[0]['evaluation_worlds']==2
    assert results[0]['range']['assumption']=='heuristic-public-likelihood-v1'
    assert sum(a['root_probability'] for a in results[0]['legal_alternatives'])==pytest.approx(1)


def test_native_alternatives_keep_saved_root_mixture_and_exact_sizes():
    hand=Hand.start(Table(('a','b'),(2000,2000)),hand_id='native-alternatives',seed=17)
    result=score_decision(hand.observe(0),PassiveTarget(),{**CONFIG,'particles':8,'equity_worlds':4},'native',93,selection_worlds=2)
    offmenu=next(a for a in result['legal_alternatives'] if a['raise_to']==250)
    assert offmenu['root_probability']==0
    assert sum(a['root_probability'] for a in result['legal_alternatives'])==pytest.approx(1)
