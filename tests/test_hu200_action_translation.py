"""HU200 identity, public witness legality, controls and deterministic sampling."""
from dataclasses import replace
from random import Random

import pytest

from src.blueprint.abstraction import HU100_SCHEMA, HU200_SCHEMA, choices, information_key
from src.blueprint.action_translation import Betting, TranslationOptions, translate, HU200_VERSION
from src.blueprint.average import AveragePolicy
from src.blueprint.solver import HU100_GAME, HU200_GAME
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken, BlindPosted, BoardDealt
from src.game.types import Action, ActionKind
from tests.test_hu100_action_translation import stored


def initial(seed=12, button=0, deck=None):
    table=Table(('a','b'),(20000,20000),button=button)
    return (Hand.from_deck(table,hand_id='fixture',deck=deck) if deck else
            Hand.start(table,hand_id='fixture',seed=seed))


def model(entries=(), zero=(), enabled=True):
    m=AveragePolicy.__new__(AveragePolicy)
    m.players=2;m.raise_cap=None;m.abstraction=HU200_SCHEMA;m.game=HU200_GAME
    m.entries=dict(entries);m.zero_mass=set(zero);m.description={}
    m.configure_translation(TranslationOptions() if enabled else None)
    return m


def witness(view):
    menu=choices(view,raise_cap=None,free_fold=False)
    i=next(i for i,e in enumerate(view.history) if isinstance(e,ActionTaken))
    return menu,information_key(view,menu,schema=HU200_SCHEMA,history_label_overrides={i:'raise-2'})


def test_hu200_witness_has_legal_engine_history_and_real_current_amounts():
    start=initial();real=start.apply(Action(ActionKind.RAISE,550));view=real.observe(real.actor)
    menu,key=witness(view);m=model([(key,stored(menu))])
    player=m.policy(891);before=player.random.getstate()
    got,p,known,info=m.distribution_with_telemetry(view)
    assert not known and info['mode']=='translated' and info['selected_key']==key
    assert got==menu and p==stored(menu)[1] and player.random.getstate()==before
    legal_witness=start.apply(Action(ActionKind.RAISE,info['witness_raise_to'][0]))
    wv=legal_witness.observe(legal_witness.actor)
    assert information_key(wv,choices(wv,raise_cap=None,free_fold=False),schema=HU200_SCHEMA)==key
    assert view.legal_actions.call_amount==450 and wv.legal_actions.call_amount==200
    assert got[1].action==Action(ActionKind.CALL)
    expected=Random(891);expected_action=expected.choices(menu,weights=p,k=1)[0].action
    assert player.choose_action(view)==expected_action and player.random.getstate()==expected.getstate()
    assert m.description['action_translation']['version']==HU200_VERSION


def test_hu200_model_game_and_table_identity_fail_closed():
    m=model();m.game=HU100_GAME
    with pytest.raises(ValueError,match='matching HU100/HU200'):
        m.configure_translation(TranslationOptions())
    view=initial().apply(Action(ActionKind.RAISE,550)).observe(1)
    menu,_=witness(view)
    with pytest.raises(ValueError,match='versioned table'):
        translate(view,menu,{},set(),TranslationOptions(max_events=1))
    m=model();other=Hand.start(Table(('a','b'),(10000,10000)),hand_id='x',seed=1).observe(0)
    with pytest.raises(ValueError,match='versioned table'):m.distribution_with_telemetry(other)
    with pytest.raises(ValueError,match='explicit HU100/HU200'):
        translate(view,menu,{},set(),TranslationOptions(),schema='unknown')


def test_hu200_exact_zero_on_menu_and_bounds_preserve_behavior():
    view=initial().apply(Action(ActionKind.RAISE,550)).observe(1);menu,key=witness(view)
    exact=information_key(view,menu,schema=HU200_SCHEMA)
    for zero in ((),(exact,)):
        value=model([(exact,stored(menu))],zero=zero)
        _,p,known,info=value.distribution_with_telemetry(view)
        assert known and p==stored(menu)[1] and info['states']==0
        assert info['mode']==('uniform' if zero else 'exact')
    a=translate(view,menu,{key:stored(menu)},set(),TranslationOptions(max_states=1),schema=HU200_SCHEMA)
    b=translate(view,menu,{key:stored(menu)},set(),TranslationOptions(max_states=1),schema=HU200_SCHEMA)
    assert a==b and a.bound_reached and a.states==1 and a.key is None
    limited=translate(view,menu,{},set(),TranslationOptions(max_events=1),schema=HU200_SCHEMA)
    assert limited.bound_reached and limited.states==0
    supported=initial().apply(Action(ActionKind.RAISE,300)).observe(1)
    off=model(enabled=False);on=model()
    assert off.distribution_with_telemetry(supported)[:3]==on.distribution_with_telemetry(supported)[:3]
    assert on.distribution_with_telemetry(supported)[3]['states']==0
    assert off.policy(9).choose_action(supported)==on.policy(9).choose_action(supported)
    jam=initial().apply(Action(ActionKind.RAISE,20000)).observe(1);menu,key=witness(jam)
    assert model([(key,stored(menu))]).distribution_with_telemetry(jam)[3]['mode']=='uniform'


def test_hu200_hidden_cards_deck_and_seed_do_not_affect_translation():
    deck=[r+s for r in '23456789TJQKA' for s in 'cdhs'];other=deck.copy()
    positions=[i for i in range(52) if i not in (0,2)]
    shuffled=[deck[i] for i in positions];Random(813).shuffle(shuffled)
    for i,c in zip(positions,shuffled):other[i]=c
    a=initial(deck=tuple(deck)).apply(Action(ActionKind.RAISE,550)).observe(1)
    b=initial(deck=tuple(other)).apply(Action(ActionKind.RAISE,550)).observe(1)
    assert a==b
    menu,key=witness(a);m=model([(key,stored(menu))]);x=m.distribution_with_telemetry(a);y=m.distribution_with_telemetry(b)
    assert x[:3]==y[:3]
    assert {k:v for k,v in x[3].items() if k!='lookup_seconds'}=={k:v for k,v in y[3].items() if k!='lookup_seconds'}
    assert m.policy(31).choose_action(a)==m.policy(31).choose_action(b)


def test_hu200_public_reducer_matches_engine_at_deep_off_menu_sizes():
    rng=Random(903)
    for seed in range(100):
        hand=initial(seed,seed%2);state=Betting.start(hand.observe(hand.actor));previous=0
        for _ in range(100):
            for event in hand.events[previous:]:
                if isinstance(event,BlindPosted):state=state.blind(event)
                elif isinstance(event,BoardDealt) and not hand.finished:
                    state=state.board(event,hand.table.button,100);assert state is not None
            previous=len(hand.events)
            if hand.finished:break
            view=hand.observe(hand.actor);legal=state.legal()
            assert state.actor==view.actor and state.pot==view.pot
            assert set(legal.kinds)==set(view.legal_actions.kinds)
            assert (legal.call_amount,legal.min_raise_to,legal.max_raise_to)==(view.legal_actions.call_amount,view.legal_actions.min_raise_to,view.legal_actions.max_raise_to)
            assert state.menu(view)==choices(view,raise_cap=None,free_fold=False)
            kind=rng.choice(legal.kinds)
            action=Action(kind,rng.randint(legal.min_raise_to,legal.max_raise_to) if kind==ActionKind.RAISE else None)
            state,paid=state.apply(action);hand=hand.apply(action)
            assert paid==next(e.paid for e in hand.events[previous:] if isinstance(e,ActionTaken))
        else:pytest.fail('Fixture exceeded action bound')
