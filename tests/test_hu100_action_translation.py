"""Deterministic observation-boundary and witness fixtures."""
from dataclasses import replace
from random import Random

import pytest

from src.blueprint.abstraction import HU100_SCHEMA, choices, information_key
from src.blueprint.action_translation import Betting, TranslationOptions, translate
from src.blueprint.average import AveragePolicy
from src.blueprint.solver import HU100_GAME
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken, BlindPosted, BoardDealt
from src.game.types import Action, ActionKind


def hand():
    return Hand.start(Table(("a", "b"), (10000, 10000)), hand_id="fixture", seed=12)


def model(entries=(), zero=(), enabled=True):
    value = AveragePolicy.__new__(AveragePolicy)
    value.players = 2; value.raise_cap = None; value.abstraction = HU100_SCHEMA
    value.game = HU100_GAME
    value.entries = dict(entries); value.zero_mass = set(zero); value.description = {}
    value.configure_translation(TranslationOptions() if enabled else None)
    return value


def offmenu(initial=None):
    return (initial or hand()).apply(Action(ActionKind.RAISE, 550)).observe(1)


def witness_key(view, label="raise-2"):
    menu = choices(view, raise_cap=None, free_fold=False)
    index = next(i for i,e in enumerate(view.history) if isinstance(e,ActionTaken)
                 and e.action.kind == ActionKind.RAISE)
    return menu, information_key(view, menu, schema=HU100_SCHEMA,
                                history_label_overrides={index:label})


def stored(menu):
    return tuple(c.name for c in menu), (0., 1., *([0.]*(len(menu)-2)))


def test_exact_hit_keeps_probabilities_and_does_no_search():
    view=offmenu(); menu=choices(view,raise_cap=None,free_fold=False)
    key=information_key(view,menu,schema=HU100_SCHEMA); value=model([(key,stored(menu))])
    got,p,known,t=value.distribution_with_telemetry(view)
    assert got==menu and p==stored(menu)[1] and known
    assert t["mode"]=="exact" and t["states"]==0 and t["selected_key"]==key


def test_nearest_translation_uses_real_menu_and_no_rng():
    view=offmenu();menu,key=witness_key(view)
    value=model([(key,stored(menu))]);player=value.policy(7);before=player.random.getstate()
    got,p,known,t=value.distribution_with_telemetry(view)
    assert player.random.getstate()==before
    assert not known and got==menu and p==stored(menu)[1] and t["mode"]=="translated"
    assert t["selected_key"]==key and t["witness_raise_to"]==(300,)
    assert t["distance"]==pytest.approx(abs(500/650-250/400))
    assert t["all_in_changes"]==0 and 0<t["states"]<=512
    assert player.choose_action(view)==Action(ActionKind.CALL)


def test_uniform_not_found_disabled_and_zero_mass():
    view=offmenu();menu,key=witness_key(view)
    for value in (model(),model([(key,stored(menu))],enabled=False),
                  model([(key,stored(menu))],zero=[key])):
        got,p,known,t=value.distribution_with_telemetry(view)
        assert got==menu and p==(1/len(menu),)*len(menu) and not known and t["mode"]=="uniform"


def test_bound_and_on_menu_missing_are_uniform():
    view=offmenu();menu,key=witness_key(view)
    found=translate(view,menu,{key:stored(menu)},set(),TranslationOptions(max_states=1))
    assert found.key is None and found.bound_reached and found.states==1
    limited=translate(view,menu,{},set(),TranslationOptions(max_events=1))
    assert limited.bound_reached and limited.states==0
    supported=hand().apply(Action(ActionKind.RAISE,300)).observe(1)
    _,_,_,info=model().distribution_with_telemetry(supported)
    assert info["states"]==0 and info["mode"]=="uniform"


def test_hidden_information_does_not_change_key_distribution_or_choice():
    deck=[r+s for r in "23456789TJQKA" for s in "cdhs"]
    other=deck.copy()
    # Button 0: seat 1's private positions are 0 and 2. All other
    # opponent cards and undealt cards change in this privileged test host.
    positions=[i for i in range(52) if i not in (0,2)]
    shuffled=[deck[i] for i in positions];Random(813).shuffle(shuffled)
    for i,c in zip(positions,shuffled):other[i]=c
    table=Table(("a","b"),(10000,10000))
    a=offmenu(Hand.from_deck(table,hand_id="fixture",deck=tuple(deck)))
    b=offmenu(Hand.from_deck(table,hand_id="fixture",deck=tuple(other)))
    assert a==b
    menu,key=witness_key(a);value=model([(key,stored(menu))])
    left=value.distribution_with_telemetry(a);right=value.distribution_with_telemetry(b)
    assert left[:3]==right[:3]
    assert {k:v for k,v in left[3].items() if k!="lookup_seconds"}=={
        k:v for k,v in right[3].items() if k!="lookup_seconds"}
    assert value.policy(998).choose_action(a)==value.policy(998).choose_action(b)
    assert not hasattr(a,"deck") and not hasattr(a,"seed")


def test_public_reducer_matches_engine_on_menu_and_off_menu():
    rng=Random(902)
    for seed in range(80):
        actual=Hand.start(Table(("a","b"),(10000,10000),button=seed%2),hand_id="parity",seed=seed)
        state=Betting.start(actual.observe(actual.actor));previous=0
        for _ in range(100):
            view=actual.observe(actual.actor) if not actual.finished else None
            for event in actual.events[previous:]:
                if isinstance(event,BlindPosted):state=state.blind(event)
                elif isinstance(event,BoardDealt) and not actual.finished:
                    state=state.board(event,actual.table.button,100)
                    assert state is not None
            previous=len(actual.events)
            if actual.finished:break
            assert state.actor==view.actor
            legal=state.legal()
            assert set(legal.kinds)==set(view.legal_actions.kinds)
            assert (legal.call_amount,legal.min_raise_to,legal.max_raise_to)==(
                view.legal_actions.call_amount,view.legal_actions.min_raise_to,view.legal_actions.max_raise_to)
            assert state.menu(view)==choices(view,raise_cap=None,free_fold=False)
            assert state.pot==view.pot and state.flags==tuple((p.folded,p.all_in) for p in view.players)
            kind=rng.choice(legal.kinds)
            action=Action(kind,rng.randint(legal.min_raise_to,legal.max_raise_to)
                          if kind==ActionKind.RAISE else None)
            state,paid=state.apply(action)
            actual=actual.apply(action)
            event=next(e for e in actual.events[previous:] if isinstance(e,ActionTaken))
            assert paid==event.paid
        else:pytest.fail("Fixture exceeded action bound")


def test_bounds_reject_invalid_options():
    for kwargs in ({"max_states":0},{"max_states":True},{"max_events":513}):
        with pytest.raises(ValueError):TranslationOptions(**kwargs)



def test_open_jam_cannot_fabricate_menu_support_or_strip_all_in():
    view=hand().apply(Action(ActionKind.RAISE,10000)).observe(1)
    menu,key=witness_key(view)
    # A non-all-in opening witness cannot match the real all-in player flags.
    _,p,_,info=model([(key,stored(menu))]).distribution_with_telemetry(view)
    assert info['mode']=='uniform' and p==(1/len(menu),)*len(menu)


def test_sampling_uses_exactly_one_unchanged_draw():
    view=offmenu();menu,key=witness_key(view);value=model([(key,stored(menu))])
    player=value.policy(891);expected=Random(891)
    action=expected.choices(menu,weights=stored(menu)[1],k=1)[0].action
    assert player.choose_action(view)==action
    assert player.random.getstate()==expected.getstate()
