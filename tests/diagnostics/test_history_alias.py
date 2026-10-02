"""Real native size aliases and production keys, rather than mocked labels."""
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, choices, information_key
from src.diagnostics.flop_check import fixture_root
from src.diagnostics.history_alias import signature, action_record
from src.arena.endgame_quality import _world
from src.game.observation import replay


def test_minimum_and_pot_bets_merge_but_unequal_menus_do_not():
    root=fixture_root('limped'); hand=_world(root,replay(root,0,()).board,{})
    menu=choices(hand.observe(hand.actor),raise_cap=None,free_fold=False)
    views=[]
    for name in ('min','pot'):
        child=hand.apply(next(c.action for c in menu if c.name==name))
        view=child.observe(child.actor)
        reply=choices(view,raise_cap=None,free_fold=False)
        views.append((view,reply))
    a,b=views
    assert a[0].legal_actions.call_amount==100
    assert b[0].legal_actions.call_amount==200
    assert signature(*a)==signature(*b)
    assert information_key(*a,schema=HU20_UNCAPPED_SCHEMA)==information_key(*b,schema=HU20_UNCAPPED_SCHEMA)
    assert signature(a[0],a[1][:-1])!=signature(*a)


def test_action_record_uses_paid_amount_and_menu_name():
    root=fixture_root('min-raised'); hand=_world(root,replay(root,0,()).board,{})
    view=hand.observe(hand.actor); menu=choices(view,raise_cap=None,free_fold=False)
    item=next(c for c in menu if c.name=='pot')
    record=action_record(view,item)
    assert record['name']=='pot' and record['paid']==view.pot
    assert record['raise_to']==400
