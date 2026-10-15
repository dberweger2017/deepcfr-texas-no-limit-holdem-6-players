"""Witness keys bind to policy tables on every street, never process registration."""
from dataclasses import replace
from pathlib import Path
import pytest

from src.blueprint.abstraction import (HU100_EQUITY_SCHEMA, HU200_EQUITY_SCHEMA,
    HU100_MENU_V2_SCHEMA, HU200_MENU_V2_SCHEMA, HU100_EQUITY_MENU_V2_SCHEMA,
    HU200_EQUITY_MENU_V2_SCHEMA, choices, information_key, schema_choices)
from src.blueprint.action_translation import TranslationOptions, translate, version_for_schema
from src.blueprint.average import AveragePolicy
from src.blueprint import equity_buckets
from src.blueprint.equity_buckets import EquityCards
from src.blueprint.solver import HU100_GAME, HU200_GAME
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from tests.native_equity_tables import write_tables
from tests.test_hu100_action_translation import stored


@pytest.mark.parametrize('stack,schema,game', [(10000,HU100_EQUITY_SCHEMA,HU100_GAME),
    (20000,HU200_EQUITY_SCHEMA,HU200_GAME),
    (10000,HU100_EQUITY_MENU_V2_SCHEMA,HU100_GAME)])
@pytest.mark.parametrize('board_size', [0,3,4,5])
def test_witness_uses_own_tables_and_legal_history(tmp_path, monkeypatch, stack, schema, game, board_size):
    h=Hand.start(Table(('a','b'),(stack,stack)),hand_id='witness',seed=12)
    initial=h
    scenes=[]
    full=h
    while not full.finished:
        v=full.observe(full.actor)
        if len(v.board)>=3: scenes.append((v.hole_cards,v.board))
        full=full.apply(Action(ActionKind.CHECK if ActionKind.CHECK in v.legal_actions.kinds else ActionKind.CALL))
    tables=tmp_path/'tables';tables.mkdir();write_tables(tables,scenes)
    cards=EquityCards(tables,pinned=False)
    monkeypatch.setattr(equity_buckets,'EQUITY_K50_SHA256',cards.sha256)
    # Deliberately poison legacy registration: a witness must use `cards`.
    monkeypatch.setattr(equity_buckets,'_registered',object())
    while len(h.observe(h.actor).board)<board_size:
        v=h.observe(h.actor)
        h=h.apply(Action(ActionKind.CHECK if ActionKind.CHECK in v.legal_actions.kinds else ActionKind.CALL))
    start=h.observe(h.actor)
    witness_choice=next(c for c in schema_choices(start,schema) if c.name=='pot')
    real=h.apply(Action(ActionKind.RAISE,550));view=real.observe(real.actor)
    witness=h.apply(witness_choice.action);wv=witness.observe(witness.actor)
    menu=schema_choices(view,schema)
    key=information_key(wv,schema_choices(wv,schema),schema=schema,equity_cards=cards)
    m=AveragePolicy.__new__(AveragePolicy)
    m.players=2;m.raise_cap=None;m.abstraction=schema;m.game=game;m.identity={'card_tables':cards.sha256}
    m.equity_cards=cards;m.entries={key:stored(menu)};m.zero_mass=set();m.description={}
    m.configure_missing_key_fallback('uniform');m.configure_translation(TranslationOptions())
    a=m.distribution_with_telemetry(view);b=m.distribution_with_telemetry(view)
    stable=lambda info:{k:v for k,v in info.items() if k!='lookup_seconds'}
    assert a[:3]==b[:3] and stable(a[3])==stable(b[3])
    assert a[3]['mode']=='translated' and a[3]['selected_key']==key
    assert key==information_key(view,menu,schema=schema,equity_cards=cards,
        history_label_overrides=dict(a[3]['overrides']))
    assert m.description['action_translation']['version']==version_for_schema(schema)
    assert 'equity-k50' in version_for_schema(schema)
    from scripts.evaluate_hu200_translation import verify_witness
    verify_witness(initial,view,menu,a[3],m)
    assert m.policy(31).choose_action(view)==m.policy(31).choose_action(view)
    if schema==HU100_EQUITY_MENU_V2_SCHEMA:
        from random import Random
        player=m.policy(891);expected=Random(891)
        expected_action=expected.choices(menu,weights=a[1],k=1)[0].action
        assert player.choose_action(view)==expected_action
        assert player.random.getstate()==expected.getstate()
        limited=translate(view,menu,m.entries,set(),TranslationOptions(max_states=1),schema=schema,equity_cards=cards)
        assert limited.key is None and limited.bound_reached and limited.states==1
        assert m.distribution_with_telemetry(wv)[3]['states']==0
        exact=information_key(view,menu,schema=schema,equity_cards=cards)
        m.entries[exact]=(tuple(c.name for c in menu),(1/len(menu),)*len(menu));m.zero_mass.add(exact)
        zero=m.distribution_with_telemetry(view)
        assert zero[2] and zero[3]['reason']=='zero-mass' and zero[3]['states']==0
        del m.entries[exact];m.zero_mass.clear()
        if board_size==0:
            jam=initial.apply(Action(ActionKind.RAISE,stack));jv=jam.observe(jam.actor)
            assert m.distribution_with_telemetry(jv)[3]['mode']=='uniform'
    m.configure_translation(None)
    assert m.distribution(view)==(menu,(1/len(menu),)*len(menu),False)
    assert m.distribution(view)==m.distribution_with_telemetry(view)[:3]
    with pytest.raises(ValueError,match='own equity tables'):
        translate(view,menu,m.entries,set(),TranslationOptions(),schema=schema)
    m.identity={'card_tables':{**cards.sha256,'flop':'f'*64}}
    with pytest.raises(ValueError,match='table identity'):
        m.configure_translation(TranslationOptions())


@pytest.mark.parametrize('schema', [HU100_MENU_V2_SCHEMA,HU200_MENU_V2_SCHEMA,
    HU200_EQUITY_MENU_V2_SCHEMA])
def test_v2_translation_explicitly_refused(schema):
    with pytest.raises(ValueError,match='supported schema'):
        version_for_schema(schema)
    m=AveragePolicy.__new__(AveragePolicy);m.abstraction=schema;m.game=HU200_GAME
    with pytest.raises(ValueError,match='matching HU100/HU200'):
        m.configure_translation(TranslationOptions())
