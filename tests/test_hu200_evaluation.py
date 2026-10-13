"""Duplicate coupling, schema-owned HU200 replay, caches and matched size controls."""
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path

import pytest

from scripts import evaluate_hu100_direct as direct
from scripts.evaluate_hu200_panel import run as panel_run
from src.arena.heuristics import STYLES, StylePolicy, pot_raise
from src.arena.schedule import build_schedule, canonical
from src.blueprint import equity_buckets
from src.blueprint.abstraction import (HU200_SCHEMA, HU200_EQUITY_SCHEMA,
    HU200_MENU_V2_SCHEMA, HU200_EQUITY_MENU_V2_SCHEMA, schema_choices, information_key)
from src.blueprint.artifact import _identity, _format
from src.blueprint.average import EXTRACTIONS, ZERO_MASS_RULES, format_for_schema
from src.blueprint.solver import PilotConfig, HU200_GAME
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.policies.files import file_hash
from tests.test_v2_consumers import dev_source


def fixture_spec(folder, schema=HU200_SCHEMA, name='candidate', cards=None):
    folder.mkdir(parents=True)
    config=PilotConfig(seed=47,raise_cap=None,abstraction=schema,game=HU200_GAME)
    identity=_identity(config)
    if cards is not None:identity['card_tables']=cards.sha256
    header={'format':_format(config),'kind':'training','abstraction':schema,
        'config':asdict(config),'identity':identity,'iteration':10,'checkpoint_format':'jsonl-v2',
        'average_rule':'opponent-sampled','table':{'player_ids':['a','b'],'stacks':[20000,20000],
        'button':0,'small_blind':50,'big_blind':100,'chip_unit':'0.01'}}
    meta={'format':format_for_schema(schema),'kind':'diagnostic-inference',
        'checkpoint_header':header,'source_checkpoint_sha256':'a'*64,
        'extraction':EXTRACTIONS['opponent-sampled'],'zero_mass_rule':ZERO_MASS_RULES['uniform']}
    hand=Hand.start(Table(('a','b'),(20000,20000)),hand_id='fixture',seed=3)
    view=hand.observe(hand.actor);menu=schema_choices(view,schema)
    row=[information_key(view,menu,schema=schema,equity_cards=cards),[c.name for c in menu],
         [1/len(menu)]*len(menu),10,1]
    path=folder/'average.gz'
    with gzip.open(path,'wt') as stream:stream.write(json.dumps(meta)+'\n'+json.dumps(row)+'\n')
    return {'name':name,'path':str(path),'sha256':file_hash(path),'format':format_for_schema(schema),
        'abstraction':schema,'bytes':path.stat().st_size,'entries':1,'iteration':10,
        'source_checkpoint_sha256':'a'*64,**({'card_buckets':str(cards.directory)} if cards is not None else {})}


def synthetic_cards(folder, specs, root, blocks, monkeypatch):
    # Populate actual binary tables from the engine's scheduled full boards/holes.
    # Betting cannot change a fixed deck, so both menu variants reach only these situations.
    from src.blueprint.equity_buckets import EquityCards
    from tests.native_equity_tables import write_tables
    folder.mkdir()
    situations=[]
    for opponent in (None,*direct.PANEL_OPPONENTS):
        plan=direct.make_plan(specs,blocks,root,'fixture',stack_bb=200,opponent=opponent)
        for b in build_schedule(plan):
            h=Hand.start(Table(('a','b'),(20000,20000),button=b.button),hand_id='cards',seed=b.deal_seeds[0])
            # Deal the full board via legal check/call transitions.
            holes=[h.observe(seat).hole_cards for seat in (0,1)]
            while not h.finished:
                v=h.observe(h.actor)
                h=h.apply(Action(ActionKind.CHECK if ActionKind.CHECK in v.legal_actions.kinds else ActionKind.CALL))
            board=h.observe(0).board
            for size in (3,4,5):
                situations.extend((hole,board[:size]) for hole in holes)
    write_tables(folder,situations)
    cards=EquityCards(folder,pinned=False);cards.directory=folder
    monkeypatch.setattr(equity_buckets,'EQUITY_K50_SHA256',cards.sha256)
    return cards


@pytest.mark.parametrize('left', [HU200_SCHEMA,HU200_MENU_V2_SCHEMA,HU200_EQUITY_SCHEMA,HU200_EQUITY_MENU_V2_SCHEMA])
@pytest.mark.parametrize('right', [HU200_SCHEMA,HU200_MENU_V2_SCHEMA,HU200_EQUITY_SCHEMA,HU200_EQUITY_MENU_V2_SCHEMA])
def test_mixed_hu200_schemas_replay_and_cold_reproduction(tmp_path,monkeypatch,left,right):
    source=dev_source(monkeypatch);root=202610130702;blocks=4
    base=[fixture_spec(tmp_path/'pre-left',name='left'),fixture_spec(tmp_path/'pre-right',name='right')]
    cards=synthetic_cards(tmp_path/'tables',base,root,blocks,monkeypatch)
    models=[fixture_spec(tmp_path/'left',left,'left',cards if left in direct.EQUITY_SCHEMAS else None),
            fixture_spec(tmp_path/'right',right,'right',cards if right in direct.EQUITY_SCHEMAS else None)]
    _,done=direct.execute(models,blocks,root,'fixture',tmp_path/'play',source,stack_bb=200,cache=tmp_path/'cache')
    checked=direct.audit(tmp_path/'play',tmp_path/'replay.json',cost_only=True)
    assert done['hands']==checked['hands_replayed']==8
    assert done['decisions']==checked['actions_replayed']
    _,repeat=direct.execute(models,blocks,root,'fixture',tmp_path/'repeat',source,
        reproduce=tmp_path/'play',stack_bb=200,cache=tmp_path/'cache')
    assert repeat['reproduced_all_hands_and_decisions']
    for name in ('hands.jsonl.gz','decisions.jsonl.gz'):
        assert gzip.decompress((tmp_path/'play'/name).read_bytes())==gzip.decompress((tmp_path/'repeat'/name).read_bytes())


def test_panel_all_opponents_pairs_arms_and_reports_only_costs_in_pilot(tmp_path,monkeypatch):
    source=dev_source(monkeypatch)
    models=[fixture_spec(tmp_path/'left',HU200_SCHEMA,'left'),fixture_spec(tmp_path/'right',HU200_MENU_V2_SCHEMA,'right')]
    result=panel_run(models,3,202610130703,tmp_path/'panel',source,cache=tmp_path/'cache',pilot=True)
    assert tuple(result['panels'])==direct.PANEL_OPPONENTS
    for name,cost in result['panels'].items():
        assert cost['primary']['hands']==12 and cost['repeat']['reproduced_all_hands_and_decisions']
        report=json.loads((tmp_path/'panel'/name/'replay.json').read_text())
        assert 'bb_per_100' not in report and 'decision' not in report and 'ci95' not in report
        assert set(report['arm_sd_bb_per_100'])=={'candidate','baseline'}
        rows=list(map(json.loads,gzip.open(tmp_path/'panel'/name/'play/hands.jsonl.gz','rt')))
        assert {r['arm'] for r in rows}=={'candidate','baseline'}
        assert all(r['initial_stacks']==[20000,20000] for r in rows)
        ds=list(map(json.loads,gzip.open(tmp_path/'panel'/name/'play/decisions.jsonl.gz','rt')))
        # Both arms reset their own instances to the same logical-role streams.
        for b in range(3):
            for r in (0,1):
                for role in (0,1):
                    streams={d['seed'] for d in ds if d['block']==b and d['rotation']==r and d['role']==role}
                    assert len(streams)<=1
                    if streams:
                        block=build_schedule(direct.make_plan(models,3,202610130703,'panel',stack_bb=200,opponent=name))[b]
                        assert streams=={direct.action_seed(202610130703,block,r,role)}
    checked=direct.audit(tmp_path/'panel'/'half_pot_pressure'/'play',tmp_path/'report.json',report=True)
    assert 'bonferroni_seven_ci' in checked and set(checked['arms'])=={'candidate','baseline'}
    # Independently recount block means; neither correlated seats nor arms are samples.
    from statistics import mean, stdev
    from math import sqrt
    from scipy.stats import t
    rows=list(map(json.loads,gzip.open(tmp_path/'panel/half_pot_pressure/play/hands.jsonl.gz','rt')))
    chip={(r['block'],r['rotation'],r['arm']):r['candidate_chips'] for r in rows}
    sample=[sum(chip[b,r,'candidate']-chip[b,r,'baseline'] for r in (0,1))/2 for b in range(3)]
    center,sd=mean(sample),stdev(sample)
    assert checked['bb_per_100']==center and checked['paired_sd_bb_per_100']==sd
    margin=float(t.ppf(1-.05/14,2))*sd/sqrt(3)
    assert checked['bonferroni_seven_ci']==[center-margin,center+margin]
    for arm in ('candidate','baseline'):
        assert checked['arms'][arm]['bb_per_100']==mean([sum(chip[b,r,arm] for r in (0,1))/2 for b in range(3)])


def test_size_controls_only_change_wager_and_preserve_rng():
    base=STYLES['pot_pressure']
    assert replace(STYLES['half_pot_pressure'],pot_numerator=3)==base
    assert replace(STYLES['matched_pot_pressure'],pot_numerator=3)==base
    # Unclipped flop opening and facing-bet raising use the post-call pot.
    h=Hand.start(Table(('a','b'),(20000,20000)),hand_id='sizes',seed=3)
    h=h.apply(Action(ActionKind.CALL)).apply(Action(ActionKind.CHECK))
    view=h.observe(h.actor)
    half,pot=pot_raise(view,1,2),pot_raise(view,2,2)
    assert half.raise_to==view.pot//2 and pot.raise_to==view.pot
    facing=h.apply(half).observe(1-h.actor)
    assert pot_raise(facing,1,2).raise_to==half.raise_to+(facing.pot+facing.legal_actions.call_amount)//2
    a=StylePolicy(STYLES['half_pot_pressure'],23);b=StylePolicy(STYLES['matched_pot_pressure'],23)
    for _ in range(50):
        assert a.choose_action(view).kind==b.choose_action(view).kind
        assert a._random.getstate()==b._random.getstate()


def test_wrong_depth_pins_cache_and_tampered_replay_fail(tmp_path,monkeypatch):
    source=dev_source(monkeypatch)
    models=[fixture_spec(tmp_path/'left',name='left'),fixture_spec(tmp_path/'right',name='right')]
    with pytest.raises(ValueError,match='game differs'):
        direct.execute(models,2,41,'bad',tmp_path/'bad-depth',source)
    with pytest.raises(ValueError,match='synced'):
        direct.local_cache(Path.home()/'Local/Research-Cloud/cache')
    link=tmp_path/'linked';link.symlink_to(Path.home()/'Local/Research-Cloud',target_is_directory=True)
    with pytest.raises(ValueError,match='synced'):direct.local_cache(link/'cache')
    registry,_=direct.execute(models,2,41,'fixture',tmp_path/'play',source,stack_bb=200,cache=tmp_path/'cache')
    traces=tmp_path/'play/decisions.jsonl.gz'
    rows=list(map(json.loads,gzip.open(traces,'rt')));rows[0]['key']='f'*32
    with gzip.open(traces,'wt') as stream:stream.write('\n'.join(map(canonical,rows))+'\n')
    complete=tmp_path/'play/complete.json';data=json.loads(complete.read_text())
    data['files'][traces.name]={'sha256':file_hash(traces),'bytes':traces.stat().st_size};complete.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='key differs'):direct.audit(tmp_path/'play',tmp_path/'bad-replay.json')
    changed=[{**models[0],'entries':2},models[1]]
    with pytest.raises(ValueError,match='Pinned'):
        direct.execute(changed,2,41,'fixture',tmp_path/'bad-pins',source,registry,stack_bb=200)


def test_hu100_main_hashes_remain_identical(tmp_path,monkeypatch):
    import hashlib
    from tests.test_hu100_direct import specs
    source=dev_source(monkeypatch)
    models=specs(tmp_path)
    _,done=direct.execute(models,32,202610130701,'identity',tmp_path/'play',source)
    assert done['hands']==64
    expected={'hands.jsonl.gz':'e0b6f3f138ff0737d12142b3054391be244e7e4848944c1b7a3e3682978dcbf6',
        'decisions.jsonl.gz':'1b901eff25b5e8e61000e48dc2c35aae3295b681c5f2e08137eb3665ea54e22c'}
    assert {name:hashlib.sha256(gzip.decompress((tmp_path/'play'/name).read_bytes())).hexdigest()
            for name in expected}==expected


def test_k50_panel_owns_tables_and_menus(tmp_path,monkeypatch):
    source=dev_source(monkeypatch);root=202610130704;blocks=2
    base=[fixture_spec(tmp_path/'pre-left',name='left'),fixture_spec(tmp_path/'pre-right',name='right')]
    cards=synthetic_cards(tmp_path/'tables',base,root,blocks,monkeypatch)
    models=[fixture_spec(tmp_path/'left',HU200_EQUITY_SCHEMA,'left',cards),
            fixture_spec(tmp_path/'right',HU200_EQUITY_MENU_V2_SCHEMA,'right',cards)]
    result=panel_run(models,blocks,root,tmp_path/'panel',source,cache=tmp_path/'cache',pilot=True)
    assert all(c['primary']['hands']==8 for c in result['panels'].values())
