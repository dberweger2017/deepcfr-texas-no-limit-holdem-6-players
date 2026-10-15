"""Real-reader trace hashing and K50-v2 paired replay/reproduction on tiny tables."""
from dataclasses import asdict
from pathlib import Path
from scripts import check_hu100_release_trace as trace
from scripts import evaluate_hu100_direct as direct
from src.blueprint.average import AveragePolicy
from src.blueprint.abstraction import HU100_SCHEMA, HU100_EQUITY_MENU_V2_SCHEMA
from src.blueprint.action_translation import TranslationOptions
from tests.test_hu200_evaluation import fixture_spec,synthetic_cards,dev_source


def test_common_trace_harness_ignores_only_elapsed_time(tmp_path,monkeypatch):
    dev_source(monkeypatch)
    spec=fixture_spec(tmp_path/'model',HU100_SCHEMA)
    pin={**trace.RELEASES['v0.5.2']['model'],'bytes':spec['bytes'],'sha256':spec['sha256'],
        'checkpoint_sha256':spec['source_checkpoint_sha256'],'seed':47,'iteration':10,'entries':1}
    monkeypatch.setitem(trace.RELEASES,'v0.5.2',{**trace.RELEASES['v0.5.2'],'model':pin})
    cache=tmp_path/'cache';model=Path(spec['path'])
    AveragePolicy(model,spec['sha256'],expected_schema=HU100_SCHEMA,sorted_rows=True,cache=cache)
    left=trace.check('v0.5.2',model,cache,tmp_path/'left')
    right=trace.check('v0.5.2',model,cache,tmp_path/'right')
    assert left['trace_sha256']==right['trace_sha256']
    assert left['all_hands_independently_replayed'] and left['hands']==256
    assert not left['outcomes_reported'] and 'bb_per_100' not in left


def test_hu100_k50_v2_on_menu_panel_replays_and_reproduces(tmp_path,monkeypatch):
    source=dev_source(monkeypatch);root=9817283
    dummy=[fixture_spec(tmp_path/'pre-a',HU100_SCHEMA,'a'),fixture_spec(tmp_path/'pre-b',HU100_SCHEMA,'b')]
    cards=synthetic_cards(tmp_path/'tables',dummy,root,3,monkeypatch,stack_bb=100)
    spec=fixture_spec(tmp_path/'model',HU100_EQUITY_MENU_V2_SCHEMA,'off',cards)
    specs=[dict(spec,name='on',translation=asdict(TranslationOptions())),dict(spec,translation=None)]
    _,done=direct.execute(specs,3,root,'translation',tmp_path/'play',source,
        opponent='half_pot_pressure',cache=tmp_path/'cache')
    checked=direct.audit(tmp_path/'play',tmp_path/'replay.json',cost_only=True)
    assert done['hands']==checked['hands_replayed']==12
    assert all(cell['off_menu']==0 for cell in checked['scripted_bets_against_candidate_menu'].values())
    _,repeat=direct.execute(specs,3,root,'translation',tmp_path/'repeat',source,
        opponent='half_pot_pressure',cache=tmp_path/'cache',reproduce=tmp_path/'play')
    assert repeat['reproduced_all_hands_and_decisions']
