"""Frozen-budget admission, identity, independent replay and outcome barrier."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import pytest
from scripts import evaluate_hu100_direct as direct
from scripts import run_hu200_k50_translation as campaign
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from tests.test_hu200_evaluation import fixture_spec, synthetic_cards, dev_source
from src.policies.files import file_hash


def cost(seconds):
    return dict(primary=dict(load_or_validation_seconds=0,play_seconds=seconds),
        repeat=dict(load_or_validation_seconds=0,play_seconds=0),audit_seconds=0,blocks=1)


def test_counts_fixed_before_readout_and_minimum_stops(tmp_path):
    costs=dict(direct=cost(1),panel={o:cost(1) for o in direct.PANEL_OPPONENTS})
    for remaining,counts in ((400000,(131072,16384)),(200000,(65536,8192)),(100000,(32768,4096))):
        chosen,ladder=campaign.choose_counts(costs,remaining)
        assert (chosen['direct'],chosen['panel'])==counts
        assert all(not row['fits'] for row in ladder[:-1])
    with pytest.raises(ValueError,match='cannot fit'):campaign.choose_counts(costs,1)
    campaign.put(tmp_path/'all-final-verified.json',dict(status='incomplete'))
    with pytest.raises(ValueError,match='All final hands'):campaign.readout(tmp_path,{})
    assert not (tmp_path/'summary.json').exists()


def test_translation_run_owns_tables_identity_and_reproduction(tmp_path,monkeypatch):
    source=dev_source(monkeypatch);root=9021841;blocks=3
    dummy=[fixture_spec(tmp_path/'pre-a',name='on'),fixture_spec(tmp_path/'pre-b',name='off')]
    cards=synthetic_cards(tmp_path/'tables',dummy,root,blocks,monkeypatch)
    spec=fixture_spec(tmp_path/'model',HU200_EQUITY_SCHEMA,'off',cards)
    specs=[dict(spec,name='on',translation=asdict(TranslationOptions())),dict(spec,translation=None)]
    registry,done=direct.execute(specs,blocks,root,'translation',tmp_path/'play',source,
        stack_bb=200,opponent='pot_pressure',cache=tmp_path/'cache')
    assert registry.models['on'].entries is registry.models['off'].entries
    assert registry.models['on'].translation is not None and registry.models['off'].translation is None
    checked=direct.audit(tmp_path/'play',tmp_path/'replay.json',cost_only=True)
    assert checked['hands_replayed']==done['hands']==12 and 'bb_per_100' not in checked
    _,repeat=direct.execute(specs,blocks,root,'translation',tmp_path/'repeat',source,
        reproduce=tmp_path/'play',stack_bb=200,opponent='pot_pressure',cache=tmp_path/'cache')
    assert repeat['reproduced_all_hands_and_decisions']
    left=map(json.loads,gzip.open(tmp_path/'play/decisions.jsonl.gz','rt'))
    right=map(json.loads,gzip.open(tmp_path/'repeat/decisions.jsonl.gz','rt'))
    assert all(direct.stable_decision(a)==direct.stable_decision(b) for a,b in zip(left,right,strict=True))
    inputs=tmp_path/'play/inputs.json';value=json.loads(inputs.read_text())
    value['play_options']['on']['action_translation']['max_states']=1
    inputs.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='run identity'):direct.audit(tmp_path/'play',tmp_path/'bad.json')


def test_v1_fixed_match_hashes(tmp_path):
    from scripts.check_v1_translation_identity import stream
    expected={10000:'6de5d7774cbc2a3fb4dc26fb52345e7e1efb6f28504a9f4f612b27565b455693',
        20000:'0caa010deb641780d6de5cfe09023eff16f4c71856b47c58d3daf65c8f66502d'}
    for stack,sha in expected.items():
        receipt=stream(stack,tmp_path/str(stack))
        assert receipt['trace_sha256']==sha and receipt['translated']==64
