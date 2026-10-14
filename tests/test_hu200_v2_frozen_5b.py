"""Declared menu experiment: one sizing choice and a panel-only decision."""
from datetime import datetime, timezone

import pytest

from scripts import run_hu200_v2_frozen_5b as campaign


def cost():
    return dict(primary=dict(load_or_validation_seconds=0,play_seconds=1),
                repeat=dict(load_or_validation_seconds=0,play_seconds=0),audit_seconds=0,blocks=1)


def test_counts_are_selected_once_from_the_two_predeclared_options():
    costs = dict(direct=cost(),panel={op:cost() for op in campaign.direct.PANEL_OPPONENTS})
    full, options = campaign.select_counts(costs,400000)
    assert (full['direct'],full['panel'],len(options)) == (131072,16384,1)
    reduced, options = campaign.select_counts(costs,160000)
    assert (reduced['direct'],reduced['panel'],len(options)) == (65536,8192,2)
    assert not options[0]['fits']
    with pytest.raises(ValueError,match='Neither'):
        campaign.select_counts(costs,1000)


def test_rule_requires_positive_pressure_lower_and_all_panel_safeguards():
    panel={op:dict(contrast=dict(ci95=[-10,10])) for op in campaign.direct.PANEL_OPPONENTS}
    assert not campaign.recommendation(panel)[0]
    panel['pot_pressure']['contrast']['ci95']=[0,20]
    assert not campaign.recommendation(panel)[0]
    panel['pot_pressure']['contrast']['ci95']=[.01,20]
    assert campaign.recommendation(panel)[0]
    panel['half_pot_pressure']['contrast']['ci95']=[-30,-20]
    assert campaign.recommendation(panel)[0]
    panel['half_pot_pressure']['contrast']['ci95']=[-30,-20.01]
    assert not campaign.recommendation(panel)[0]
    del panel['random']
    with pytest.raises(ValueError,match='seven'):
        campaign.recommendation(panel)


def test_fixed_closeout_is_22_madrid_with_a_science_reserve():
    assert datetime.fromtimestamp(campaign.END,timezone.utc).isoformat()=='2026-10-14T20:00:00+00:00'
    assert campaign.END-campaign.SCIENCE_END==1800


def test_audited_specs_pass_real_model_admission_before_hands(tmp_path):
    from tests.test_hu200_evaluation import fixture_spec
    from src.arena.registry import PolicyRegistry
    from src.blueprint.abstraction import HU200_SCHEMA, HU200_MENU_V2_SCHEMA
    models=[]
    for name, schema in (('left',HU200_SCHEMA),('right',HU200_MENU_V2_SCHEMA)):
        toy=fixture_spec(tmp_path/name,schema,name)
        receipt=dict(status='verified',all_nodes_verified=1,counts=dict(entries=1),
                     average_sha256=toy['sha256'],checkpoint_sha256='a'*64)
        spec=campaign.evaluation_spec(tmp_path/name/'average.gz',name,schema,None,receipt)
        assert spec['entries']==1 and spec['iteration']==10 and spec['source_checkpoint_sha256']=='a'*64
        models.append(spec)
    plan=campaign.direct.make_plan(models,2,270,'admission',stack_bb=200)
    registry=PolicyRegistry(plan,sorted_average_rows=True)
    campaign.direct.validate(registry,models,plan)
    assert not list(tmp_path.rglob('hands.jsonl.gz'))


def test_admission_continuation_refuses_any_existing_hand_file(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'BASE',tmp_path)
    (tmp_path/'pilot-direct.log').write_text("KeyError: 'entries'\n")
    (tmp_path/'hands.jsonl.gz').write_bytes(b'preserve')
    with pytest.raises(ValueError,match='pre-hand'):
        campaign.continue_admission()
    assert not (tmp_path/'baseline-spec-corrected.json').exists()
