"""Replication must preserve the fixed cap, resume pin and frozen membership."""
import pytest
from scripts import train_hu100_v053_seeds as campaign


def test_authorized_recipe_uses_fixed_cap_and_exact_seed_order():
    assert campaign.SEEDS == (2026101501,2026101502)
    assert campaign.CAP == 97_290_497
    args=campaign.train_args(2026101501,1_000_000_000,campaign.CAP,'target','telemetry')
    for flag,value in [('--seed',2026101501),('--roots-per-seat',1),('--nodes',1_000_000_000),
                       ('--max-entries',97_290_497),('--menu','v2'),('--average-rule','opponent-sampled')]:
        assert args[args.index(flag)+1]==value
    assert '--freeze-at-cap' not in args


def test_same_cardinality_different_keys_is_rejected():
    parent={'iteration':104400}
    final={'frozen_table':{'entries':97,'since_iteration':104400},'native_state':{'completed_nodes':5_000_000_001}}
    initial={'entries':97,'sorted_key_sha256':'aaa'}
    rows=[{'diagnostics':{'entries':97}}]
    campaign.verify_frozen(parent,final,initial,dict(initial),rows)
    with pytest.raises(ValueError,match='membership'):
        campaign.verify_frozen(parent,final,initial,dict(initial,sorted_key_sha256='bbb'),rows)
    final['frozen_table']['since_iteration']=104401
    with pytest.raises(ValueError,match='boundary'):
        campaign.verify_frozen(parent,final,initial,dict(initial),rows)


def test_whole_campaign_costs_do_not_scale_fixed_admission():
    from scripts import evaluate_hu100_v053_seeds as evaluation
    one=dict(fixed_seconds=1000,block_work_seconds=.1,blocks=64)
    costs={label:dict(one) for label in evaluation.labels()}
    choice,options=evaluation.select_counts(costs,100)
    assert len(costs)==17 and choice['fixed_seconds']==17000
    assert choice['panel']==16384 and choice['direct']==131072 and len(options)==1
    costs={label:dict(one,fixed_seconds=0,block_work_seconds=5) for label in evaluation.labels()}
    choice,options=evaluation.select_counts(costs,0)
    assert choice['panel']==8192 and choice['direct']==65536 and len(options)==2
    costs={label:dict(one,block_work_seconds=100) for label in evaluation.labels()}
    with pytest.raises(ValueError,match='Neither'):
        evaluation.select_counts(costs,0)
    with pytest.raises(ValueError,match='Every'):
        evaluation.select_counts({},0)


def test_fair_direct_v2_off_release_translation_on_replays(tmp_path,monkeypatch):
    from dataclasses import asdict
    from scripts import evaluate_hu100_direct as direct
    from src.blueprint.abstraction import HU100_SCHEMA,HU100_MENU_V2_SCHEMA
    from src.blueprint.action_translation import TranslationOptions
    from tests.test_v2_consumers import synthetic,dev_source
    source=dev_source(monkeypatch)
    specs=[dict(synthetic(tmp_path/'v2',HU100_MENU_V2_SCHEMA,'frozen'),translation=None),
        dict(synthetic(tmp_path/'v1',HU100_SCHEMA,'release'),translation=asdict(TranslationOptions()))]
    registry,done=direct.execute(specs,2,20261015286991,'qualification-test',tmp_path/'play',source,
        stack_bb=100,cache=tmp_path/'cache')
    del registry
    checked=direct.audit(tmp_path/'play',tmp_path/'replay.json',cost_only=True)
    assert checked['hands_replayed']==done['hands']
    assert {key.split('/')[0] for key in checked['coverage']}=={'frozen','release'}
    _,repeat=direct.execute(specs,2,20261015286991,'qualification-test',tmp_path/'repeat',source,
        stack_bb=100,cache=tmp_path/'cache',cache_read_only=True,reproduce=tmp_path/'play')
    assert repeat['reproduced_all_hands_and_decisions']
