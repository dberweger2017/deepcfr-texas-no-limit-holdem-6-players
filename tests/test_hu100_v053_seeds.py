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
