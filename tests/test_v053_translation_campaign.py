"""Fixed-count science guards, complete safeguard family and pre-readout barrier."""
import pytest
from scripts import guard_v053_m1 as guard
from scripts import run_hu100_v053_translation as campaign
from scripts.evaluate_hu100_direct import PANEL_OPPONENTS


def sample(**updates):
    return dict(pressure_level=1,free_percent=50,swap_bytes=100,disk_free_bytes=40*guard.GIB,
        power_raw="Now drawing from 'Battery Power'\n 79%; discharging;",**updates)


def test_battery_allowed_and_all_resource_breaches_sustained():
    assert guard.battery_percent(sample())==79
    assert guard.sustained(sample(),0,100,0,{}) is None
    for key,value in [('pressure_level',4),('free_percent',7),
            ('swap_bytes',4_000_000_101),('disk_free_bytes',16*guard.GIB-1)]:
        bad={**sample(),key:value};since={}
        assert guard.sustained(bad,0,100,5,since) is None
        assert guard.sustained(bad,0,100,64.9,since) is None
        assert guard.sustained(bad,0,100,65,since) is not None
        assert guard.sustained(sample(),0,100,66,since) is None and not since
    with pytest.raises(ValueError,match='battery'):
        guard.battery_percent({'power_raw':'unreadable'})


def test_seven_safeguards_use_upper_strictly_below_minus_twenty():
    panel={op:dict(contrast=dict(ci95=[-100,-20])) for op in PANEL_OPPONENTS}
    assert campaign.safeguards(panel)['adopt_translation']
    for op in PANEL_OPPONENTS:
        changed={**panel,op:dict(contrast=dict(ci95=[-100,-20.00001]))}
        verdict=campaign.safeguards(changed)
        assert not verdict['adopt_translation'] and verdict['severe_regressions'][op]
    with pytest.raises(ValueError,match='seven'):
        campaign.safeguards({})


def test_fixed_counts_budget_scales_only_block_work_and_global_barrier(tmp_path):
    def cost():
        return dict(primary=dict(load_or_validation_seconds=2,play_seconds=3),
            repeat=dict(load_or_validation_seconds=5,play_seconds=7),
            audit=dict(fixed_setup_seconds=11,block_work_seconds=13),blocks=1)
    costs={name:cost() for name in ('direct',*('panel-'+o for o in PANEL_OPPONENTS))}
    quote=campaign.measured_quote(costs)
    assert campaign.COUNTS==dict(direct=131072,panel=16384)
    assert quote['fixed_seconds']==8*18
    assert quote['scaled_block_seconds']==23*(131072+7*16384)
    assert quote['active_work_budget_seconds']==1.25*(8*18+23*(131072+7*16384))+1800
    with pytest.raises(ValueError,match='All final hands'):
        campaign.readout(tmp_path)
    assert not (tmp_path/'summary.json').exists()


def test_final_resumes_and_readout_share_one_declared_science_budget(tmp_path):
    campaign.put(tmp_path/'canonical-plan.json',dict(quote=dict(active_work_budget_seconds=1000)))
    for stage,seconds in [('final-000',300),('readout-000',40)]:
        campaign.put(tmp_path/'guards'/stage/'receipt.json',dict(seconds=seconds))
    assert campaign.remaining_science_seconds(tmp_path)==660
    campaign.put(tmp_path/'guards/final-001/receipt.json',dict(seconds=661))
    with pytest.raises(ValueError,match='budget exhausted'):
        campaign.remaining_science_seconds(tmp_path)
