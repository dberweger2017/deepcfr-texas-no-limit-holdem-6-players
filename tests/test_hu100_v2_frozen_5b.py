"""Cost admission, exact rule boundaries and the seven-script HU100 replay path."""
from dataclasses import asdict
import pytest

from scripts import evaluate_hu100_direct as direct
from scripts import evaluate_hu100_v2_frozen_5b as campaign
from src.blueprint.abstraction import HU100_SCHEMA, HU100_MENU_V2_SCHEMA
from src.blueprint.action_translation import TranslationOptions
from tests.test_v2_consumers import synthetic, dev_source


def costs(fixed=0, block_work=1):
    one = dict(fixed_seconds=fixed,block_work_seconds=block_work,blocks=64)
    return dict(direct=one,panel={c+'-'+o:one for c in ('primary','learning') for o in direct.PANEL_OPPONENTS})


def test_one_off_costs_are_never_multiplied_by_requested_blocks():
    quote=campaign.projected_cost(dict(fixed_seconds=1200,block_work_seconds=2,blocks=64),16384)
    assert quote==dict(fixed_seconds=1200,scaled_block_seconds=512,total_seconds=1712)
    choice,options=campaign.select_counts(costs(fixed=1000,block_work=.1),100)
    assert (choice['panel'],choice['direct'])==(16384,131072)
    assert choice['fixed_seconds']==15000 and len(options)==1
    choice,options=campaign.select_counts(costs(block_work=5),0)
    assert (choice['panel'],choice['direct'])==(8192,65536) and len(options)==2
    with pytest.raises(ValueError,match='Neither'):
        campaign.select_counts(costs(block_work=300),0)


def test_owner_rule_is_strict_on_pot_lower_and_panel_upper():
    panel={o:dict(contrast=dict(ci95=[-10,10])) for o in direct.PANEL_OPPONENTS}
    assert campaign.recommendation(panel)[0]
    panel['tight_aggressive']['contrast']['ci95']=[-30,-20]
    assert campaign.recommendation(panel)[0]
    panel['tight_aggressive']['contrast']['ci95']=[-30,-20.01]
    assert not campaign.recommendation(panel)[0]
    panel['tight_aggressive']['contrast']['ci95']=[-30,0]
    panel['pot_pressure']['contrast']['ci95']=[-20,10]
    assert not campaign.recommendation(panel)[0]
    panel['pot_pressure']['contrast']['ci95']=[-19.99,10]
    assert campaign.recommendation(panel)[0]
    with pytest.raises(ValueError,match='Every'):
        campaign.recommendation({'pot_pressure':panel['pot_pressure']})


@pytest.mark.parametrize('opponent',direct.PANEL_OPPONENTS)
def test_hu100_v2_off_v1_translation_on_replays_and_reproduces(tmp_path,monkeypatch,opponent):
    source=dev_source(monkeypatch)
    specs=[dict(synthetic(tmp_path/'v2',HU100_MENU_V2_SCHEMA,'frozen'),translation=None),
        dict(synthetic(tmp_path/'v1',HU100_SCHEMA,'release'),translation=asdict(TranslationOptions()))]
    _,done=direct.execute(specs,2,20261014272091,'smoke',tmp_path/'play',source,
        opponent=opponent,cache=tmp_path/'cache')
    replay=direct.audit(tmp_path/'play',tmp_path/'audit.json',cost_only=True)
    assert replay['hands_replayed']==done['hands']==8 and 'bb_per_100' not in replay
    assert replay['seconds']==pytest.approx(replay['fixed_setup_seconds']+replay['block_work_seconds'])
    assert replay['block_work_seconds']>=replay['raw_hash_seconds']>=0
    _,repeat=direct.execute(specs,2,20261014272091,'smoke',tmp_path/'repeat',source,
        opponent=opponent,cache=tmp_path/'cache',cache_read_only=True,reproduce=tmp_path/'play')
    assert repeat['reproduced_all_hands_and_decisions']
