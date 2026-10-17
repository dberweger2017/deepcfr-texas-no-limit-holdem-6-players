import pytest
from scripts.report_hu200_river_final import interval


def test_block_interval_known_reference_and_bonferroni_width():
    ordinary=interval([1,2,3,4]);adjusted=interval([1,2,3,4],.05/7)
    assert ordinary['mean_bb100']==2.5
    assert ordinary['interval_bb100']==pytest.approx([.445739743239121,4.554260256760879])
    assert adjusted['interval_bb100'][0]<ordinary['interval_bb100'][0]
    with pytest.raises(ValueError):interval([float('nan'),0])


def test_paired_prefix_rejects_turn_or_opponent_divergence_and_identical_path_mispayment():
    from copy import deepcopy
    from scripts.evaluate_hu200_river_final import verify_common_prefix
    hand=dict(decisions=[dict(index=0,seat=0,role=0,street='river',observation_sha256='same',action={'kind':'check'})],net_chips=[100,-100])
    assert verify_common_prefix(hand,deepcopy(hand)) is None
    changed=deepcopy(hand);changed['decisions'][0]['action']={'kind':'raise'}
    assert verify_common_prefix(hand,changed)==0
    for field,value in [('street','turn'),('role',1)]:
        first=deepcopy(hand);other=deepcopy(changed)
        first['decisions'][0][field]=other['decisions'][0][field]=value
        with pytest.raises(ValueError,match='outside hero river'):verify_common_prefix(first,other)
    wrong=deepcopy(hand);wrong['net_chips']=[-100,100]
    with pytest.raises(ValueError,match='settle differently'):verify_common_prefix(hand,wrong)
