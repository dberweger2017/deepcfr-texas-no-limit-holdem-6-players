import pytest
from scripts.report_hu200_river_final import interval


def test_block_interval_known_reference_and_bonferroni_width():
    ordinary=interval([1,2,3,4]);adjusted=interval([1,2,3,4],.05/7)
    assert ordinary['mean_bb100']==2.5
    assert ordinary['interval_bb100']==pytest.approx([.445739743239121,4.554260256760879])
    assert adjusted['interval_bb100'][0]<ordinary['interval_bb100'][0]
    with pytest.raises(ValueError):interval([float('nan'),0])
