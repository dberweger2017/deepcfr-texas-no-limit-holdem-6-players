"""Cost admission and paired seed/seat weighting have independent fixtures."""
import math
import pytest
from scripts.run_hu20_k50_first_pass_direct import project, block_values, paired_statistics, compare_statistics


def test_quote_scales_play_not_fixed_load_and_obeys_ceiling():
    rows=[{'load_seconds':100,'play_seconds':10}]*6
    quote=project(rows,5,1000,sd=100,trace_bytes=10000)
    assert quote['projected_final_play_seconds']==1560
    assert quote['projected_final_replay_seconds']==80
    assert quote['projected_total_seconds']==4850
    assert quote['admitted']
    assert not project(rows,5,20000,sd=100,trace_bytes=10000)['admitted']
    assert not project(rows,5,1000,sd=1000,trace_bytes=10000)['admitted']


def test_paired_rotations_and_seeds_use_equal_weight_with_button_alternation():
    # Same button payoff in either rotation; button alternates physical seats.
    cells={(l,b,r):100*l+b+(10 if r==b%2 else -10)
           for l in (1,2,3) for b in range(4) for r in (0,1)}
    assert block_values(cells,4)==[200,201,202,203]
    result=paired_statistics(cells,4)
    assert result['overall']['bb_per_100']==201.5
    assert result['positions']['button']['bb_per_100']==211.5
    assert result['positions']['big_blind']['bb_per_100']==191.5
    assert result['seed_by_seat']['2026100603']['button']['bb_per_100']==311.5
    assert result['lineages']['1']['bb_per_100']==101.5
    with pytest.raises(ValueError,match='coordinate'):
        block_values({k:v for k,v in cells.items() if k!=(2,0,1)},4)


def test_independent_arithmetic_rejects_changed_interval():
    cells={(l,b,r):l*b+r for l in (1,2,3) for b in range(4) for r in (0,1)}
    expected=paired_statistics(cells,4)
    reported=paired_statistics(cells,4)
    reported['positions']['button']['ci95'][0]+=1
    with pytest.raises(ValueError,match='interval'):
        compare_statistics(expected,reported)
