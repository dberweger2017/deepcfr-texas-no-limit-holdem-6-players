"""Statistical sample units and missing-lineage behavior in preliminary curves."""
import pytest

from scripts.report_hu20_500m_preliminary import aggregate, cell


def test_seeds_and_roles_do_not_multiply_block_sample_size():
    series={(s,100,'opponent'):{b:{'button':s+b,'big_blind':s+b+2}
                               for b in range(40)} for s in (1,2,3)}
    values=aggregate(series,[1,2,3],100,'opponent')
    assert values==[3+b for b in range(40)]
    reported=cell(values)
    assert reported['blocks']==40
    assert reported['bb_per_100']==22.5
    assert reported['bb_per_hand']==.225


def test_missing_seed_is_pending_not_smaller_aggregate():
    series={(1,100,'opponent'):{0:{'button':100,'big_blind':100}},
            (2,100,'opponent'):{0:{'button':-100,'big_blind':-100}}}
    assert aggregate(series,[1,2,3],100,'opponent') is None


def test_mismatched_block_sets_cannot_enter_paired_curve():
    series={(1,100,'opponent'):{0:{'button':10,'big_blind':10}},
            (2,100,'opponent'):{1:{'button':10,'big_blind':10}}}
    with pytest.raises(ValueError,match='block sets'):
        aggregate(series,[1,2],100,'opponent')
