from copy import deepcopy
import pytest
from scripts.report_dr2x2_ac import block_values,interval,SEEDS


def test_lbr_completion_metadata_survives_observation_wrapper_without_another_action():
    from scripts.evaluate_robustness import last_lbr_telemetry
    from src.diagnostics.robustness import LocalBestResponse
    from src.diagnostics.stackoff_tails import RecordingOpponent
    opponent=object.__new__(LocalBestResponse)
    trace={'requested_samples':4,'samples':2,'completed':False,'zero_likelihood_events':3}
    opponent.telemetry=[trace]
    wrapped=RecordingOpponent(opponent,[])
    assert last_lbr_telemetry(opponent) is trace
    assert last_lbr_telemetry(wrapped) is trace
    assert len(opponent.telemetry)==1 and wrapped.decisions==[]
    assert last_lbr_telemetry(object()) is None


def test_call_tails_come_from_decisions_without_optional_legacy_totals():
    from scripts.report_dr2x2_ac import call_counts
    row={'actions':[{'logical_player':p,'kind':k,'observation':{'call_amount':call,'stack':stack}}
                    for p,k,call,stack in [(0,'call',800,1000),(0,'call',500,500),(0,'call',0,0),(1,'call',1200,1200),(0,'raise',900,900)]]}
    assert call_counts(row)=={'large_calls':1,'allin_calls':1}


def test_three_seed_contrasts_are_inside_the_deal_block_and_roles():
    series={}
    for i,s in enumerate(SEEDS):
        for cell in ('A','C'):
            series['primary','lbr',s,cell,'current']={b:{r:(i+1)*100+(b+1)*(i+1)*(1 if cell=='C' else 0)+(10 if r=='button' else -10) for r in ('button','big_blind')} for b in range(4)}
    terms=[('C','current',1),('A','current',-1)]
    values=block_values(series,'primary','lbr',terms)
    assert values==[2,4,6,8]
    result=interval(values);assert result['blocks']==4 and result['bb_per_100']==5
    assert result['sample_variance_chips']==pytest.approx(20/3)
    assert block_values(series,'primary','lbr',terms,role='button')==values
    damaged=deepcopy(series);damaged.pop(('primary','lbr',SEEDS[1],'A','current'))
    with pytest.raises(ValueError,match='Missing'):block_values(damaged,'primary','lbr',terms)


def test_blocks_cannot_be_intersected_to_hide_partial_evidence():
    series={('primary','p',s,c,'current'):{0:{'button':1,'big_blind':2},1:{'button':3,'big_blind':4}} for s in SEEDS for c in ('A','C')}
    series['primary','p',SEEDS[0],'C','current'].pop(1)
    with pytest.raises(ValueError,match='Mismatched'):block_values(series,'primary','p',[('C','current',1),('A','current',-1)])
