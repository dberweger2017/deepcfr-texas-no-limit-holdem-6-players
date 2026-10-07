from copy import deepcopy
import numpy as np
import pytest
from scripts.report_global_bucket_validation import summarize, coverage, METRICS


def test_paired_board_bootstrap_preserves_constant_contrast_and_weights():
    rows=[]
    for spot,weight,value in (("a",1.,1.),("b",3.,2.)):
        for lineage in (1,2,3):
            for seat in (0,1):
                row=dict(spot=spot,board_weight=weight)
                row.update({m:value+i*.1 for i,m in enumerate(METRICS)})
                rows.append(row)
    result=summarize(rows)
    assert result["e_cross_v1"]["mean"]==pytest.approx(1.75)
    assert result["e_cross_v1"]["ci95"]==pytest.approx([1.,2.])
    assert result["global200_minus_global50"]["mean"]==pytest.approx(.1)
    assert result["global200_minus_global50"]["ci95"]==pytest.approx([.1,.1])
    reversed_result=summarize(list(reversed(rows)))
    assert reversed_result==result


def test_coverage_weights_mass_and_requires_all_frozen_cells():
    rows=[]
    for fold in (0,1):
        for lineage in (1,2,3):
            rows.append(dict(evaluation_fold=fold,lineage=lineage,board_weight=2.,coverage={
                m:{s:[100.,1.,1] for s in ("turn","river")} for m in ("e_global50","e_global200")}))
    cells=coverage(rows)
    assert len(cells)==24
    assert all(c["fraction"]==.01 and c["total_reach"]==200 for c in cells)
    with pytest.raises(ValueError,match="coverage cell"):coverage(rows[:-1])
    broken=deepcopy(rows);broken[0]["coverage"]["e_global50"]["turn"][1]=101
    with pytest.raises(ValueError,match="missing decision reach"):coverage(broken)
