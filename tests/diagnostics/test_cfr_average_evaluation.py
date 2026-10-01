"""Pairing, coverage and complete native hands with generated small exports."""
import json
from pathlib import Path

import pytest

from scripts.evaluate_hu20_cfr_average import run, play, summarize
from src.diagnostics.cfr_average import DiagnosticAverage, extract
from tests.diagnostics.test_cfr_average import fixture


def test_real_runner_uses_both_extractions_and_exact_pairs(tmp_path):
    _,_,checkpoint,current,spec=fixture(tmp_path);avg=tmp_path/'average.gz';exported=extract(checkpoint,spec,avg)
    models=[{**spec,'name':'current','strategy':'current','path':current.name,'bytes':current.stat().st_size},
            {**spec,'name':'average','strategy':'average','path':avg.name,'bytes':avg.stat().st_size,'sha256':exported['sha256']}]
    panels=[{'name':'uniform','rule':'uniform','contract':'menu','blocks':2},
            {'name':'native_pressure','rule':'pressure','contract':'native','blocks':2},
            {'name':'selective','rule':'selective_stackoff','contract':'menu','blocks':2}]
    result=run({'models':models,'panels':panels,'root':195,'max_seconds':30},tmp_path,tmp_path,tmp_path/'run')
    assert result['status']=='complete' and result['hands']==24 and len(result['changes'])==3
    for panel in result['panels']:
        assert panel['counts']['hands']==4
        assert sum(panel['whole_hand_partitions'][p]['sum_chips'] for p in panel['whole_hand_partitions'])==pytest.approx(panel['overall']['bb_per_100']*4)
        assert panel['counts']['fallback']==panel['coverage'].get('missing',0)


def test_changes_are_paired_and_positions_not_pooled(tmp_path):
    _,_,checkpoint,_,spec=fixture(tmp_path);avg=tmp_path/'avg.gz';data=extract(checkpoint,spec,avg)
    source=DiagnosticAverage(avg,data['sha256']);panel={'name':'passive','rule':'passive','contract':'menu'}
    base=[play(source,{**spec,'strategy':'current'},panel,196,b,r) for b in range(2) for r in (0,1)]
    rows=[]
    for seed in (1,2,3):
        rows.extend({**r,'seed':seed} for r in base)
        rows.extend({**r,'seed':seed,'strategy':'average','target_chips':r['target_chips']+(10 if r['rotation']==r['button'] else 30)} for r in base)
    result=summarize(rows)
    assert len(result['three_lineage_changes'])==1
    assert result['three_lineage_changes'][0]['average_minus_current']['bb_per_100']==20
    assert all(c['positions']['button']['bb_per_100']==10 and c['positions']['big_blind']['bb_per_100']==30 for c in result['changes'])
    with pytest.raises(ValueError,match='Duplicate'):summarize([*base,base[0]])
    with pytest.raises(ValueError,match='Incomplete'):summarize(base[:-1])
