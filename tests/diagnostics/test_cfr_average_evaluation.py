"""Pairing, coverage and complete native hands with generated small exports."""
import json
from pathlib import Path

import pytest

from scripts.evaluate_hu20_cfr_average import run, play, summarize
from src.blueprint.average import AveragePolicy
from src.diagnostics.cfr_average import extract
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
    from scripts.audit_hu20_cfr_average import audit_run
    assert audit_run(tmp_path/'run')['actual_hands_replayed']==24
    summary=tmp_path/'run/summary.json';summary.write_text(summary.read_text()+' ')
    with pytest.raises(ValueError,match='bytes'):audit_run(tmp_path/'run')


def test_changes_are_paired_and_positions_not_pooled(tmp_path):
    _,_,checkpoint,_,spec=fixture(tmp_path);avg=tmp_path/'avg.gz';data=extract(checkpoint,spec,avg)
    source=AveragePolicy(avg,data['sha256']);panel={'name':'passive','rule':'passive','contract':'menu'}
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


def test_interruption_preserves_records_and_finishes_only_missing_coordinates(tmp_path,monkeypatch):
    import gzip
    import subprocess
    from scripts.complete_hu20_cfr_average import complete,surviving_rows
    from scripts.audit_hu20_cfr_average import audit_run
    _,_,checkpoint,current,spec=fixture(tmp_path);avg=tmp_path/'average.gz';exported=extract(checkpoint,spec,avg)
    models=[{**spec,'name':'current','strategy':'current','path':current.name,'bytes':current.stat().st_size},
            {**spec,'name':'average','strategy':'average','path':avg.name,'bytes':avg.stat().st_size,'sha256':exported['sha256']}]
    plan={'models':models,'panels':[{'name':'uniform','rule':'uniform','contract':'menu','blocks':2}],
          'root':197,'max_seconds':3600}
    original=tmp_path/'original';assert run(plan,tmp_path,tmp_path,original)['status']=='complete'
    file=original/'average.hands.jsonl.gz'
    with gzip.open(file,'rt') as f:retained=[json.loads(line) for line in f][:2]
    file.write_bytes(gzip.compress((''.join(json.dumps(r)+'\n' for r in retained)).encode(),mtime=0)[:-8])
    assert surviving_rows(file)[0]==retained and surviving_rows(file)[1]
    before=file.read_bytes();head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    anchor=int(subprocess.check_output(['git','show','-s','--format=%ct',head],text=True))
    monkeypatch.setattr('scripts.complete_hu20_cfr_average.time',lambda:anchor+1)
    out=tmp_path/'completed';result=complete(plan,original,tmp_path,tmp_path,out,head)
    assert result['status']=='complete' and result['hands']==8
    assert result['recovery']['new_hands']==2 and result['recovery']['retained_original_hands']==6
    assert file.read_bytes()==before and (out/(file.name+'.interrupted')).read_bytes()==before
    assert (out/'current.hands.jsonl.gz').read_bytes()==(original/'current.hands.jsonl.gz').read_bytes()
    assert audit_run(out)['actual_hands_replayed']==8
    monkeypatch.setattr('scripts.complete_hu20_cfr_average.time',lambda:anchor+plan['max_seconds']+1)
    expired=complete(plan,original,tmp_path,tmp_path,tmp_path/'expired',head)
    assert expired['status']=='incomplete' and expired['hands']==0 and 'expired' in expired['failure']
