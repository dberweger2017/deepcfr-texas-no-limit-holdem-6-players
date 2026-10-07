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


def test_retained_zip_response_skips_statistics_and_materializes_exact_input(tmp_path):
    import gzip
    import hashlib
    import json
    import zipfile
    from scripts.run_global_bucket_validation import response, materialize_compact
    path=tmp_path/'native.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('response.jsonl','{"event":"pooling_statistics","groups":[]}\n{"event":"gate","gate":"V1","passed":true}\n')
    assert response(path)==[{"event":"gate","gate":"V1","passed":True}]
    assert len(response(path,include_statistics=True))==2
    compact=tmp_path/'compact.json';raw=b'{"card_state":"unchanged"}\n'
    with gzip.open(str(compact)+'.gz','wb') as output:output.write(raw)
    request=tmp_path/'request.json';request.write_text(json.dumps({'compact_path':str(compact)}))
    job={'request':str(request),'compact_sha256':hashlib.sha256(raw).hexdigest()}
    _,active=materialize_compact(job)
    assert active.read_bytes()==raw
    active.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='compact hash'):materialize_compact(job)


def test_family_guard_counts_registered_trees_once_and_rejects_reused_pids(tmp_path, monkeypatch):
    import json
    from scripts.run_global_bucket_validation import family_rss
    (tmp_path/'background-roots.json').write_text(json.dumps({'roots':[
        {'pid':10,'command_fragment':'validation-run'}, {'pid':20,'command_fragment':'validation-run'},
        {'pid':99,'command_fragment':'validation-run'}]}))
    snapshot='PID PPID RSS COMMAND\n10 1 100 validation-run\n11 10 200 validation-child\n20 1 300 validation-run\n21 20 400 native\n99 1 999 unrelated-job\n'
    monkeypatch.setattr('scripts.run_global_bucket_validation.subprocess.check_output',lambda *a,**kw:snapshot)
    assert family_rss(tmp_path,10)==1000*1024
    assert family_rss(tmp_path,11)==1000*1024
