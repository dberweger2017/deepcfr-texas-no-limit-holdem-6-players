import gzip
import json
from pathlib import Path
import pytest
from scripts.hu20_k50_first_pass_assets import TABLES, recovery_gate


def checkpoints(base, recovered_value=0.0):
    (base/'gates').mkdir()
    header = {'identity': {'tables': {name: sha for name, (_,sha) in TABLES.items()}}, 'iteration': 10}
    row = ['abc', ['check', 'raise-min'], [0.0,1.0], [1.0,2.0], 3]
    recovered = [*row]; recovered[2] = [recovered_value,1.0]
    for name, metadata, value in [('k50-fresh-4m', header, row), ('k50-resumed-4m', dict(header,native_state={'completed_nodes':4000001}), recovered)]:
        with gzip.open(base/'gates'/f'{name}.json.gz','wt') as out:
            out.write(json.dumps(metadata)+'\n'+json.dumps(value)+'\n')


def test_recovery_gate_allows_only_native_metadata(tmp_path):
    checkpoints(tmp_path)
    recovery_gate(tmp_path)
    result=json.loads((tmp_path/'recovery-gate.json').read_text())
    assert result['status']=='verified' and result['all_rows_identical']==1


def test_recovery_gate_rejects_signed_zero_change(tmp_path):
    checkpoints(tmp_path, -0.0)
    with pytest.raises(ValueError,match='row differs'):
        recovery_gate(tmp_path)
    assert not (tmp_path/'recovery-gate.json').exists()


def test_recovery_gate_rejects_extra_header_change(tmp_path):
    checkpoints(tmp_path)
    path=tmp_path/'gates/k50-resumed-4m.json.gz'
    with gzip.open(path,'rt') as source:
        header=json.loads(source.readline()); row=source.readline()
    header['iteration']=11
    with gzip.open(path,'wt') as out:
        out.write(json.dumps(header)+'\n'+row)
    with pytest.raises(ValueError,match='header differs'):
        recovery_gate(tmp_path)
