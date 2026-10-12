"""Archive selection keeps raw evidence and rejects changed indexed dependencies."""
import hashlib
import json
from pathlib import Path
import pytest
from scripts import archive_hu20_k50_first_pass as archive


def fixture(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    base=archive.BASE;base.mkdir(parents=True)
    def pin(name):
        path=base/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(name.encode())
        return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def put(name,value):
        (base/name).write_text(json.dumps(value))
    table_members=[pin('tables/'+n) for n in ('flop-k50.bin','turn-k50.bin','river-k50.bin')]
    put('tables-restoration.json',{'members':table_members,'archive':{},'drive_id':'table-id','drive_folder':'folder','retrieval':'table-command'})
    reference=pin('inputs/reference.json.gz')
    put('reference-restoration.json',{'member':reference,'archive':{},'drive_folder':'folder','archive_member':'reference','retrieval':'reference-command'})
    control=pin('inputs/control.json.gz')
    put('controls-restoration.json',[{'member':control,'archive':{},'archive_member':'control','manifest_sha256':'manifest','retrieval':'control-command'}])
    release=pin('release/released.jsonl.gz')
    put('release-restoration.json',{'asset':release,'retrieval':'release-command'})
    evidence=pin('direct/final-primary/all-actions.hands.jsonl.gz')
    new_model=pin('training/K50-new.json.gz')
    closed_guard=pin('guards/operations/training/resources.jsonl')
    pin('guards/operations/archive-final/resources.jsonl')
    pin('guards/status.json')
    archive.BINARY.parent.mkdir(parents=True);archive.BINARY.write_bytes(b'binary')
    Path('planning').mkdir();Path('planning/decisions.json').write_text('{}')
    monkeypatch.setattr(archive.subprocess,'check_output',lambda *a,**k:'')
    return reference,evidence,new_model,closed_guard


def test_archive_keeps_all_new_models_actions_and_closed_guard_logs(tmp_path,monkeypatch):
    reference,evidence,new_model,closed_guard=fixture(tmp_path,monkeypatch)
    files,excluded=archive.inventory()
    paths={str(p) for p,_ in files}
    assert {evidence['path'],new_model['path'],closed_guard['path']} <= paths
    assert reference['path'] not in paths
    assert len(excluded)==6
    assert str(archive.BASE/'guards/operations/archive-final/resources.jsonl') not in paths
    assert any(member=='provenance/planning/decisions.json' for _,member in files)


def test_archive_refuses_changed_indexed_input(tmp_path,monkeypatch):
    reference,*_=fixture(tmp_path,monkeypatch)
    Path(reference['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='mismatch'):
        archive.inventory()
