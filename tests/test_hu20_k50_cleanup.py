"""Deletion eligibility is narrow and stable; guards, symlinks and changes stay."""
from pathlib import Path
from scripts import cleanup_hu20_k50_first_pass as cleanup


def pin(path):
    stat=path.stat()
    return {'bytes':stat.st_size,'source_mtime_ns':stat.st_mtime_ns}


def test_cleanup_rejects_shared_paths_symlinks_and_modified_payload(tmp_path,monkeypatch):
    monkeypatch.setattr(cleanup,'ROOT',tmp_path)
    base=tmp_path/'results/own';base.mkdir(parents=True)
    monkeypatch.setattr(cleanup,'BASE',base)
    path=base/'checkpoint.json.gz';path.write_bytes(b'payload')
    snapshot=pin(path)
    assert cleanup.owned_candidate(path,snapshot)==(True,None)
    path.write_bytes(b'changed-payload')
    assert not cleanup.owned_candidate(path,snapshot)[0]
    other=tmp_path/'other-root';other.mkdir();foreign=other/'model.gz';foreign.write_bytes(b'other')
    assert not cleanup.owned_candidate(foreign,pin(foreign))[0]
    link=base/'symlink.gz';link.symlink_to(foreign)
    assert not cleanup.owned_candidate(link,pin(foreign))[0]


def test_cleanup_preserves_fixed_budget_failure_latch_and_compact_receipts(tmp_path,monkeypatch):
    monkeypatch.setattr(cleanup,'ROOT',tmp_path)
    base=tmp_path/'results/own';base.mkdir(parents=True)
    monkeypatch.setattr(cleanup,'BASE',base)
    for name in ('guards/budget.json','guards/campaign-failure.json','audits/model.json'):
        path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'{}')
        assert not cleanup.owned_candidate(path,pin(path))[0]
