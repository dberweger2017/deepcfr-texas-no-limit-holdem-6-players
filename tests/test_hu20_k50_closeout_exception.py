"""The owner exception is restricted to the exact metadata-only startup failure."""
import json
from pathlib import Path
import pytest
from scripts.guard_hu20_k50_first_pass import approved_archive_repair, file_sha, record_failure


def setup(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    out=Path('guards')
    folder=out/'operations/archive-closeout-sidecar';folder.mkdir(parents=True)
    worker=Path('planning/seal-archive-closeout.py');worker.parent.mkdir();worker.write_text('bootstrap corrected\n')
    (out/'campaign-failure.json').write_text(json.dumps({'operation':'archive-closeout-sidecar','failure':"RuntimeError('Worker exit 1')"}))
    (out/'budget.json').write_text(json.dumps({'started':1,'deadline':21601,'swap_bytes':123}))
    (folder/'receipt.json').write_text(json.dumps({'status':'failed','returncode':1}))
    (folder/'worker.log').write_text("ModuleNotFoundError: No module named 'scripts'\n")
    command=[str(tmp_path/'.venv/bin/python'),str(tmp_path/worker)]
    approval=Path('planning/approval.json')
    data={'owner_statement':'Exception permitted','authorized_operation':'archive-closeout-startup-repair',
          'command':command,'original_failure_sha256':file_sha(out/'campaign-failure.json'),
          'original_budget_sha256':file_sha(out/'budget.json'),'corrected_worker_sha256':file_sha(worker)}
    approval.write_text(json.dumps(data))
    return out,command,approval


def test_exact_archival_startup_exception_preserves_original_latch(tmp_path,monkeypatch):
    out,command,approval=setup(tmp_path,monkeypatch)
    before=(out/'campaign-failure.json').read_bytes()
    assert approved_archive_repair(out,'archive-closeout-startup-repair',command,approval)==file_sha(approval)
    folder=out/'operations/new-repair';folder.mkdir()
    record_failure(out,folder,'new-repair','another failure')
    assert (out/'campaign-failure.json').read_bytes()==before
    assert json.loads((folder/'failure.json').read_text())['failure']=='another failure'


def test_exception_cannot_resume_science_or_change_fixed_budget(tmp_path,monkeypatch):
    out,command,approval=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='exact archival'):
        approved_archive_repair(out,'final-direct-play',command,approval)
    with pytest.raises(ValueError,match='exact archival'):
        approved_archive_repair(out,'archive-closeout-startup-repair',[command[0],'-m','scripts.run_hu20_k50_first_pass_training'],approval)
    (out/'budget.json').write_text('changed')
    with pytest.raises(ValueError,match='exact archival'):
        approved_archive_repair(out,'archive-closeout-startup-repair',command,approval)


def test_no_exception_or_different_failure_stays_terminal(tmp_path,monkeypatch):
    out,command,approval=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='Terminal'):
        approved_archive_repair(out,'archive-closeout-startup-repair',command,None)
    (out/'operations/archive-closeout-sidecar/worker.log').write_text('Guard breach: memory pressure')
    with pytest.raises(ValueError,match='exact archival'):
        approved_archive_repair(out,'archive-closeout-startup-repair',command,approval)


def test_inventory_correction_is_limited_to_known_metadata_path_failure(tmp_path,monkeypatch):
    out,command,approval=setup(tmp_path,monkeypatch)
    prior=out/'operations/archive-closeout-startup-repair';prior.mkdir()
    (prior/'receipt.json').write_text(json.dumps({'status':'failed','returncode':1}))
    expected="FileNotFoundError: [Errno 2] No such file or directory: 'results/hu20-k50-5b/campaign-failure.json'"
    (prior/'worker.log').write_text(expected)
    data=json.loads(approval.read_text())
    data.update(authorized_operation='archive-closeout-corrected-inventory',
                prior_metadata_failure_receipt_sha256=file_sha(prior/'receipt.json'))
    approval.write_text(json.dumps(data))
    assert approved_archive_repair(out,'archive-closeout-corrected-inventory',command,approval)==file_sha(approval)
    (prior/'worker.log').write_text('Nonfinite scientific result')
    with pytest.raises(ValueError,match='different metadata failure'):
        approved_archive_repair(out,'archive-closeout-corrected-inventory',command,approval)


def test_cleanup_needs_broad_owner_scope_and_completed_closeout(tmp_path,monkeypatch):
    out,command,approval=setup(tmp_path,monkeypatch)
    (tmp_path/'final-readout.json').write_text(json.dumps({'status':'verified'}))
    closed=out/'operations/archive-closeout-corrected-inventory';closed.mkdir()
    (closed/'receipt.json').write_text(json.dumps({'status':'complete'}))
    worker=tmp_path/'scripts/cleanup_hu20_k50_first_pass.py';worker.parent.mkdir();worker.write_text('cleanup reviewed\n')
    command=[str(tmp_path/'.venv/bin/python'),str(worker),'execute']
    data=json.loads(approval.read_text())
    data.update(authorized_operation='cleanup-owned-originals',command=command,
                corrected_worker_sha256=file_sha(worker))
    approval.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='broad owner'):
        approved_archive_repair(out,'cleanup-owned-originals',command,approval)
    data['owner_statement']='everything falls under the exception, finish everything needed to close up the pr and do a correct cleanup'
    approval.write_text(json.dumps(data))
    assert approved_archive_repair(out,'cleanup-owned-originals',command,approval)==file_sha(approval)
    (closed/'receipt.json').write_text(json.dumps({'status':'failed'}))
    with pytest.raises(ValueError,match='broad owner'):
        approved_archive_repair(out,'cleanup-owned-originals',command,approval)
