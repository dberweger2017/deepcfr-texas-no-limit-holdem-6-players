"""Storage-stop recovery keeps scientific bytes, limits and failure provenance."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import continue_hu200_capacity as c
from scripts import schedule_hu200_capacity as q
from scripts.run_hu200_capacity_training import pin
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.diagnostics.cfr_average import extract
from src.policies.files import file_hash
from tests.diagnostics.test_cfr_average import fixture


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def test_one_native_run_only_and_revalidation_precedes_storage(tmp_path):
    queue=c.Continuation.__new__(c.Continuation)
    queue.base,queue.tasks,queue.done=tmp_path,{},set()
    queue.build()
    assert len(queue.tasks)==7
    trains=[t for t in queue.tasks.values() if t['kind']=='train']
    assert len(trains)==1 and trains[0]['seed']==c.SEEDS[0]
    for seed in c.SEEDS:
        audit=queue.tasks[f'audit-{seed}-20000000000']
        move=queue.tasks[f'offload-{seed}-20000000000']
        assert queue.ready(audit)==(seed!=c.SEEDS[0])
        assert not queue.ready(move)
        queue.done.add(audit['name'])
        assert queue.ready(move)


@pytest.mark.parametrize('kind,expected',[('train',20_000_000_000),('audit',23_000_000_000),('restore',10_000_000_000),('retire',0)])
def test_existing_scheduler_output_reservations_unchanged(kind,expected):
    assert q.Queue.__new__(q.Queue).output_reservation(dict(kind=kind))==expected


def test_only_read_only_audits_and_storage_have_metadata_reservations():
    queue=c.Continuation.__new__(c.Continuation)
    for seed in c.SEEDS:
        assert queue.output_reservation(dict(kind='audit',seed=seed))==(23_000_000_000 if seed==c.SEEDS[0] else c.METADATA_RESERVE)
        assert queue.output_reservation(dict(kind='retire',seed=seed))==c.METADATA_RESERVE
        assert queue.output_reservation(dict(kind='train',seed=seed))==20_000_000_000


def test_downtime_credit_closes_once_without_resetting_prior_clock():
    resumed=c.STOPPED_EPOCH+4321.25
    expected=c.PRIOR_DEADLINE+4321.25
    assert c.adjusted_deadline(c.STOPPED_EPOCH,resumed)==expected
    assert c.adjusted_deadline(c.STOPPED_EPOCH,resumed)==expected
    with pytest.raises(ValueError):c.adjusted_deadline(c.STOPPED_EPOCH-1,resumed)
    with pytest.raises(ValueError):c.adjusted_deadline(c.STOPPED_EPOCH,c.STOPPED_EPOCH-1)
    with pytest.raises(ValueError):c.adjusted_deadline(c.STOPPED_EPOCH,c.PRIOR_DEADLINE)


def test_sealed_ledger_rejects_extra_credit(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'approval',lambda base:None)
    monkeypatch.setattr(c,'PINS',tmp_path/'pins')
    write(c.PINS/c.APPROVAL,dict(status='owner-approved'))
    resumed=c.STOPPED_EPOCH+3601.25
    value=dict(prior_ledger_sha256=c.PRIOR_LEDGER_SHA,approval_sha256=file_hash(c.PINS/c.APPROVAL),
        stopped_epoch=c.STOPPED_EPOCH,resumed_epoch=resumed,extension_seconds=3601.25,
        deadline_epoch=c.adjusted_deadline(c.STOPPED_EPOCH,resumed))
    path=c.phase(tmp_path)/'deadline.json';write(path,value)
    assert c.checked_ledger(tmp_path)==value
    value['extension_seconds']+=1;write(path,value)
    with pytest.raises(ValueError):c.checked_ledger(tmp_path)


def exports_fixture(tmp_path):
    folder=tmp_path/'model';folder.mkdir()
    _,_,checkpoint,current,spec=fixture(folder)
    average=folder/'average.gz';extract(checkpoint,spec,average)
    checkpoint.rename(folder/'checkpoint.gz');current.rename(folder/'checkpoint.current.gz');average.rename(folder/'checkpoint.average.gz')
    checkpoint=folder/'checkpoint.gz'
    rows=gzip.decompress(checkpoint.read_bytes()).decode().splitlines();header=json.loads(rows[0])
    native=dict(checkpoint_sha256=file_hash(checkpoint),iteration=header['iteration'],diagnostics=dict(entries=len(rows)-1))
    write(folder/'native-record.json',native);write(folder/'saved-bytes.json',dict(file=pin(checkpoint),header=header))
    (folder/'audit.stop').write_text('Preserve original failure\n')
    return folder


def test_reuses_exports_without_invoking_native_or_changing_any_bytes(monkeypatch,tmp_path):
    folder=exports_fixture(tmp_path);before={p.name:p.read_bytes() for p in folder.iterdir()}
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**kw:pytest.fail('Native export or training must not run'))
    proof=tmp_path/'fresh-proof'
    checked=c.validate_existing_exports(folder,proof,expected_schema=HU20_UNCAPPED_SCHEMA)
    assert checked['audit']['status']=='verified'
    assert (proof/'audit.json').read_bytes()==(folder/'audit.json').read_bytes()
    assert c.q.read(proof/'adopted-inputs.json')['export_invoked'] is False
    for name,raw in before.items():assert (folder/name).read_bytes()==raw


def test_reuse_does_not_accept_truncated_export_or_publish_success(tmp_path):
    folder=exports_fixture(tmp_path);current=folder/'checkpoint.current.gz';current.write_bytes(current.read_bytes()[:-8])
    with pytest.raises((EOFError,ValueError)):
        c.validate_existing_exports(folder,tmp_path/'proof',expected_schema=HU20_UNCAPPED_SCHEMA)
    assert not (folder/'audit.json').exists() and not (tmp_path/'proof/audit.json').exists()


def setup_move(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'mount_checked',lambda:None)
    monkeypatch.setattr(c,'HDD',tmp_path)
    monkeypatch.setattr(c.shutil,'disk_usage',lambda p:SimpleNamespace(free=100_000_000_000))
    source=tmp_path/'nvme/checkpoint.gz';source.parent.mkdir();source.write_bytes(b'exact research bytes\0\1')
    target=tmp_path/'hdd/checkpoint.gz';journal=tmp_path/'journal.jsonl'
    return source,target,pin(source),tmp_path/'stop.request',journal


def test_move_unlinks_only_after_complete_matching_independent_readback(monkeypatch,tmp_path):
    src,dst,expected,stop,journal=setup_move(monkeypatch,tmp_path)
    result=c.relocate_one(src,dst,expected,stop,journal)
    assert not src.exists() and dst.read_bytes()==b'exact research bytes\0\1'
    assert file_hash(dst)==result['sha256']==expected['sha256']
    assert dst.stat().st_mtime_ns==expected['mtime_ns']
    assert [json.loads(x)['state'] for x in journal.read_text().splitlines()]==['verified-before-unlink','relocated-source-unlinked']


def test_bad_source_sha_preserves_source_and_partial_destination(monkeypatch,tmp_path):
    src,dst,expected,stop,journal=setup_move(monkeypatch,tmp_path);expected['sha256']='0'*64
    with pytest.raises(ValueError,match='source SHA'):c.relocate_one(src,dst,expected,stop,journal)
    assert src.exists() and not dst.exists() and dst.with_name(dst.name+'.copying').exists()
    assert not journal.exists()


def test_changed_source_or_existing_destination_never_overwritten(monkeypatch,tmp_path):
    src,dst,expected,stop,journal=setup_move(monkeypatch,tmp_path)
    dst.parent.mkdir();dst.write_bytes(b'existing')
    with pytest.raises(ValueError):c.relocate_one(src,dst,expected,stop,journal)
    assert src.exists() and dst.read_bytes()==b'existing'


def test_storage_stop_preserves_source(monkeypatch,tmp_path):
    src,dst,expected,stop,journal=setup_move(monkeypatch,tmp_path);stop.write_text('stop')
    with pytest.raises(RuntimeError,match='stop requested'):c.relocate_one(src,dst,expected,stop,journal)
    assert src.exists() and not dst.exists()
