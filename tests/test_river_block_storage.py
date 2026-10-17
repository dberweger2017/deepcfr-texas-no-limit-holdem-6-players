"""Destructive fault cases use owned copies, never research originals."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from src.arena.river_block_storage import Journal, MANIFEST, canonical, receive, retire, seal


def fixture(tmp_path):
    source=tmp_path/'owned';source.mkdir();(source/'nested').mkdir()
    (source/'nested/policy.json').write_text('{"complete":true}\n')
    (source/'block.json').write_text('{"all_hands_verified":true}\n')
    identity=dict(panel='fixture',block=0,declaration_sha256='fixture')
    manifest=seal(source,identity);sha=hashlib.sha256((canonical(manifest)+'\n').encode()).hexdigest()
    destination=tmp_path/'arrival';shutil.copytree(source,destination)
    journal=Journal(tmp_path/'receipts.sqlite');journal.prepare(identity,manifest,dict(fixture=True))
    return source,destination,journal,manifest,sha


def test_verified_arrival_then_exact_retirement(tmp_path):
    source,remote,journal,manifest,sha=fixture(tmp_path)
    arrival=receive(remote,sha);journal.update('fixture',0,'verified',arrival=arrival)
    receipt=retire(source,manifest,arrival,journal)
    assert not source.exists() and (remote/'nested/policy.json').exists()
    assert receipt['retired_files']==sorted([*manifest['files'],MANIFEST])
    assert journal.get('fixture',0)['phase']=='retired'
    assert receive(remote,sha)==arrival


def test_corrupt_arrival_preserves_all_local_files(tmp_path):
    source,remote,journal,manifest,sha=fixture(tmp_path)
    (remote/'nested/policy.json').write_text('corrupt')
    with pytest.raises(ValueError,match='checksum/size'):
        receive(remote,sha)
    assert (source/'nested/policy.json').exists()
    with pytest.raises(ValueError,match='matching verified arrival'):
        retire(source,manifest,dict(status='unverified',manifest_sha256=sha,identity=manifest['identity']),journal)


def test_changed_local_file_blocks_all_retirement(tmp_path):
    source,remote,journal,manifest,sha=fixture(tmp_path)
    arrival=receive(remote,sha);journal.update('fixture',0,'verified',arrival=arrival)
    (source/'block.json').write_text('changed')
    with pytest.raises(ValueError):retire(source,manifest,arrival,journal)
    assert (source/'nested/policy.json').exists() and (source/'block.json').exists()


def test_crash_during_retirement_resumes_from_durable_arrival(tmp_path):
    source,remote,journal,manifest,sha=fixture(tmp_path)
    arrival=receive(remote,sha);journal.update('fixture',0,'verified',arrival=arrival)
    def crash(name):raise OSError('injected power interruption')
    with pytest.raises(OSError,match='interruption'):
        retire(source,manifest,arrival,journal,after_unlink=crash)
    assert journal.get('fixture',0)['phase']=='retiring'
    arrival=receive(remote,sha)
    result=retire(source,manifest,arrival,journal)
    assert len(result['already_absent_after_verified_interruption'])==1
    assert not source.exists() and journal.get('fixture',0)['phase']=='retired'


def test_missing_or_extra_unverified_files_never_retired(tmp_path):
    source,remote,journal,manifest,sha=fixture(tmp_path)
    arrival=receive(remote,sha);journal.update('fixture',0,'verified',arrival=arrival)
    (source/'unverified.txt').write_text('retain')
    with pytest.raises(ValueError):retire(source,manifest,arrival,journal)
    assert (source/'unverified.txt').exists() and (source/'nested/policy.json').exists()


def test_symlinks_and_unsafe_manifest_members_are_rejected(tmp_path):
    source,remote,journal,manifest,sha=fixture(tmp_path)
    outside=tmp_path/'protected';outside.write_text('original')
    (source/'external').symlink_to(outside)
    with pytest.raises(ValueError,match='nonregular'):seal(source,manifest['identity'])
    assert outside.read_text()=='original'
