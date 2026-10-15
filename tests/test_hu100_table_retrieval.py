"""Restoration checks the whole archive, member manifest and pinned table bytes."""
import io
import json
import tarfile
import pytest
from scripts import retrieve_hu100_tables as restore


def test_restoration_and_rehashed_wrong_member(tmp_path, monkeypatch):
    pins={}; members=[]
    for name in ('flop-k50.bin','turn-k50.bin','river-k50.bin'):
        source=tmp_path/name; source.write_bytes(('fixture '+name).encode())
        pin=dict(bytes=source.stat().st_size,sha256=restore.sha(source))
        pins[name]=pin; members.append(dict(path=name,**pin))
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(dict(members=members)))
    monkeypatch.setattr(restore,'CARD_TABLES',pins)
    monkeypatch.setattr(restore,'MANIFEST_SHA256',restore.sha(manifest))
    for wrong in (False,True):
        archive=tmp_path/f'archive-{wrong}.tar.gz'
        with tarfile.open(archive,'w:gz') as bundle:
            for name in pins:
                data=(tmp_path/name).read_bytes()
                if wrong and name=='river-k50.bin':
                    data=data[:-1]+b'X'
                info=tarfile.TarInfo(restore.PREFIX+name);info.size=len(data)
                bundle.addfile(info,io.BytesIO(data))
        monkeypatch.setattr(restore,'ARCHIVE_BYTES',archive.stat().st_size)
        monkeypatch.setattr(restore,'ARCHIVE_SHA256',restore.sha(archive))
        if wrong:
            with pytest.raises(ValueError,match='Exactness mismatch'):
                restore.retrieve(archive,manifest,tmp_path/'wrong')
        else:
            result=restore.retrieve(archive,manifest,tmp_path/'restored')
            assert result['status']=='verified' and len(result['members'])==3
            with pytest.raises(ValueError,match='Preserve'):
                restore.retrieve(archive,manifest,tmp_path/'restored')
