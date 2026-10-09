import hashlib
import json
from zipfile import ZipFile

import pytest

from scripts.restore_hu100_staged_archive import restore, safe_member
from src.policies.files import file_hash


def test_restore_selected_alias_from_unique_canonical_bytes(tmp_path):
    data=b'exact model bytes'
    pin={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    manifest={'members':{'research/calibration/models/policy.gz':pin},'hardlink_aliases':{
        'research/training/seed/average.gz':{'canonical_member':'research/calibration/models/policy.gz',**pin}}}
    raw=json.dumps(manifest).encode()
    archive=tmp_path/'campaign.zip'
    with ZipFile(archive,'x') as z:
        z.writestr('ARCHIVE-MANIFEST.json',raw)
        z.writestr('research/calibration/models/policy.gz',data)
    destination=tmp_path/'restored'
    restore(archive,destination,file_hash(archive),hashlib.sha256(raw).hexdigest(),['research/training/seed/average.gz'])
    canonical=destination/'research/calibration/models/policy.gz'
    alias=destination/'research/training/seed/average.gz'
    assert alias.read_bytes()==data and alias.stat().st_ino==canonical.stat().st_ino
    with pytest.raises(FileExistsError):
        restore(archive,destination,file_hash(archive),hashlib.sha256(raw).hexdigest(),[])
    with pytest.raises(ValueError):restore(archive,tmp_path/'wrong','0'*64,hashlib.sha256(raw).hexdigest(),[])
    for name in ('/research/absolute','research/../escape','other/file'):
        with pytest.raises(ValueError):safe_member(name)
