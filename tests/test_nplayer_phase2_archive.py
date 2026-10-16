"""Archive member and transport-part readbacks preserve exact bytes."""
import hashlib
import json
import zipfile
from scripts.archive_nplayer_phase2 import seal


def test_transport_parts_reassemble_the_verified_archive(tmp_path):
    root=tmp_path/'source';root.mkdir()
    (root/'payload.gz').write_bytes(bytes(range(256))*11)
    owner=tmp_path/'owner.json'
    owner.write_text(json.dumps({'number':292,'state':'OPEN','headRefName':'feature/nplayer-blueprint-phase2'}))
    out=tmp_path/'archive'
    receipt=seal(root,['payload.gz'],out,owner,part_bytes=1000)
    combined=b''.join((out/p['member']).read_bytes() for p in receipt['parts'])
    assert hashlib.sha256(combined).hexdigest()==receipt['sha256']
    assert all(p['bytes']<=1000 for p in receipt['parts'])
    with zipfile.ZipFile(out/'evidence.zip') as z:
        assert z.read('payload.gz')==(root/'payload.gz').read_bytes()
        manifest=json.loads(z.read('MANIFEST.json'))
        assert manifest['members'][0]['sha256']==hashlib.sha256(z.read('payload.gz')).hexdigest()

    from scripts.restore_nplayer_phase2_archive import restore
    restored=tmp_path/'restored'
    result=restore(out/'receipt.json',out,restored,['payload.gz'])
    assert result['status']=='passed'
    assert (restored/'payload.gz').read_bytes()==(root/'payload.gz').read_bytes()
    assert (out/'evidence.zip.part0000').exists()
