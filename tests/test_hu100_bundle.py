"""HU100 release bundles: exact bytes, provenance, translation and publication binding."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import build_hu100_bundle as builder
from src.policies import hu100_bundle as verifier

RELEASE_NAMES = sorted(verifier.RELEASES)


@pytest.fixture(params=RELEASE_NAMES)
def prepared(request, tmp_path, monkeypatch):
    """A release whose pinned model is a tiny file with that release's exact header."""
    release = request.param
    monkeypatch.setattr(builder, 'check_source', lambda *_: None)
    model = dict(verifier.RELEASES[release]['model'])
    header = {'format': model['format'], 'kind': 'diagnostic-inference',
              'extraction': model['extraction'], 'source_checkpoint_sha256': model['checkpoint_sha256'],
              'checkpoint_header': {'config': {'seed': model['seed'], 'game': model['game'], 'raise_cap': None},
                                    'abstraction': model['schema'], 'iteration': model['iteration'],
                                    'average_rule': 'opponent-sampled', 'identity': {'players': 2}}}
    source = tmp_path / 'small.gz'
    source.write_bytes(gzip.compress((json.dumps(header) + '\n').encode(), mtime=0))
    model.update(bytes=source.stat().st_size, sha256=verifier.sha(source))
    monkeypatch.setitem(verifier.RELEASES, release, {**verifier.RELEASES[release], 'model': model})
    directory = tmp_path / 'package'
    builder.build(release, source, directory, 'a' * 40)
    return release, source, directory


def rehash(directory, release):
    (directory / 'SHA256SUMS').write_text(''.join(
        f'{verifier.sha(directory / name)}  {name}\n' for name in sorted(verifier.assets(release))))


def test_exact_bytes_source_and_preservation(prepared):
    release, source, directory = prepared
    model = verifier.RELEASES[release]['model']['file']
    assert source.read_bytes() == (directory / model).read_bytes()
    assert (directory / verifier.RELEASES[release]['verifier']).is_file()
    result = verifier.verify_bundle(directory, 'a' * 40)
    assert (result['status'], result['release']) == ('unpublished-owner-review', release)
    with pytest.raises(ValueError, match='source'):
        verifier.verify_bundle(directory, 'b' * 40)
    with pytest.raises(ValueError, match='Publication'):
        verifier.verify_bundle(directory, 'a' * 40, require_publication=True)
    with pytest.raises(ValueError, match='Preserve'):
        builder.build(release, source, directory, 'a' * 40)


def test_approved_bundle_binds_its_own_source_and_tag(prepared, tmp_path):
    release, source, _ = prepared
    directory = tmp_path / 'approved'
    builder.build(release, source, directory, 'c' * 40, publication_approved=True)
    data = json.loads((directory / 'release-manifest.json').read_text())
    assert (data['status'], data['release_tag'], data['approved_release_source_commit']) == (
        'owner-approved-publication', release, 'c' * 40)
    assert verifier.verify_bundle(directory, 'c' * 40, require_publication=True)['status'] == 'owner-approved-publication'
    with pytest.raises(ValueError, match='Publication'):
        verifier.verify_bundle(directory, require_publication=True)
    with pytest.raises(ValueError, match='source'):
        verifier.verify_bundle(directory, 'a' * 40, require_publication=True)
    # Another source cannot be named in an approved manifest, even rehashed.
    data['approved_release_source_commit'] = 'a' * 40
    (directory / 'release-manifest.json').write_text(json.dumps(data)); rehash(directory, release)
    with pytest.raises(ValueError, match='identity'):
        verifier.verify_bundle(directory, 'c' * 40)


@pytest.mark.parametrize('change', [
    {'owner_publication_approval': True}, {'owner_publication_approval': 0},
    {'release_tag': 'v0.5.0'}, {'status': 'published'},
    {'inference': {'adapter': 'direct-v1', 'translation': None}},
    {'model': {'seed': 2026100902}}, {'provenance': {}},
    {'package_source_commit': 'bad'}, {'evidence': {'external_benchmark_established': True}},
])
def test_rehashed_manifest_cannot_waive_pins_or_publication_hold(prepared, change):
    release, _, directory = prepared
    path = directory / 'release-manifest.json'
    data = json.loads(path.read_text()); data.update(change)
    path.write_text(json.dumps(data)); rehash(directory, release)
    with pytest.raises(ValueError):
        verifier.verify_bundle(directory)


def test_a_bundle_cannot_claim_another_release(prepared):
    release, _, directory = prepared
    other = next(r for r in RELEASE_NAMES if r != release)
    path = directory / 'release-manifest.json'
    data = json.loads(path.read_text()); data['release'] = other
    path.write_text(json.dumps(data)); rehash(directory, release)
    with pytest.raises(ValueError):
        verifier.verify_bundle(directory)


def test_corruption_duplicate_missing_symlink(prepared):
    release, _, directory = prepared
    sums = directory / 'SHA256SUMS'; old = sums.read_text()
    sums.write_text(old + old.splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='duplicate'): verifier.verify_bundle(directory)
    sums.write_text('\n'.join(old.splitlines()[:-1]))
    with pytest.raises(ValueError, match='Missing'): verifier.verify_bundle(directory)
    sums.write_text(old)
    model = directory / verifier.RELEASES[release]['model']['file']
    model.write_bytes(model.read_bytes() + b'X'); rehash(directory, release)
    with pytest.raises(ValueError, match='fixed export'): verifier.verify_bundle(directory)
    model.unlink(); model.symlink_to(directory / 'MODEL_CARD.md')
    with pytest.raises(ValueError, match='regular'): verifier.verify_bundle(directory)


def test_published_v050_manifest_still_verifies():
    """The rebuilt v0.5.0 manifest equals the published one, so its bundle keeps verifying."""
    source = 'b9c9bd161422e87371a2641ae4c60304c66eed79'
    data = verifier.manifest('v0.5.0', source, True)
    assert hashlib_sha(json.dumps(data, indent=2, sort_keys=True) + '\n') == \
        '9a7a7209087e5e8ec7d93ab06f382233777cefcbbe0e905f48ea845468cee729'
    assert verifier.assets('v0.5.0') >= {'verify_v050_bundle.py', 'release-manifest.json'}


def hashlib_sha(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize('release', RELEASE_NAMES)
def test_runtime_release_selection_pins_translation_and_keeps_hu20_catalog(tmp_path, monkeypatch, release):
    from tests.play_ui.test_hu100_research import fixture_policy
    from src.policies import hu100_release
    from src.play_api.server import single_table
    from src.play_api.configuration import inference_record, PlayTable
    from src.play_api.service import PlayService, PlayError
    from src.play_api.versions import DEFAULT_VERSION, RELEASES
    policy = fixture_policy(tmp_path, True)
    monkeypatch.setattr(hu100_release, 'verify_bundle', lambda _: {'release': release})
    monkeypatch.setattr(hu100_release, 'load_pinned', lambda *a, **k: policy)
    args = SimpleNamespace(hu100_release=tmp_path, hu100_research=None,
                           data_dir=tmp_path / 'data', source_version='test')
    runtime = single_table(args)
    try:
        assert runtime.model_catalog()['default'] == release
        assert inference_record(policy) == verifier.INFERENCE
        assert runtime.model_info()['name'] == verifier.RELEASES[release]['name']
        assert runtime.model_info()['research'] is (release == 'v0.5.0')
        assert (tmp_path / 'data' / release / 'private.sqlite').exists()
        assert DEFAULT_VERSION == 'v0.4.2'
        assert [r.version for r in RELEASES] == ['v0.4.2', 'v0.4.1', 'v0.4.0']
        with pytest.raises(ValueError, match='incompatible'):
            PlayService(tmp_path / 'wrong.sqlite', policy, table=PlayTable())
        with pytest.raises(PlayError):
            runtime.create('release-wrong-001', {'modelVersion': 'v0.4.2'})
        policy.configure_translation(None)
        with pytest.raises(ValueError, match='translation'):
            hu100_release.load_policy(tmp_path)
    finally:
        runtime.close()


def test_cli_rejects_depth_override_and_optional_translation():
    import subprocess
    import sys
    for args in (['--stack-bb', '20'], ['--translate-off-menu'], ['--stack-bb', '200']):
        result = subprocess.run([sys.executable, '-m', 'src.play_api.server',
                                 '--hu100-release', 'missing', *args], capture_output=True, text=True)
        assert result.returncode == 2
        assert 'Traceback' not in result.stderr


def test_smoke_driver_http_and_independent_audit_with_small_fixture(tmp_path, monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer
    from tests.play_ui.test_hu100_research import fixture_policy
    from src.policies import hu100_release
    from src.play_api.server import handler_for
    from src.play_api.service import _model_info, PlayService
    from src.play_api.spectator import SpectatorService
    from src.play_api.versions import VersionedTables
    from scripts.smoke_hu100_release import Runtime, exercise, audit
    release = 'v0.5.1'
    policy = fixture_policy(tmp_path, True)
    (tmp_path / 'release-manifest.json').write_text(json.dumps({'release': release}))
    data = tmp_path / 'data'; out = tmp_path / 'out'; out.mkdir()
    base = data / release
    identity = {'version': release, **_model_info(policy)}
    human = PlayService(base / 'private.sqlite', policy)
    spectator = SpectatorService(base / 'spectator.sqlite', {release: policy}, {release: identity})
    tables = VersionedTables({release: human}, default_version=release,
                             spectator=spectator, identities={release: identity})
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(tables, 'fixture-token', 0))
    server.RequestHandlerClass = handler_for(tables, 'fixture-token', server.server_port)
    thread = threading.Thread(target=server.serve_forever); thread.start()
    runtime = Runtime(tmp_path, data, out, 'fixture')
    runtime.process = SimpleNamespace(pid=123)
    runtime.port = server.server_port; runtime.token = 'fixture-token'
    try:
        result = exercise(runtime, {'restricted': 1, 'free': 1, 'spectator': 1}, 'fixture')
        assert result['hands'] == 3 and result['off_menu_550_raises'] == 1
    finally:
        server.shutdown(); thread.join(); server.server_close(); tables.close()
    monkeypatch.setattr(hu100_release, 'load_policy', lambda _: (release, policy))
    result = audit(data, tmp_path, out)
    assert result['counts'] == {'free': 1, 'restricted': 1, 'spectator': 1}


def test_archive_retrieval_checks_archive_manifest_and_member(prepared, tmp_path, monkeypatch):
    import hashlib
    from zipfile import ZipFile
    from scripts import retrieve_hu100_model as retriever
    release, source, _ = prepared
    pins = verifier.RELEASES[release]
    member = pins['provenance']['member']
    manifest = json.dumps({'members': [{'path': member, 'bytes': source.stat().st_size,
                                        'sha256': verifier.sha(source)}]}).encode()
    archive = tmp_path / 'archive.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr('ARCHIVE-MANIFEST.json', manifest); z.write(source, member)
    provenance = {**pins['provenance'], 'archive_bytes': archive.stat().st_size,
                  'archive_sha256': verifier.sha(archive), 'manifest_sha256': hashlib.sha256(manifest).hexdigest()}
    monkeypatch.setitem(verifier.RELEASES, release, {**pins, 'provenance': provenance})
    monkeypatch.setattr(retriever.shutil, 'disk_usage', lambda _: SimpleNamespace(free=20 * 1024**3))
    result = retriever.retrieve(release, archive, tmp_path / 'restored')
    assert (tmp_path / 'restored' / pins['model']['file']).read_bytes() == source.read_bytes()
    assert result['release'] == release
    with pytest.raises(ValueError, match='Preserve'):
        retriever.retrieve(release, archive, tmp_path / 'restored')
    monkeypatch.setattr(retriever.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match='Insufficient'):
        retriever.retrieve(release, archive, tmp_path / 'no-space')


def test_browser_completion_waits_for_complete_remote_copy(tmp_path, monkeypatch):
    from scripts import smoke_hu100_release as smoke
    path = tmp_path / 'browser-done.json'; path.write_text('')
    monkeypatch.setattr(smoke, 'sleep', lambda _: path.write_text('{"complete_hands":2}'))
    assert smoke.browser_receipt(path, 1) == {'complete_hands': 2}
    path.write_text('{partial')
    with pytest.raises(TimeoutError, match='Browser allowance'):
        smoke.browser_receipt(path, 0)
