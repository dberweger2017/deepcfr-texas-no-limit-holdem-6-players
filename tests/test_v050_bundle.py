"""Fixed v0.5.0 bytes, provenance, translation and publication binding."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import build_v050_bundle as builder
from src.policies import v050_bundle as verifier


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, 'check_source', lambda _: None)
    model = dict(verifier.MODEL)
    header = {'format': model['format'], 'kind': 'diagnostic-inference',
              'extraction': model['extraction'], 'source_checkpoint_sha256': model['checkpoint_sha256'],
              'checkpoint_header': {'config': {'seed': model['seed'], 'game': model['game'], 'raise_cap': None},
                                    'abstraction': model['schema'], 'iteration': model['iteration'],
                                    'average_rule': 'opponent-sampled', 'identity': {'players': 2}}}
    source = tmp_path / 'small.gz'
    source.write_bytes(gzip.compress((json.dumps(header) + '\n').encode(), mtime=0))
    model.update(bytes=source.stat().st_size, sha256=verifier.sha(source))
    monkeypatch.setattr(verifier, 'MODEL', model)
    directory = tmp_path / 'package'
    builder.build(source, directory, 'a' * 40)
    return source, directory


def rehash(directory):
    (directory / 'SHA256SUMS').write_text(''.join(
        f'{verifier.sha(directory / name)}  {name}\n' for name in sorted(verifier.ASSETS)))


def test_exact_bytes_source_and_preservation(prepared):
    source, directory = prepared
    assert source.read_bytes() == (directory / verifier.ASSET_NAME).read_bytes()
    assert verifier.verify_bundle(directory, 'a' * 40)['status'] == 'unpublished-owner-review'
    with pytest.raises(ValueError, match='source'):
        verifier.verify_bundle(directory, 'b' * 40)
    with pytest.raises(ValueError, match='Publication'):
        verifier.verify_bundle(directory, 'a' * 40, require_publication=True)
    with pytest.raises(ValueError, match='Preserve'):
        builder.build(source, directory, 'a' * 40)


def test_approved_bundle_binds_its_own_source_and_tag(prepared, tmp_path):
    source, _ = prepared
    directory = tmp_path / 'approved'
    builder.build(source, directory, 'c' * 40, publication_approved=True)
    data = json.loads((directory / 'release-manifest.json').read_text())
    assert (data['status'], data['release_tag'], data['approved_release_source_commit']) == (
        'owner-approved-publication', 'v0.5.0', 'c' * 40)
    result = verifier.verify_bundle(directory, 'c' * 40, require_publication=True)
    assert result['status'] == 'owner-approved-publication'
    with pytest.raises(ValueError, match='Publication'):
        verifier.verify_bundle(directory, require_publication=True)
    with pytest.raises(ValueError, match='source'):
        verifier.verify_bundle(directory, 'a' * 40, require_publication=True)
    # Another source cannot be named in an approved manifest, even rehashed.
    data['approved_release_source_commit'] = 'a' * 40
    (directory / 'release-manifest.json').write_text(json.dumps(data)); rehash(directory)
    with pytest.raises(ValueError, match='identity'):
        verifier.verify_bundle(directory, 'c' * 40)


@pytest.mark.parametrize('change', [
    {'owner_publication_approval': True}, {'owner_publication_approval': 0},
    {'release_tag': 'v0.5.0'}, {'status': 'published'},
    {'inference': {'adapter': 'direct-v1', 'translation': None}},
    {'model': {'seed': 2026100902}}, {'provenance': {}},
    {'package_source_commit': 'bad'}, {'evidence': {'pr215_overall_recipe_qualified': True}},
])
def test_rehashed_manifest_cannot_waive_pins_or_publication_hold(prepared, change):
    _, directory = prepared
    path = directory / 'release-manifest.json'
    data = json.loads(path.read_text()); data.update(change)
    path.write_text(json.dumps(data)); rehash(directory)
    with pytest.raises(ValueError):
        verifier.verify_bundle(directory)


def test_corruption_duplicate_missing_symlink(prepared):
    _, directory = prepared
    sums = directory / 'SHA256SUMS'; old = sums.read_text()
    sums.write_text(old + old.splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='duplicate'): verifier.verify_bundle(directory)
    sums.write_text('\n'.join(old.splitlines()[:-1]))
    with pytest.raises(ValueError, match='Missing'): verifier.verify_bundle(directory)
    sums.write_text(old)
    model = directory / verifier.ASSET_NAME
    model.write_bytes(model.read_bytes() + b'X'); rehash(directory)
    with pytest.raises(ValueError, match='fixed PR207'): verifier.verify_bundle(directory)
    model.unlink(); model.symlink_to(directory / 'MODEL_CARD.md')
    with pytest.raises(ValueError, match='regular'): verifier.verify_bundle(directory)


def test_runtime_release_selection_pins_translation_and_keeps_hu20_catalog(tmp_path, monkeypatch):
    from tests.play_ui.test_hu100_research import fixture_policy
    from src.policies import v050
    from src.play_api.server import single_table
    from src.play_api.configuration import inference_record, PlayTable
    from src.play_api.service import PlayService, PlayError
    from src.play_api.versions import DEFAULT_VERSION, RELEASES
    policy = fixture_policy(tmp_path, True)
    monkeypatch.setattr(v050, 'verify_bundle', lambda _: None)
    monkeypatch.setattr(v050, 'load_research', lambda *a, **k: policy)
    args = SimpleNamespace(v050=tmp_path, hu100_research=None,
                           data_dir=tmp_path / 'data', source_version='test')
    runtime = single_table(args)
    try:
        assert runtime.model_catalog()['default'] == 'v0.5.0'
        assert inference_record(policy) == verifier.INFERENCE
        assert runtime.model_info()['name'] == v050.NAME
        assert DEFAULT_VERSION == 'v0.4.2'
        assert [r.version for r in RELEASES] == ['v0.4.2', 'v0.4.1', 'v0.4.0']
        with pytest.raises(ValueError, match='incompatible'):
            PlayService(tmp_path / 'wrong.sqlite', policy, table=PlayTable())
        with pytest.raises(PlayError):
            runtime.create('release-wrong-001', {'modelVersion': 'v0.4.2'})
        policy.configure_translation(None)
        with pytest.raises(ValueError, match='translation'):
            v050.load_policy(tmp_path)
    finally:
        runtime.close()


def test_cli_rejects_depth_override_and_optional_translation():
    import subprocess
    import sys
    for args in (['--stack-bb', '20'], ['--translate-off-menu'], ['--stack-bb', '200']):
        result = subprocess.run([sys.executable, '-m', 'src.play_api.server',
                                 '--v050', 'missing', *args], capture_output=True, text=True)
        assert result.returncode == 2
        assert 'Traceback' not in result.stderr


def test_smoke_driver_http_and_independent_audit_with_small_fixture(tmp_path, monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer
    from tests.play_ui.test_hu100_research import fixture_policy
    from src.policies import v050
    from src.play_api.server import handler_for
    from src.play_api.service import _model_info, PlayService
    from src.play_api.spectator import SpectatorService
    from src.play_api.versions import VersionedTables
    from scripts.smoke_v050_release import Runtime, exercise, audit
    policy = fixture_policy(tmp_path, True)
    data = tmp_path / 'data'; out = tmp_path / 'out'; out.mkdir()
    base = data / verifier.RELEASE
    identity = {'version': verifier.RELEASE, **_model_info(policy)}
    human = PlayService(base / 'private.sqlite', policy)
    spectator = SpectatorService(base / 'spectator.sqlite', {verifier.RELEASE: policy}, {verifier.RELEASE: identity})
    tables = VersionedTables({verifier.RELEASE: human}, default_version=verifier.RELEASE,
                             spectator=spectator, identities={verifier.RELEASE: identity})
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
    monkeypatch.setattr(v050, 'load_policy', lambda _: policy)
    result = audit(data, tmp_path, out)
    assert result['counts'] == {'free': 1, 'restricted': 1, 'spectator': 1}


def test_retained_retrieval_checks_input_and_copy_without_new_zip(prepared, tmp_path, monkeypatch):
    from scripts import retrieve_v050_model as retriever
    source, _ = prepared
    monkeypatch.setattr(retriever, 'RETAINED_SOURCE', source)
    monkeypatch.setattr(retriever, 'MODEL', verifier.MODEL)
    monkeypatch.setattr(retriever.shutil, 'disk_usage', lambda _: SimpleNamespace(free=20 * 1024**3))
    result = retriever.retrieve_retained(source, tmp_path / 'restored')
    assert (tmp_path / 'restored' / verifier.ASSET_NAME).read_bytes() == source.read_bytes()
    assert result['archive_audit'] == 'existing accepted PR207 receipt; no fresh whole-ZIP verification'
    with pytest.raises(ValueError, match='Preserve'):
        retriever.retrieve_retained(source, tmp_path / 'restored')
    other = tmp_path / 'other.gz'; other.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match='indexed'):
        retriever.retrieve_retained(other, tmp_path / 'wrong')
    monkeypatch.setattr(retriever.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match='Insufficient'):
        retriever.retrieve_retained(source, tmp_path / 'no-space')


def test_browser_completion_waits_for_complete_remote_copy(tmp_path, monkeypatch):
    from scripts import smoke_v050_release as smoke
    path = tmp_path / 'browser-done.json'; path.write_text('')
    monkeypatch.setattr(smoke, 'sleep', lambda _: path.write_text('{"complete_hands":2}'))
    assert smoke.browser_receipt(path, 1) == {'complete_hands': 2}
    path.write_text('{partial')
    with pytest.raises(TimeoutError, match='Browser allowance'):
        smoke.browser_receipt(path, 0)
