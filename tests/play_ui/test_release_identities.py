"""Published manifest bytes remain pinned and bound to the loaded model."""

from pathlib import Path

import pytest

from src.play_api import releases
from src.play_api.o_candidate import MODEL_SHA256
from src.play_api.service import MODEL_SHA256 as R_SHA256
from tests.play_ui.test_service import FixturePolicy


@pytest.mark.parametrize('version,sha,format_id', [
    ('v0.4.0', R_SHA256, 'holdem-hu20-native-reopening-blueprint-v1'),
    ('v0.4.1', MODEL_SHA256, 'holdem-hu20-stored-cfr-average-diagnostic-v1'),
])
def test_manifest_pin_binding_and_corruption(tmp_path, monkeypatch, version, sha, format_id):
    policy = FixturePolicy()
    policy.spec = type('Identity', (), {'sha256': sha})()
    policy.format_id = format_id
    identity = releases.release_identity(version, policy)
    assert identity['manifestSha256'] == releases.MANIFEST_SHA256[version]
    assert identity['version'] == version and identity['sha256'] == sha
    policy.spec = type('Identity', (), {'sha256': 'e' * 64})()
    with pytest.raises(ValueError, match='differs from loaded model'):
        releases.release_identity(version, policy)
    data = (releases.MANIFESTS / f'{version}.json').read_bytes()
    (tmp_path / f'{version}.json').write_bytes(data + b'\n')
    monkeypatch.setattr(releases, 'MANIFESTS', tmp_path)
    with pytest.raises(ValueError, match='manifest hash differs'):
        releases.release_identity(version, policy)
