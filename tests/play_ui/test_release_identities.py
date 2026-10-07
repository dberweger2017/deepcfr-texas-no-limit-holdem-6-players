"""Published manifest bytes remain pinned and bound to the loaded model."""

from dataclasses import replace

import pytest

from src.play_api.versions import RELEASES
from src.policies.v041 import MODEL_SHA256
from src.policies.v040 import EXPECTED_SHA256 as R_SHA256
from tests.play_ui.test_service import FixturePolicy


@pytest.mark.parametrize('version,sha,format_id', [
    ('v0.4.0', R_SHA256, 'holdem-hu20-native-reopening-blueprint-v1'),
    ('v0.4.1', MODEL_SHA256, 'holdem-hu20-stored-cfr-average-diagnostic-v1'),
])
def test_manifest_pin_binding_and_corruption(version, sha, format_id):
    policy = FixturePolicy()
    policy.spec = type('Identity', (), {'sha256': sha})()
    policy.format_id = format_id
    release = next(item for item in RELEASES if item.version == version)
    identity = release.identity(policy)
    assert identity['manifestSha256'] == release.manifest_sha256
    assert identity['version'] == version and identity['sha256'] == sha
    policy.spec = type('Identity', (), {'sha256': 'e' * 64})()
    with pytest.raises(ValueError, match='differs from loaded model'):
        release.identity(policy)
    release = replace(release, manifest_sha256='0' * 64)
    with pytest.raises(ValueError, match='manifest hash differs'):
        release.identity(policy)
