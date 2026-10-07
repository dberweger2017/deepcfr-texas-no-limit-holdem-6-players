"""Small, byte-pinned release manifests paired with the existing model loaders."""

import hashlib
import json
from pathlib import Path

from src.play_api.service import _model_info

MANIFESTS = Path(__file__).resolve().parents[2] / 'configs/play/release-manifests'
MANIFEST_SHA256 = {
    'v0.4.0': '1383de5fe5f60ef829ad0410e75bd871a936e5297c8030f534f69eb69c2362af',
    'v0.4.1': '8d1a85bea7fd2bad3d4a8526ad95e858239fa162d14e19369a47c739097659f5',
}
RELEASE_BASE = 'https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/download'


def release_identity(version: str, policy) -> dict:
    data = (MANIFESTS / f'{version}.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256[version]:
        raise ValueError('Pinned release manifest hash differs')
    manifest = json.loads(data)
    model = manifest['model']
    info = _model_info(policy)
    if (manifest.get('release', manifest.get('candidate')) != version
            or any(model[key] != info[key] for key in ('sha256', 'game', 'schema', 'format'))
            or model['players'] != 2 or model['raise_cap'] is not None):
        raise ValueError('Release manifest differs from loaded model')
    if not info['name'].startswith(version + ' · '):
        info['name'] = f"{version} · {info['name']}"
    return {'version': version, **info, 'manifestSha256': MANIFEST_SHA256[version],
            'manifestUrl': f'{RELEASE_BASE}/{version}/release-manifest.json'}
