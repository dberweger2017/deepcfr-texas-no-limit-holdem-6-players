"""Release loading builds once, maps read-only, and refuses bad local caches."""
import json
import shutil

import numpy as np
import pytest

from src.blueprint import policy_cache
from src.blueprint.abstraction import HU200_SCHEMA
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy
from src.blueprint.compact_policy import CompactBuilder
from src.policies import hu200_release as release
from src.policies.files import file_hash
from tests.test_policy_cache import average, contents
from tests.test_streaming_average_cache import dimensions


@pytest.fixture
def small_release(tmp_path, monkeypatch):
    model, sha, views = average(tmp_path)
    baseline = AveragePolicy(model, sha, expected_schema=HU200_SCHEMA, translation=TranslationOptions())
    bundle = tmp_path / 'bundle'; bundle.mkdir()
    assembled = tmp_path / 'bundle-model'; assembled.mkdir()
    shutil.copyfile(model, assembled / model.name)
    n, slots = dimensions(model)
    pin = {'file': model.name, 'sha256': sha, 'entries': n, 'seed': 123, 'iteration': 2,
           'checkpoint_sha256': baseline.description['source_checkpoint_sha256']}
    monkeypatch.setattr(release, 'MODELS', {'blend': pin})
    monkeypatch.setattr(release, 'ACTION_SLOTS', slots)
    monkeypatch.setattr(release, 'SCHEMA', HU200_SCHEMA)
    monkeypatch.setattr(release, 'EXTRACTIONS', {'opponent-sampled': baseline.description['strategy']})
    monkeypatch.setattr(release, 'INFERENCE', {'adapter': 'cache-test',
                        'translation': baseline.description['action_translation']})
    # This three-row legacy-card fixture tests real cache I/O; the full-size
    # release smoke exercises the pinned K50 tables and admission checks.
    def load_average(*args, **kwargs):
        kwargs.pop('card_buckets')
        return AveragePolicy(*args, **kwargs)
    monkeypatch.setattr(release, 'AveragePolicy', load_average)
    monkeypatch.setattr(release, 'verify_bundle', lambda *_a, **_k: {'release': 'v0.5.5'})
    def verify(path):
        if file_hash(path) != sha:
            raise ValueError('Pinned model differs')
    monkeypatch.setattr(release, 'verify_model', verify)
    return bundle, tmp_path / 'bundle-cache', pin, baseline, views


def test_first_start_builds_and_every_start_maps_verified_arrays(small_release, monkeypatch):
    bundle, cache, pin, baseline, views = small_release
    _, first = release.load_policy(bundle)
    assert contents(first, views) == contents(baseline, views)
    assert isinstance(first.entries.probabilities, np.memmap)
    assert not first.entries.probabilities.flags.writeable
    monkeypatch.setattr(release, 'build_cache', lambda *_: pytest.fail('Rebuilt an existing cache'))
    monkeypatch.setattr(CompactBuilder, 'add', lambda *_: pytest.fail('Parsed rows on a cached start'))
    _, second = release.load_policy(bundle)
    assert contents(second, views) == contents(baseline, views)
    assert not second.entries.probabilities.flags.writeable
    assert policy_cache.folder(cache, pin['sha256'], HU200_SCHEMA).is_dir()


@pytest.mark.parametrize('damage', ['array', 'identity', 'missing-file', 'symlink'])
def test_bad_cache_is_preserved_and_reported(small_release, damage):
    bundle, cache, pin, _, _ = small_release
    release.load_policy(bundle)
    target = policy_cache.folder(cache, pin['sha256'], HU200_SCHEMA)
    if damage == 'array':
        path = target / 'probabilities.npy'
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    elif damage == 'identity':
        path = target / 'manifest.json'; data = json.loads(path.read_text())
        data['source_sha256'] = '0' * 64; path.write_text(json.dumps(data))
    elif damage == 'missing-file':
        (target / 'keys.npy').unlink()
    else:
        path = target / 'flags.npy'; path.rename(target / 'saved-flags.npy')
        path.symlink_to('saved-flags.npy')
    before = {p.name: p.lstat().st_mtime_ns for p in target.iterdir()}
    with pytest.raises(ValueError, match='HU200 play cache refused.*move it aside'):
        release.load_policy(bundle)
    assert {p.name: p.lstat().st_mtime_ns for p in target.iterdir()} == before


def test_model_checked_before_existing_cache(small_release):
    bundle, _, _, _, _ = small_release
    release.load_policy(bundle)
    (bundle.parent / 'bundle-model/average.gz').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='Pinned model differs'):
        release.load_policy(bundle)
