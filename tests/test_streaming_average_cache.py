"""A bounded cache builder must preserve the existing loader's exact bytes."""
import gzip
import json

import pytest

from scripts.build_streaming_average_cache import build
from src.blueprint import policy_cache
from src.blueprint.abstraction import HU200_SCHEMA
from src.blueprint.average import AveragePolicy
from tests.test_policy_cache import average, contents


def dimensions(path):
    with gzip.open(path, 'rt') as source:
        next(source)
        rows = list(map(json.loads, source))
    return len(rows), sum(len(row[1]) for row in rows)


def test_streaming_files_and_play_match_the_existing_cache(tmp_path):
    path, sha, views = average(tmp_path)
    normal = tmp_path/'normal'
    streamed = tmp_path/'streamed'
    expected = AveragePolicy(path, sha, expected_schema=HU200_SCHEMA, cache=normal, sorted_rows=True)
    n, slots = dimensions(path)
    manifest = build(path, sha, HU200_SCHEMA, streamed, n, slots)
    for name in (*policy_cache.ARRAYS, 'manifest'):
        suffix = '.json' if name == 'manifest' else '.npy'
        assert ((policy_cache.folder(normal, sha, HU200_SCHEMA)/(name+suffix)).read_bytes()
                == (policy_cache.folder(streamed, sha, HU200_SCHEMA)/(name+suffix)).read_bytes())
    actual = AveragePolicy(path, sha, expected_schema=HU200_SCHEMA, cache=streamed, cache_read_only=True)
    assert contents(expected, views) == contents(actual, views)
    assert sum(p['bytes'] for p in manifest['files'].values()) == 35*n+8*slots+768


def test_bad_dimensions_preserve_a_partial_and_never_publish(tmp_path):
    path, sha, _ = average(tmp_path)
    n, slots = dimensions(path)
    with pytest.raises(ValueError, match='counts differ'):
        build(path, sha, HU200_SCHEMA, tmp_path/'bad', n+1, slots)
    target = policy_cache.folder(tmp_path/'bad', sha, HU200_SCHEMA)
    assert not target.exists()
    assert target.with_name(target.name+'.streaming-partial').exists()


def test_existing_cache_and_synced_paths_are_refused(tmp_path, monkeypatch):
    path, sha, _ = average(tmp_path)
    n, slots = dimensions(path)
    root = tmp_path/'cache'
    build(path, sha, HU200_SCHEMA, root, n, slots)
    with pytest.raises(FileExistsError):
        build(path, sha, HU200_SCHEMA, root, n, slots)
    with pytest.raises(ValueError, match='outside synced'):
        build(path, sha, HU200_SCHEMA, tmp_path/'CloudStorage'/'bad', n, slots)


def test_build_works_without_research_dependencies(tmp_path, monkeypatch):
    import builtins
    path, sha, _ = average(tmp_path)
    n, slots = dimensions(path)
    original = builtins.__import__
    def runtime_only(name, *args, **kwargs):
        if name.startswith(('scipy', 'scripts.evaluate_hu100_direct')):
            raise AssertionError('Play cache imported research dependencies')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', runtime_only)
    build(path, sha, HU200_SCHEMA, tmp_path/'runtime-only', n, slots)
