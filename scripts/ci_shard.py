"""Partition all collected pytest items by file across isolated CI runners."""

import hashlib

import pytest


def shard_for(path, count):
    # The first CI timings put both long HU20 integrations in bucket 0:
    # 93.73s + 59.70s for their largest cases alone. Balance intact files.
    if count == 2 and path in (
        "tests/test_hu20_scaling.py", "tests/test_hu20_scaling_recovery.py"
    ):
        return 1
    return int.from_bytes(hashlib.sha256(path.encode()).digest()[:8], "big") % count


def pytest_addoption(parser):
    parser.addoption("--ci-shard", type=int, default=None)
    parser.addoption("--ci-shards", type=int, default=2)


def pytest_collection_modifyitems(config, items):
    shard = config.getoption("--ci-shard")
    if shard is None:
        return
    count = config.getoption("--ci-shards")
    if count < 1 or not 0 <= shard < count:
        raise pytest.UsageError("CI shard must be in [0, ci-shards)")
    selected, deselected = [], []
    for item in items:
        path = item.path.relative_to(config.rootpath).as_posix()
        (selected if shard_for(path, count) == shard else deselected).append(item)
    # Keep every test in a file together and preserve pytest's collection order.
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_report_header(config):
    shard = config.getoption("--ci-shard")
    if shard is not None:
        return f"CI file shard {shard + 1}/{config.getoption('--ci-shards')}"
