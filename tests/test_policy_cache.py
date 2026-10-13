"""The binary average cache: identical lookups to a parsed load, verified before every use."""

import numpy as np
import pytest

from src.blueprint import policy_cache
from src.blueprint.abstraction import HU200_SCHEMA, choices, information_key
from src.blueprint.artifact import save_training
from src.blueprint.average import AveragePolicy
from src.blueprint.compact_policy import CompactBuilder
from src.blueprint.solver import HU200_GAME, BlueprintTrainer, Node, PilotConfig
from src.diagnostics.cfr_average import extract
from src.game.hand import Hand, Table
from src.policies.files import file_hash


def average(tmp_path):
    """A small HU200 average: three first decisions, one of them without average mass."""
    table = Table(('player-0', 'player-1'), (20000, 20000))
    trainer = BlueprintTrainer(table, PilotConfig(seed=123, raise_cap=None, abstraction=HU200_SCHEMA, game=HU200_GAME))
    trainer.iteration = 2
    views = []
    for seed, mass in ((3, 1.0), (4, 0.0), (11, 2.0)):
        hand = Hand.start(table, hand_id=f'h{seed}', seed=seed)
        view = hand.observe(hand.actor)
        menu = choices(view, raise_cap=None, free_fold=False)
        trainer.nodes[information_key(view, menu, schema=HU200_SCHEMA)] = Node(
            tuple(c.name for c in menu), [float(i) for i in range(len(menu))],
            [mass * (i + 1) for i in range(len(menu))], 10 + seed)
        views.append(view)
    checkpoint, output = tmp_path / 'checkpoint.gz', tmp_path / 'average.gz'
    save_training(trainer, checkpoint)
    extract(checkpoint, {'seed': 123, 'iteration': 2, 'checkpoint_sha256': file_hash(checkpoint)}, output,
            expected_schema=HU200_SCHEMA)
    return output, file_hash(output), views


def contents(policy, views):
    return (dict(policy.entries.items()), set(policy.zero_mass), dict(policy.visits.items()),
            policy.description, [policy.distribution(v) for v in views])


def test_cached_loads_match_a_parsed_load_and_skip_the_rows(tmp_path, monkeypatch):
    path, sha, views = average(tmp_path)
    parsed = AveragePolicy(path, sha, expected_schema=HU200_SCHEMA)
    cache = tmp_path / 'cache'
    first = AveragePolicy(path, sha, expected_schema=HU200_SCHEMA, cache=cache)
    assert policy_cache.folder(cache, sha, HU200_SCHEMA).is_dir()
    assert isinstance(first.entries.probabilities, np.memmap)
    monkeypatch.setattr(CompactBuilder, 'add', lambda *a: pytest.fail('a cached load parsed rows'))
    second = AveragePolicy(path, sha, expected_schema=HU200_SCHEMA, cache=cache)
    assert contents(parsed, views) == contents(first, views) == contents(second, views)
    assert len(parsed.zero_mass) == 1 and len(parsed.entries) == 3


def test_a_changed_or_mismatched_cache_is_refused(tmp_path):
    path, sha, _ = average(tmp_path)
    cache = tmp_path / 'cache'
    AveragePolicy(path, sha, expected_schema=HU200_SCHEMA, cache=cache)
    target = policy_cache.folder(cache, sha, HU200_SCHEMA)
    data = bytearray((target / 'probabilities.npy').read_bytes())
    data[-1] ^= 1
    (target / 'probabilities.npy').write_bytes(bytes(data))
    with pytest.raises(ValueError, match='differs from its manifest: probabilities'):
        AveragePolicy(path, sha, expected_schema=HU200_SCHEMA, cache=cache)
    (target / 'manifest.json').write_text((target / 'manifest.json').read_text().replace(sha, '0' * 64))
    with pytest.raises(ValueError, match='manifest differs'):
        AveragePolicy(path, sha, expected_schema=HU200_SCHEMA, cache=cache)
    # The model's own hash is still checked before any cache is consulted.
    with pytest.raises(ValueError, match='hash differs'):
        AveragePolicy(path, '1' * 64, expected_schema=HU200_SCHEMA, cache=cache)
