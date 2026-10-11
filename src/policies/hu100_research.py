"""Explicit #207 terminal research pin; never a release/catalog default."""

from dataclasses import dataclass
from pathlib import Path

from src.blueprint.abstraction import HU100_SCHEMA
from src.blueprint.action_translation import TranslationOptions, VERSION
from src.blueprint.average import AveragePolicy, EXTRACTIONS, HU100_FORMAT
from src.policies.files import file_hash

MODEL_SHA256 = '47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9'
MODEL_BYTES = 1173264021
CHECKPOINT_SHA256 = 'cca0b54a609f47c60b29e9fe5a920475a91ec641df2ff0e3ea48fdac615147ec'
SEED = 2026100601
ITERATION = 885307
ENTRIES = 41010014
ACTUAL_NODES = 1000002065
VERSION_ID = 'research-hu100-pr207-terminal'


@dataclass(frozen=True)
class Identity:
    sha256: str


def load_pinned(path: Path, model: dict, *, translation: bool):
    """An HU100 average whose bytes, lineage and extraction equal `model`'s pins."""
    if type(translation) is not bool:
        raise ValueError('Translation must be explicitly enabled or disabled')
    if not path.is_file() or path.is_symlink() or path.stat().st_size != model['bytes']:
        raise ValueError('HU100 model must be the pinned regular file with matching size')
    if file_hash(path) != model['sha256']:
        raise ValueError('HU100 model SHA256 differs')
    policy = AveragePolicy(path, model['sha256'], expected_schema=HU100_SCHEMA,
                           translation=TranslationOptions() if translation else None)
    info = policy.description
    if (info['training_seed'] != model['seed'] or info['iteration'] != model['iteration']
            or info['entries'] != model['entries'] or info['source_checkpoint_sha256'] != model['checkpoint_sha256']
            or info['strategy'] != EXTRACTIONS['opponent-sampled']
            or info.get('zero_mass_rule') is not None):
        raise ValueError('HU100 model lineage or extraction differs')
    policy.spec = Identity(model['sha256'])
    policy.format_id = HU100_FORMAT
    policy.adapter_id = VERSION if translation else 'direct-v1'
    return policy


def load_policy(path: Path, *, translation=False):
    policy = load_pinned(path, {'bytes': MODEL_BYTES, 'sha256': MODEL_SHA256, 'seed': SEED,
                                'iteration': ITERATION, 'entries': ENTRIES,
                                'checkpoint_sha256': CHECKPOINT_SHA256}, translation=translation)
    policy.name = 'Research HU100 · PR207 terminal · 1,000,002,065 nodes · seed 2026100601'
    policy.research = True
    return policy
