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


def verify(path: Path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != MODEL_BYTES:
        raise ValueError('HU100 research model must be the pinned regular file with matching size')
    if file_hash(path) != MODEL_SHA256:
        raise ValueError('HU100 research model SHA256 differs')
    return MODEL_SHA256


def load_policy(path: Path, *, translation=False):
    if type(translation) is not bool:
        raise ValueError('Translation must be explicitly enabled or disabled')
    verify(path)
    policy = AveragePolicy(path, MODEL_SHA256, expected_schema=HU100_SCHEMA,
                           translation=TranslationOptions() if translation else None)
    info = policy.description
    if (info['training_seed'] != SEED or info['iteration'] != ITERATION
            or info['entries'] != ENTRIES or info['source_checkpoint_sha256'] != CHECKPOINT_SHA256
            or info['strategy'] != EXTRACTIONS['opponent-sampled']
            or info.get('zero_mass_rule') is not None):
        raise ValueError('HU100 research lineage or extraction differs')
    policy.spec = Identity(MODEL_SHA256)
    policy.name = 'Research HU100 · PR207 terminal · 1,000,002,065 nodes · seed 2026100601'
    policy.format_id = HU100_FORMAT
    policy.adapter_id = VERSION if translation else 'direct-v1'
    policy.research = True
    return policy
