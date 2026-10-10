"""The v0.5.0 HU100 release: the #207 1B average with public-history translation."""

from pathlib import Path

from src.policies.v050_bundle import ASSET_NAME, INFERENCE, RELEASE, verify_bundle
from src.policies.hu100_research import load_policy as load_research
from src.play_api.configuration import inference_record

VERSION_ID = RELEASE
NAME = 'v0.5.0 · HU100 · O1B · seed 2026100601 · translation 512/128'


def load_policy(bundle: Path):
    # Validate the whole bundle before loading. The reader independently checks
    # every entry and the retained lineage; translation is part of this identity.
    verify_bundle(bundle)
    policy = load_research(bundle / ASSET_NAME, translation=True)
    if inference_record(policy) != INFERENCE:
        raise ValueError('v0.5.0 translation identity differs')
    policy.name = NAME
    return policy
