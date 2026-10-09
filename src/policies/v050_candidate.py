"""Opt-in unpublished HU100 candidate; release defaults remain separately pinned."""

from pathlib import Path

from src.policies.v050_bundle import ASSET_NAME, CANDIDATE, INFERENCE, verify_bundle
from src.policies.hu100_research import load_policy as load_research
from src.play_api.configuration import inference_record

VERSION_ID = CANDIDATE


def load_policy(bundle: Path):
    # Validate the whole package before loading. The reader independently checks
    # every entry and the retained lineage; translation is part of this identity.
    verify_bundle(bundle)
    policy = load_research(bundle / ASSET_NAME, translation=True)
    if inference_record(policy) != INFERENCE:
        raise ValueError('Candidate translation identity differs')
    policy.name = 'v0.5.0 candidate · HU100 · PR207 O1B · seed 2026100601 · translation 512/128'
    return policy
