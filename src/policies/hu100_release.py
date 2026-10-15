"""Published HU100 releases: a verified bundle's exact average, with public-history translation."""

from pathlib import Path

from src.policies.hu100_bundle import RELEASES, inference, verify_bundle, verify_model
from src.policies.hu100_research import load_pinned
from src.play_api.configuration import inference_record

# Sessions store the full model identity. v0.5.0 shipped with the research flag set, so it
# keeps it and its sessions stay resumable; later releases are not labeled research.
RESEARCH_FLAG = {'v0.5.0': True}


def load_policy(bundle: Path):
    """The bundle's release and policy. The whole bundle is verified before loading; the reader
    then checks every entry and the lineage again, and translation is part of the identity."""
    release = verify_bundle(bundle)['release']
    pins = RELEASES[release]
    if 'part_bytes' in pins:
        model = bundle.parent / (bundle.name + '-model') / pins['model']['file']
        if model.exists() or model.is_symlink():
            verify_model(model, release)
        else:
            verify_bundle(bundle, assemble=model)
        policy = load_pinned(model, pins['model'], translation=True, card_buckets=bundle)
    else:
        policy = load_pinned(bundle / pins['model']['file'], pins['model'], translation=True)
    if inference_record(policy) != inference(release):
        raise ValueError(f'{release} translation identity differs')
    policy.name = pins['name']
    policy.research = RESEARCH_FLAG.get(release, False)
    return release, policy
