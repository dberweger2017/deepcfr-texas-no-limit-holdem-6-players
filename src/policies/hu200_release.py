"""The released HU200 blend: fixed bytes, K50 cards and public-history translation."""
from pathlib import Path

from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy, EXTRACTIONS
from src.play_api.configuration import inference_record
from src.policies.hu100_research import Identity
from src.policies.hu200_bundle import FORMAT, INFERENCE, MODELS, RELEASES, SCHEMA, verify_bundle, verify_model


def load_policy(bundle: Path):
    release = verify_bundle(bundle)['release']
    if release != 'v0.5.5':
        raise ValueError('Research seeds are verification-only assets, not a table release')
    pin = MODELS['blend']
    model = bundle.parent / (bundle.name + '-model') / pin['file']
    if model.exists() or model.is_symlink():
        verify_model(model)
    else:
        verify_bundle(bundle, assemble=model.parent)
    policy = AveragePolicy(model, pin['sha256'], expected_schema=SCHEMA,
                           translation=TranslationOptions(), card_buckets=bundle, sorted_rows=True)
    info = policy.description
    if (info['training_seed'] != pin['seed'] or info['iteration'] != pin['iteration']
            or info['entries'] != pin['entries'] or info['source_checkpoint_sha256'] != pin['checkpoint_sha256']
            or info['strategy'] != EXTRACTIONS['opponent-sampled'] or info.get('zero_mass_rule') is not None):
        raise ValueError('HU200 blend lineage or extraction differs')
    policy.spec = Identity(pin['sha256'])
    policy.format_id = FORMAT
    policy.adapter_id = INFERENCE['adapter']
    if inference_record(policy) != INFERENCE:
        raise ValueError('HU200 translation identity differs')
    policy.name = RELEASES[release]['name']
    policy.research = False
    return release, policy
