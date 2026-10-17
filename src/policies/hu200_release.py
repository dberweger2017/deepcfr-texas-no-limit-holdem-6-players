"""The released HU200 blend: fixed bytes, K50 cards and public-history translation."""
from pathlib import Path

from scripts.build_streaming_average_cache import build as build_cache
from src.blueprint import policy_cache
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy, EXTRACTIONS
from src.play_api.configuration import inference_record
from src.policies.hu100_research import Identity
from src.policies.files import local_cache
from src.policies.hu200_bundle import FORMAT, INFERENCE, MODELS, RELEASES, SCHEMA, verify_bundle, verify_model

ACTION_SLOTS = 482975338


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
    cache = bundle.parent / (bundle.name + '-cache')
    if cache.is_symlink():
        raise ValueError('HU200 play cache must be a local directory, not a symlink')
    cache = local_cache(cache)
    target = policy_cache.folder(cache, pin['sha256'], SCHEMA)
    try:
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise ValueError('Invalid cache directory')
        if not target.exists():
            build_cache(model, pin['sha256'], SCHEMA, cache, pin['entries'], ACTION_SLOTS)
        for name in ('manifest.json', *(n + '.npy' for n in policy_cache.ARRAYS)):
            if (target / name).is_symlink():
                raise ValueError('Cache files must not be symlinks')
        # The streaming builder bounds the first load's RAM. Every load, including
        # that first one, verifies the cache and maps its arrays read-only.
        policy = AveragePolicy(model, pin['sha256'], expected_schema=SCHEMA,
                               translation=TranslationOptions(), card_buckets=bundle,
                               cache=cache, cache_read_only=True)
        if len(policy.entries.probabilities) != ACTION_SLOTS:
            raise ValueError('HU200 cache action count differs')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f'HU200 play cache refused at {target}; preserve it, move it aside '
                         'and restart to rebuild from the verified model') from error
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
