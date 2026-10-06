"""Stage the unchanged O candidate for review; never publish or create a tag."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from src.play_api.o_candidate import ASSET_NAME, CHECKPOINT_SHA256, MODEL_BYTES, MODEL_SHA256, verify
from src.diagnostics.cfr_average import EXTRACTIONS, FORMAT
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.blueprint.solver import HU20_UNCAPPED_GAME


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def build(source, destination, source_sha, *, publication_approved=False):
    verify(source)
    if type(publication_approved) is not bool:
        raise ValueError('Publication approval must be explicit boolean')
    if len(source_sha) != 40 or any(c not in '0123456789abcdef' for c in source_sha):
        raise ValueError('Full lowercase preparation Git SHA required')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Preserve the existing candidate bundle')
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination / ASSET_NAME)
    verify(destination / ASSET_NAME)
    docs = Path(__file__).resolve().parents[1] / 'docs/releases/v0.4.1'
    for name in ('MODEL_CARD.md', 'RELEASE_NOTES.md'):
        shutil.copyfile(docs / name, destination / name)
    manifest = {
        'candidate': 'v0.4.1',
        'status': 'owner-approved-publication' if publication_approved else 'unpublished-owner-review',
        'preparation_source_commit': source_sha,
        'approved_release_source_commit': source_sha if publication_approved else None,
        'owner_publication_approval': publication_approved,
        'confirmation': 'https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/176',
        'plan_sha256': '49a1d6a05e82eaf929c5f76a275c9e3ed90a5a99b129cbb2817dd4b69c2c9a1b',
        'model': {'file': ASSET_NAME, 'bytes': MODEL_BYTES, 'sha256': MODEL_SHA256,
                  'format': FORMAT, 'game': HU20_UNCAPPED_GAME, 'schema': HU20_UNCAPPED_SCHEMA,
                  'seed': 2026100601, 'iteration': 2126271, 'training_budget_nodes': 1000000000,
                  'checkpoint_sha256': CHECKPOINT_SHA256, 'extraction': EXTRACTIONS['opponent-sampled'],
                  'players': 2, 'raise_cap': None, 'resumable': False},
        'dependency': {'pokers_git_revision': '5db20e3d5d6862b32a7402035c1340b622d3b005', 'python': '3.11'},
        'publication_gate': 'Owner chat approval; approved merged source; regenerate manifest before publication',
    }
    (destination / 'release-manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    names = (ASSET_NAME, 'MODEL_CARD.md', 'RELEASE_NOTES.md', 'release-manifest.json')
    (destination / 'SHA256SUMS').write_text(''.join(f'{sha(destination / name)}  {name}\n' for name in names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    build(args.source, args.out, args.source_sha)
    print(f'Staged unpublished O candidate in {args.out}')


if __name__ == '__main__':
    main()
