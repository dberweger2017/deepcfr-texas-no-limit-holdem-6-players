"""Package the exact PR207 bytes; explicit approval binds the reviewed release source."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from src.policies.v050_bundle import ASSET_NAME, ASSETS, manifest, sha, verify_bundle, verify_model


def check_source(source_sha):
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if head != source_sha:
        raise ValueError('Package source must be the checked-out commit')
    if subprocess.run(['git', 'diff', '--quiet', source_sha, '--'], cwd=repo).returncode:
        raise ValueError('Package source must have clean tracked bytes')
    for path in ('src/policies/v050_bundle.py', 'docs/releases/v0.5.0/MODEL_CARD.md',
                 'docs/releases/v0.5.0/RELEASE_NOTES.md', 'docs/releases/v0.5.0/INSTALL.md'):
        committed = subprocess.check_output(['git', 'show', f'{source_sha}:{path}'], cwd=repo)
        if committed != (repo / path).read_bytes():
            raise ValueError('Package asset source bytes differ')


def build(source, destination, source_sha, *, publication_approved=False):
    source, destination = Path(source), Path(destination)
    data = manifest(source_sha, publication_approved)
    check_source(source_sha)
    verify_model(source)
    if destination.is_symlink() or destination.exists():
        raise ValueError('Preserve existing package; choose a fresh destination')
    destination.mkdir(parents=True)
    shutil.copyfile(source, destination / ASSET_NAME)
    repo = Path(__file__).resolve().parents[1]
    for name in ('MODEL_CARD.md', 'RELEASE_NOTES.md', 'INSTALL.md'):
        text = (repo / 'docs/releases/v0.5.0' / name).read_text()
        (destination / name).write_text(text.replace('PACKAGE_SOURCE_COMMIT', source_sha))
    shutil.copyfile(repo / 'src/policies/v050_bundle.py', destination / 'verify_v050_bundle.py')
    (destination / 'release-manifest.json').write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    (destination / 'SHA256SUMS').write_text(''.join(f'{sha(destination / name)}  {name}\n' for name in sorted(ASSETS)))
    return verify_bundle(destination, source_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--publication-approved', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.out, args.source_sha,
                           publication_approved=args.publication_approved), sort_keys=True))


if __name__ == '__main__':
    main()
