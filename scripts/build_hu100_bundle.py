"""Package an HU100 release's exact model bytes; explicit approval binds the reviewed release source."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from src.policies.hu100_bundle import RELEASES, assets, manifest, part_layout, sha, verify_bundle, verify_model


def check_source(source_sha, release):
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if head != source_sha:
        raise ValueError('Package source must be the checked-out commit')
    if subprocess.run(['git', 'diff', '--quiet', source_sha, '--'], cwd=repo).returncode:
        raise ValueError('Package source must have clean tracked bytes')
    for path in ('src/policies/hu100_bundle.py', *(f'docs/releases/{release}/{name}'
                 for name in ('MODEL_CARD.md', 'RELEASE_NOTES.md', 'INSTALL.md'))):
        committed = subprocess.check_output(['git', 'show', f'{source_sha}:{path}'], cwd=repo)
        if committed != (repo / path).read_bytes():
            raise ValueError('Package asset source bytes differ')


def build(release, source, destination, source_sha, *, publication_approved=False, card_tables=None):
    source, destination = Path(source), Path(destination)
    check_source(source_sha, release)
    verify_model(source, release)
    pins = RELEASES[release]
    if destination.is_symlink() or destination.exists():
        raise ValueError('Preserve existing package; choose a fresh destination')
    for name, pin in pins.get('card_tables', {}).items():
        path = Path(card_tables) / name if card_tables is not None else None
        if (path is None or path.is_symlink() or not path.is_file()
                or path.stat().st_size != pin['bytes'] or sha(path) != pin['sha256']):
            raise ValueError('Pinned card tables required: ' + name)
    destination.mkdir(parents=True)
    parts = None
    if 'part_bytes' in pins:
        parts = []
        with source.open('rb') as src:
            for part in part_layout(release):
                target = destination / part['file']
                with target.open('xb') as dst:
                    remaining = part['bytes']
                    while remaining:
                        chunk = src.read(min(8 * 1024**2, remaining))
                        if not chunk:
                            raise ValueError('Model ended inside a part')
                        dst.write(chunk); remaining -= len(chunk)
                parts.append({**part, 'sha256': sha(target)})
            if src.read(1):
                raise ValueError('Model exceeds fixed part layout')
        for name in pins['card_tables']:
            shutil.copyfile(Path(card_tables) / name, destination / name)
    else:
        shutil.copyfile(source, destination / pins['model']['file'])
    data = manifest(release, source_sha, publication_approved, model_parts=parts)
    repo = Path(__file__).resolve().parents[1]
    for name in ('MODEL_CARD.md', 'RELEASE_NOTES.md', 'INSTALL.md'):
        text = (repo / 'docs/releases' / release / name).read_text()
        (destination / name).write_text(text.replace('PACKAGE_SOURCE_COMMIT', source_sha))
    shutil.copyfile(repo / 'src/policies/hu100_bundle.py', destination / pins['verifier'])
    (destination / 'release-manifest.json').write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    (destination / 'SHA256SUMS').write_text(''.join(f'{sha(destination / name)}  {name}\n' for name in sorted(assets(release))))
    return verify_bundle(destination, source_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', choices=sorted(RELEASES), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--publication-approved', action='store_true')
    parser.add_argument('--card-tables', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.release, args.source, args.out, args.source_sha,
                           publication_approved=args.publication_approved, card_tables=args.card_tables), sort_keys=True))


if __name__ == '__main__':
    main()
