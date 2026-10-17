"""Build unchanged HU200 release bytes from a clean reviewed source commit."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from src.policies.hu200_bundle import CARD_TABLES, DOCS, MODELS, RECIPE, RELEASES, VERIFIER, assets, manifest, part_layout, sha, verify_bundle, verify_model


def check_source(source, release):
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != source:
        raise ValueError('Package source must be the checked-out commit')
    if subprocess.run(['git', 'diff', '--quiet', source, '--'], cwd=repo).returncode:
        raise ValueError('Package source must have clean tracked bytes')
    for path in ('src/policies/hu200_bundle.py', *(f'docs/releases/{release}/{name}' for name in DOCS)):
        if subprocess.check_output(['git', 'show', f'{source}:{path}'], cwd=repo) != (repo / path).read_bytes():
            raise ValueError('Package asset source bytes differ')
    if release == 'hu200-frozen-20b-seeds':
        recipe = repo / 'docs/releases' / release / 'BLEND_CONFIG.json'
        if recipe.stat().st_size != RECIPE['bytes'] or sha(recipe) != RECIPE['sha256']:
            raise ValueError('Pinned blend recipe source differs')
    return repo


def build(release, sources, destination, source_sha, tables, *, publication_approved=False):
    repo = check_source(source_sha, release)
    pins = RELEASES[release]
    if set(sources) != set(pins['models']): raise ValueError('Exact release model inputs required')
    for model, path in sources.items(): verify_model(Path(path), model)
    for name, pin in CARD_TABLES.items():
        path = Path(tables) / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != pin['bytes'] or sha(path) != pin['sha256']:
            raise ValueError('Pinned card tables required: ' + name)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink(): raise ValueError('Preserve existing bundle')
    destination.mkdir(parents=True)
    parts = {}
    for model, path in sources.items():
        parts[model] = []
        with Path(path).open('rb') as src:
            for part in part_layout(model):
                target = destination / part['file']
                with target.open('xb') as dst:
                    remaining = part['bytes']
                    while remaining:
                        chunk = src.read(min(8 * 1024**2, remaining))
                        if not chunk: raise ValueError('Model ended inside a part')
                        dst.write(chunk); remaining -= len(chunk)
                parts[model].append({**part, 'sha256': sha(target)})
            if src.read(1): raise ValueError('Model exceeds fixed part layout')
    for name in CARD_TABLES: shutil.copyfile(Path(tables) / name, destination / name)
    for name in DOCS:
        # These three files ship byte for byte from the tagged source.
        shutil.copyfile(repo / 'docs/releases' / release / name, destination / name)
    if release == 'hu200-frozen-20b-seeds':
        shutil.copyfile(repo / 'docs/releases' / release / 'BLEND_CONFIG.json', destination / RECIPE['file'])
    shutil.copyfile(repo / 'src/policies/hu200_bundle.py', destination / VERIFIER)
    data = manifest(release, source_sha, publication_approved, model_parts=parts)
    (destination / 'release-manifest.json').write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    (destination / 'SHA256SUMS').write_text(''.join(f'{sha(destination / name)}  {name}\n' for name in sorted(assets(release))))
    return verify_bundle(destination, source_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', choices=RELEASES, required=True)
    parser.add_argument('--model', action='append', required=True, help='MODEL_ID=PATH (blend or full seed)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--card-tables', type=Path, required=True)
    parser.add_argument('--publication-approved', action='store_true')
    args = parser.parse_args()
    sources = dict(item.split('=', 1) for item in args.model)
    if len(sources) != len(args.model): parser.error('Duplicate model input')
    print(json.dumps(build(args.release, sources, args.out, args.source_sha, args.card_tables,
                           publication_approved=args.publication_approved), sort_keys=True))


if __name__ == '__main__': main()
