"""Rebuild the exact HU200 blend from the frozen-seed release's verified parts.

This is a reproducibility check, not training or model selection. Every output
is fresh and retained on failure. It needs Rust, Python 3.11, the source checkout
named by the bundle, and 70 GB free after downloading the research assets.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
from time import monotonic

from scripts.build_hu200_bundle import check_source
from src.policies.hu200_bundle import MODELS, RECIPE, SEEDS, sha, verify_bundle, verify_model

RELEASE = 'hu200-frozen-20b-seeds'
CHECKPOINT_BYTES = 6042193732


def put(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def rebuild(bundle: Path, out: Path, source: str, *, require_publication: bool = False) -> dict:
    repo = check_source(source, RELEASE)
    bundle, requested_out = Path(bundle).resolve(), Path(out)
    if requested_out.exists() or requested_out.is_symlink(): raise ValueError('Preserve existing rebuild evidence')
    out = requested_out.resolve()
    cargo = shutil.which('cargo')
    if cargo is None: raise ValueError('Rust cargo is required on PATH')
    if out.resolve().is_relative_to(bundle): raise ValueError('Rebuild outside the immutable bundle')
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 70 * 10**9:
        raise ValueError('Rebuild needs 70 GB free after downloading the assets')
    verified = verify_bundle(bundle, source, require_publication=require_publication)
    if verified['release'] != RELEASE: raise ValueError('The complete frozen-seed collection is required')
    out.mkdir(); started = monotonic(); commands = []
    env = dict(os.environ, CARGO_BUILD_JOBS='1', RAYON_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', CARGO_TARGET_DIR=str((out / 'native-target').resolve()))

    def run(label, command):
        tick = monotonic()
        with (out / (label + '.stdout')).open('x') as stdout, (out / (label + '.stderr')).open('x') as stderr:
            result = subprocess.run(list(map(str, command)), cwd=repo, env=env, stdout=stdout, stderr=stderr)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        record = {'label': label, 'command': list(map(str, command)), 'returncode': result.returncode,
                  'seconds': monotonic() - tick,
                  'cumulative_child_peak_rss_bytes': usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}
        commands.append(record); put(out / 'commands.json', commands)
        if result.returncode: raise RuntimeError(f'Rebuild step {label} failed; preserve its logs')
        return record

    inputs = out / 'inputs'
    verify_bundle(bundle, source, require_publication=require_publication, assemble=inputs)
    put(out / 'assembled-inputs.json', {model: {'path': str(inputs / pin['file']), 'bytes': pin['bytes'], 'sha256': pin['sha256']}
                                      for model, pin in MODELS.items() if model != 'blend'})
    run('build-tools', [cargo, 'build', '--release', '--locked', '--manifest-path',
                        repo / 'native/hu20-trainer/Cargo.toml', '--example', 'hu200_blend', '--bin', 'hu20-trainer'])
    extension = '.exe' if sys.platform == 'win32' else ''
    trainer = out / 'native-target/release' / ('hu20-trainer' + extension)
    tool = out / 'native-target/release/examples' / ('hu200_blend' + extension)
    averages = out / 'reexported-seeds'; averages.mkdir()
    for seed in SEEDS:
        exported = averages / MODELS[seed]['file']
        run('export-' + seed, [trainer, 'export', inputs / MODELS[seed + '-checkpoint']['file'],
                               '--average', exported, '--zero-mass', 'uniform'])
        verify_model(exported, seed)
        put(out / ('verified-' + seed + '.json'), {'sha256': MODELS[seed]['sha256'], 'bytes': exported.stat().st_size})
    recipe = bundle / RECIPE['file']
    scan = out / 'scan.json'
    run('scan', [tool, 'scan', recipe, scan, out / 'sample.checkpoint.gz', '--input-dir', inputs])
    statistics = json.loads(scan.read_text())
    if (statistics['union_keys'] != MODELS['blend']['entries'] or statistics['action_slots'] != 482975338
            or statistics['configuration_sha256'] != RECIPE['sha256']):
        raise ValueError('Rebuilt union or configuration differs')
    if shutil.disk_usage(out).free < 18 * 1024**3 + CHECKPOINT_BYTES + MODELS['blend']['bytes']:
        raise ValueError('Preserve the 18 GiB disk floor through merge and export')
    gate = out / 'admission.json'
    put(gate, {'status': 'admitted', 'configuration_sha256': RECIPE['sha256'],
               'all_seed_averages_reproduced': True, 'scan_report': str(scan.resolve()), 'scan_sha256': sha(scan),
               'purpose': 'Exact frozen release reconstruction; no training, evaluation or selection',
               'source': source})
    checkpoint = out / 'rebuilt.checkpoint.gz'
    run('merge', [tool, 'merge', recipe, out / 'merge.json', checkpoint, gate, '--input-dir', inputs])
    if checkpoint.stat().st_size != CHECKPOINT_BYTES or sha(checkpoint) != MODELS['blend']['checkpoint_sha256']:
        raise ValueError('Rebuilt checkpoint bytes/SHA256 differ')
    average = out / MODELS['blend']['file']
    run('export-blend', [trainer, 'export', checkpoint, '--average', average, '--zero-mass', 'uniform'])
    verify_model(average)
    receipt = {'status': 'verified', 'source': source, 'bundle_manifest_sha256': sha(bundle / 'release-manifest.json'),
               'publication_required': require_publication, 'recipe_sha256': sha(recipe),
               'all_seed_averages_reproduced': True, 'union_keys': statistics['union_keys'],
               'checkpoint': {'bytes': checkpoint.stat().st_size, 'sha256': sha(checkpoint)},
               'average': {'bytes': average.stat().st_size, 'sha256': sha(average)},
               'commands': commands, 'seconds': monotonic() - started,
               'no_training_evaluation_or_selection': True}
    put(out / 'rebuild-verification.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expect-source', required=True)
    parser.add_argument('--require-publication', action='store_true')
    args = parser.parse_args()
    print(json.dumps(rebuild(args.bundle, args.out, args.expect_source,
                             require_publication=args.require_publication), sort_keys=True))


if __name__ == '__main__': main()
