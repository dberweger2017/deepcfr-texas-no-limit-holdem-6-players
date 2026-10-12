"""Build the native trainer so its bytes don't depend on where the checkout lives.

Rust embeds absolute source paths in panic messages, and Cargo hashes the build flags into
every crate's symbols, so a plain `cargo build` of the same source in two worktrees gives two
different binaries. This build copies `native/` to one fixed staging directory, builds there
and maps that directory and Cargo's home out of every embedded path. The flags are then the
same from any checkout, and the same source, lockfile, toolchain, linker and Cargo home give
the same bytes. It prints a receipt that names them; when two receipts differ in any of them,
their binaries are expected to differ. The binary is copied to the usual target path.

    python -m scripts.build_native_trainer [--receipt PATH]
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'native'
MANIFEST = NATIVE / 'hu20-trainer' / 'Cargo.toml'
BINARY = NATIVE / 'hu20-trainer' / 'target' / 'release' / 'hu20-trainer'
# Fixed, so the remapping flags never name the checkout. One build at a time owns it.
STAGE = Path('/tmp/hu20-trainer-build').resolve()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output(command):
    result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=ROOT)
    return (result.stdout + result.stderr).strip()


def remaps():
    """Path prefixes to replace; the longest first, so nested homes map correctly."""
    cargo_home = Path(os.environ.get('CARGO_HOME', Path.home() / '.cargo')).resolve()
    pairs = [(STAGE / 'native', 'native'), (cargo_home, 'cargo-home')]
    return sorted(pairs, key=lambda pair: -len(str(pair[0])))


def stage():
    """A fresh copy of the checkout's native sources, without build outputs."""
    try:
        STAGE.mkdir(parents=True)
    except FileExistsError:
        raise SystemExit(f'{STAGE} exists: another build is running, or a failed one left it; remove it first')
    shutil.copytree(NATIVE, STAGE / 'native', ignore=shutil.ignore_patterns('target'))


def build():
    if os.environ.get('RUSTFLAGS') or os.environ.get('CARGO_ENCODED_RUSTFLAGS'):
        raise SystemExit('Unset RUSTFLAGS/CARGO_ENCODED_RUSTFLAGS: they would replace the path remapping')
    # The encoded form keeps paths with spaces intact; arguments are separated by 0x1f.
    flags = '\x1f'.join(f'--remap-path-prefix={source}={target}' for source, target in remaps())
    env = dict(os.environ, CARGO_ENCODED_RUSTFLAGS=flags, CARGO_TARGET_DIR=str(STAGE / 'target'))
    stage()
    try:
        subprocess.run(['cargo', 'build', '--release', '--locked', '--manifest-path',
                        str(STAGE / 'native' / 'hu20-trainer' / 'Cargo.toml')],
                       check=True, cwd=STAGE / 'native' / 'hu20-trainer', env=env)
        BINARY.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(STAGE / 'target' / 'release' / 'hu20-trainer', BINARY)
    finally:
        shutil.rmtree(STAGE)
    clean = not output(['git', 'status', '--porcelain', '--', 'native'])
    return {
        'binary': str(BINARY.relative_to(ROOT)),
        'binary_bytes': BINARY.stat().st_size,
        'binary_sha256': sha256(BINARY),
        'native_tree': output(['git', 'rev-parse', 'HEAD:native']) if clean else None,
        'native_tree_clean': clean,
        'cargo_lock_sha256': sha256(MANIFEST.parent / 'Cargo.lock'),
        'rustc': output(['rustc', '-vV']).splitlines(),
        'linker': output(['ld', '-v']).splitlines()[0],
        'remapped_prefixes': [target for _, target in remaps()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--receipt', type=Path, help='also write the receipt to this new file')
    args = parser.parse_args()
    receipt = build()
    text = json.dumps(receipt, indent=2, sort_keys=True) + '\n'
    if args.receipt:
        with args.receipt.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
