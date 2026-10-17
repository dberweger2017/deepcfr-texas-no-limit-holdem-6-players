"""HU200 split trust anchors, source binding and verification-only research models."""
import gzip
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
from scripts import build_hu200_bundle as builder
from src.policies import hu200_bundle as verifier


@pytest.fixture(params=verifier.RELEASES)
def package(request, tmp_path, monkeypatch):
    release = request.param
    monkeypatch.setattr(builder, 'check_source', lambda *_: Path(__file__).resolve().parents[1])
    monkeypatch.setattr(verifier, 'PART_BYTES', 4096)
    tables = tmp_path / 'tables'; tables.mkdir()
    pins = {}
    for name in verifier.CARD_TABLES:
        p = tables / name; p.write_bytes(name.encode())
        pins[name] = {'bytes': p.stat().st_size, 'sha256': verifier.sha(p)}
    monkeypatch.setattr(verifier, 'CARD_TABLES', pins)
    monkeypatch.setattr(builder, 'CARD_TABLES', pins)
    sources = {}
    for model in verifier.RELEASES[release]['models']:
        pin = dict(verifier.MODELS[model])
        header = {'format': verifier.FORMAT, 'kind': 'diagnostic-inference', 'extraction': verifier.EXTRACTION,
                  'source_checkpoint_sha256': pin['checkpoint_sha256'],
                  'zero_mass_rule': 'uniform in retained menu; reported separately from missing keys',
                  'checkpoint_header': {'config': {'seed': pin['seed'], 'game': verifier.GAME, 'raise_cap': None},
                    'abstraction': verifier.SCHEMA, 'iteration': pin['iteration'], 'average_rule': 'opponent-sampled',
                    'identity': {'players': 2, 'card_tables': {n.removesuffix('-k50.bin'): p['sha256'] for n, p in pins.items()}},
                    'table': {'stacks': [20000, 20000], 'small_blind': 50, 'big_blind': 100, 'chip_unit': '0.01'}}}
        if model == 'blend':
            header['checkpoint_header']['blend_provenance'] = {
                'model_identity': pin['synthetic_identity'], 'kind': 'synthetic-sequence-form-average',
                'inputs': [{'seed': p['seed'], 'average_sha256': p['sha256'], 'sha256': p['checkpoint_sha256'],
                            'iterations': p['iteration']} for m, p in verifier.MODELS.items() if m in verifier.SEEDS]}
        if pin.get('kind') == 'checkpoint':
            header = {**header['checkpoint_header'], 'kind': 'training',
                      'format': 'holdem-hu200-native-reopening-blueprint-v1',
                      'native_state': {'completed_nodes': pin['actual_nodes']}}
        p = tmp_path / pin['file']
        p.write_bytes(gzip.compress((json.dumps(header) + '\n' + random.Random(5).randbytes(4096).hex()).encode(), mtime=0))
        pin.update(bytes=p.stat().st_size, sha256=verifier.sha(p))
        monkeypatch.setitem(verifier.MODELS, model, pin); sources[model] = p
    # Research documents have their own directory; fixture content is irrelevant to binary verification.
    original_copy = builder.shutil.copyfile
    def copy(source, destination):
        if Path(source).name in verifier.DOCS and not Path(source).exists():
            Path(destination).write_text('Research seed fixture\n'); return destination
        return original_copy(source, destination)
    monkeypatch.setattr(builder.shutil, 'copyfile', copy)
    directory = tmp_path / 'bundle'
    builder.build(release, sources, directory, 'a' * 40, tables)
    return release, directory, sources, tables


def rehash(directory, release):
    (directory / 'SHA256SUMS').write_text(''.join(f'{verifier.sha(directory / n)}  {n}\n' for n in sorted(verifier.assets(release))))


def test_roundtrip_and_publication(package, tmp_path):
    release, directory, sources, tables = package
    assembled = tmp_path / 'assembled'
    result = verifier.verify_bundle(directory, 'a' * 40, assemble=assembled)
    assert result['release'] == release
    for model, source in sources.items():
        assert (assembled / verifier.MODELS[model]['file']).read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match='Publication'):
        verifier.verify_bundle(directory, 'a' * 40, require_publication=True)
    approved = tmp_path / 'approved'
    builder.build(release, sources, approved, 'b' * 40, tables, publication_approved=True)
    assert verifier.verify_bundle(approved, 'b' * 40, require_publication=True)['owner_publication_approval']
    with pytest.raises(ValueError, match='source'):
        verifier.verify_bundle(approved, 'a' * 40, require_publication=True)
    assert verifier.RELEASES[release]['latest'] is (release == 'v0.5.5')


def test_rehashed_parts_cannot_substitute_model(package, tmp_path):
    release, directory, _, _ = package
    path = directory / 'release-manifest.json'; data = json.loads(path.read_text())
    model = next(iter(data['model_parts'])); part = data['model_parts'][model][-1]
    p = directory / part['file']; payload = p.read_bytes()
    p.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1])); part['sha256'] = verifier.sha(p)
    path.write_text(json.dumps(data)); rehash(directory, release)
    output = tmp_path / 'failed'
    with pytest.raises(ValueError, match='pinned HU200'):
        verifier.verify_bundle(directory, assemble=output)
    assert not list(output.iterdir())


def test_rehashed_table_identity_order_and_approval_rejected(package):
    release, directory, _, _ = package
    path = directory / 'release-manifest.json'; original = path.read_text()
    for change in ({'inference': {}}, {'owner_publication_approval': 0}, {'latest': False if release == 'v0.5.5' else True}):
        data = json.loads(original); data.update(change); path.write_text(json.dumps(data)); rehash(directory, release)
        with pytest.raises(ValueError): verifier.verify_bundle(directory)
    data = json.loads(original); next(iter(data['model_parts'].values())).reverse()
    path.write_text(json.dumps(data)); rehash(directory, release)
    with pytest.raises(ValueError, match='ordered'): verifier.verify_bundle(directory)
    path.write_text(original)
    (directory / 'river-k50.bin').write_bytes(b'wrong'); rehash(directory, release)
    with pytest.raises(ValueError, match='Pinned card table'): verifier.verify_bundle(directory)


def test_hu200_cli_refuses_other_depths_and_translation_override():
    for arguments in (['--stack-bb', '20'], ['--stack-bb', '100'], ['--translate-off-menu']):
        result = subprocess.run([sys.executable, '-m', 'src.play_api.server', '--hu200-release', 'missing', *arguments],
                                capture_output=True, text=True)
        assert result.returncode == 2 and 'Traceback' not in result.stderr
