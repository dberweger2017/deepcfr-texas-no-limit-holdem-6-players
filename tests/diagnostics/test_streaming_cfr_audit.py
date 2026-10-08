"""Format/coverage corruption checks using immutable checkpoints, without training."""
import gzip
import json
from pathlib import Path
import subprocess

import pytest

from tests.diagnostics.test_cfr_average import fixture
from src.diagnostics.cfr_average import extract, audit
from src.policies.files import file_hash
from src.blueprint.streaming import current_rows

BINARY = Path('native/hu20-trainer/target/release/hu20-trainer')


def write(path, text):
    path.write_bytes(gzip.compress(text.encode(), mtime=0))


def inputs(tmp_path):
    _, _, checkpoint, current, spec = fixture(tmp_path)
    average = tmp_path / 'average.gz'
    extract(checkpoint, spec, average)
    return checkpoint, current, average, spec


def verify(checkpoint, current, average, spec):
    return audit(checkpoint, current, average,
                 {**spec, 'checkpoint_sha256': file_hash(checkpoint), 'sha256': file_hash(current)},
                 file_hash(average))


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'extra', 'probability', 'metadata', 'truncated', 'trailing', 'duplicate_field'])
def test_current_corruption_rejected_even_with_updated_hash(tmp_path, mutation):
    checkpoint, current, average, spec = inputs(tmp_path)
    text = gzip.decompress(current.read_bytes()).decode()
    document = json.loads(text)
    key, row = next(iter(document['entries'].items()))
    if mutation == 'duplicate':
        text = text.replace('"entries":{', '"entries":{' + json.dumps(key) + ':' + json.dumps(row) + ',')
    elif mutation == 'missing':
        del document['entries'][key]; text = json.dumps(document)
    elif mutation == 'extra':
        document['entries']['a' * 32] = row; text = json.dumps(document)
    elif mutation == 'probability':
        document['entries'][key][1] = [0, 1]; text = json.dumps(document)
    elif mutation == 'metadata':
        document['iteration'] += 1; text = json.dumps(document)
    elif mutation == 'truncated':
        text = text[:-1]
    elif mutation == 'trailing':
        text += '{}'
    elif mutation == 'duplicate_field':
        text = '{"iteration":10,' + text[1:]
    write(current, text)
    with pytest.raises((ValueError, TypeError)):
        verify(checkpoint, current, average, spec)


@pytest.mark.parametrize('which', ['checkpoint', 'average', 'current'])
def test_truncated_gzip_is_rejected(tmp_path, which):
    checkpoint, current, average, spec = inputs(tmp_path)
    path = {'checkpoint': checkpoint, 'current': current, 'average': average}[which]
    path.write_bytes(path.read_bytes()[:-6])
    with pytest.raises((ValueError, EOFError)):
        verify(checkpoint, current, average, spec)


def test_duplicate_checkpoint_rejected_by_extraction_and_audit(tmp_path):
    checkpoint, current, average, spec = inputs(tmp_path)
    lines = gzip.decompress(checkpoint.read_bytes()).decode().splitlines()
    lines.append(lines[1]); write(checkpoint, '\n'.join(lines) + '\n')
    # Bind metadata to the changed checkpoint so duplicate detection itself is exercised.
    spec['checkpoint_sha256'] = file_hash(checkpoint)
    emitted = gzip.decompress(average.read_bytes()).decode().splitlines()
    metadata = json.loads(emitted[0]); metadata['source_checkpoint_sha256'] = spec['checkpoint_sha256']
    write(average, '\n'.join([json.dumps(metadata), *emitted[1:], emitted[1]]) + '\n')
    with pytest.raises(ValueError, match='Duplicate'):
        verify(checkpoint, current, average, spec)
    with pytest.raises(ValueError, match='Duplicate'):
        extract(checkpoint, spec, tmp_path / 'duplicate-extract.gz')
    assert not (tmp_path / 'duplicate-extract.gz').exists()


def test_unordered_checkpoint_and_current_are_fully_verified(tmp_path):
    checkpoint, current, average, spec = inputs(tmp_path)
    lines = gzip.decompress(checkpoint.read_bytes()).decode().splitlines()
    write(checkpoint, '\n'.join([lines[0], *reversed(lines[1:])]) + '\n')
    spec['checkpoint_sha256'] = file_hash(checkpoint)
    average.unlink(); extract(checkpoint, spec, average)
    document = json.loads(gzip.decompress(current.read_bytes()))
    document['entries'] = dict(reversed(list(document['entries'].items())))
    write(current, json.dumps(document))
    assert verify(checkpoint, current, average, spec)['all_nodes_verified'] == 2


class Chunked:
    def __init__(self, text, size):
        self.text = text; self.size = size
    def read(self, _):
        result, self.text = self.text[:self.size], self.text[self.size:]
        return result


@pytest.mark.parametrize('size', [1, 2, 7, 65536])
def test_json_chunk_boundaries_escaping_and_numbers(size):
    document = {'iteration': 100123, 'entries': {'a' * 32: [['check"\\\n'], [1.0]]}, 'kind': 'inference'}
    metadata = {}
    assert dict(current_rows(Chunked(json.dumps(document), size), metadata)) == document['entries']
    assert metadata == {k: v for k, v in document.items() if k != 'entries'}


@pytest.mark.skipif(not BINARY.exists(), reason='build native trainer')
@pytest.mark.parametrize('mode', ['both', 'current', 'average'])
def test_native_streams_keep_current_sort_and_average_source_order(tmp_path, mode):
    checkpoint, current, average, spec = inputs(tmp_path)
    lines = gzip.decompress(checkpoint.read_bytes()).decode().splitlines()
    # Canonical native menu, deliberately unordered source rows.
    rows = [[key, ['check', 'pot'], [1.0, 2.0], [0.0, 3.0], 1] for key in ('f' * 32, '0' * 32)]
    write(checkpoint, lines[0] + '\n' + '\n'.join(map(json.dumps, rows)) + '\n')
    args = [str(BINARY), 'export', str(checkpoint)]
    if mode != 'average': args += ['--current', str(tmp_path / 'native-current.gz')]
    if mode != 'current': args += ['--average', str(tmp_path / 'native-average.gz')]
    subprocess.run(args, check=True, capture_output=True)
    if mode != 'average':
        data = json.loads(gzip.decompress((tmp_path / 'native-current.gz').read_bytes()))
        assert list(data['entries']) == sorted(row[0] for row in rows)
    if mode != 'current':
        data = [json.loads(x) for x in gzip.open(tmp_path / 'native-average.gz', 'rt')]
        assert [row[0] for row in data[1:]] == [row[0] for row in rows]


@pytest.mark.skipif(not BINARY.exists(), reason='build native trainer')
@pytest.mark.parametrize('mutation', ['duplicate', 'malformed', 'truncated', 'cap', 'bound'])
def test_native_rejects_corrupt_checkpoint_without_publishing(tmp_path, mutation):
    checkpoint, _, _, _ = inputs(tmp_path)
    lines = gzip.decompress(checkpoint.read_bytes()).decode().splitlines()
    header = json.loads(lines[0])
    rows = [[key, ['check'], [1.0], [2.0], 1] for key in ('f' * 32, '0' * 32)]
    if mutation == 'duplicate': rows.append(rows[0])
    elif mutation == 'malformed': rows[0][2] = []
    elif mutation == 'cap': header['config']['max_entries'] = 1
    elif mutation == 'bound': rows[0][3] = [1e6]
    write(checkpoint, json.dumps(header) + '\n' + '\n'.join(map(json.dumps, rows)) + '\n')
    if mutation == 'truncated': checkpoint.write_bytes(checkpoint.read_bytes()[:-6])
    out = tmp_path / 'native-average.gz'
    result = subprocess.run([str(BINARY), 'export', str(checkpoint), '--average', str(out)], capture_output=True)
    assert result.returncode != 0
    assert not out.exists()


@pytest.mark.parametrize('number', ['1.23', '1e12', '-1.23e-12'])
@pytest.mark.parametrize('size', [1, 2, 3, 7])
def test_scalar_number_boundaries(number, size):
    from src.blueprint.streaming import JsonStream
    assert JsonStream(Chunked(number + ',', size)).value() == json.loads(number)


@pytest.mark.parametrize('invalid', ['\v', '\f', '\u00a0'])
def test_non_json_whitespace_rejected(invalid):
    with pytest.raises(ValueError):
        list(current_rows(Chunked('{"entries":{}' + invalid + '}', 1), {}))


def test_malformed_early_row_does_not_read_rest_of_export():
    source = Chunked('{"entries":{"' + 'a' * 32 + '":[?,' + ' ' * 1000000 + ']}}', 64)
    with pytest.raises(ValueError):
        list(current_rows(source, {}))
    assert len(source.text) > 900000
