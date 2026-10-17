"""The no-workspace audit agrees with the original and rejects corrupt streams."""
import gzip
import json
import pytest

from scripts.audit_hu200_capacity_sorted import audit_sorted
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.diagnostics.cfr_average import audit, extract
from src.policies.files import file_hash
from tests.diagnostics.test_cfr_average import fixture


def run_sorted(checkpoint, current, average, spec):
    return audit_sorted(checkpoint, current, average, spec, file_hash(average), expected_schema=HU20_UNCAPPED_SCHEMA)


def lines(path):
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]


def write_lines(path, rows):
    path.write_bytes(gzip.compress(('\n'.join(json.dumps(row) for row in rows) + '\n').encode(), mtime=0))


@pytest.mark.parametrize('rule', ['traverser-reach', 'opponent-sampled'])
@pytest.mark.parametrize('fallback', ['uniform', 'current'])
def test_matches_reference_audit_exactly(tmp_path, rule, fallback):
    _, _, checkpoint, current, spec = fixture(tmp_path)
    if rule == 'opponent-sampled':
        rows = lines(checkpoint)
        rows[0]['average_rule'] = rule
        rows[-1][3] = [17.0]  # Opponent-only positive mass without a traverser visit.
        write_lines(checkpoint, rows)
        spec['checkpoint_sha256'] = file_hash(checkpoint)
    average = tmp_path / 'average.gz'
    extract(checkpoint, spec, average, zero_mass=fallback)
    assert run_sorted(checkpoint, current, average, spec) == audit(checkpoint, current, average, spec, file_hash(average))


@pytest.mark.parametrize('damage', ['probability', 'average-key', 'missing-row', 'current-probability', 'late-metadata', 'current-trailing', 'duplicate-row'])
def test_rejects_semantically_corrupt_files_even_with_matching_input_hashes(tmp_path, damage):
    _, _, checkpoint, current, spec = fixture(tmp_path)
    average = tmp_path / 'average.gz'
    extract(checkpoint, spec, average)
    if damage in ('probability', 'average-key', 'missing-row', 'duplicate-row'):
        rows = lines(average)
        if damage == 'probability':rows[1][2] = [1] + [0] * (len(rows[1][2]) - 1)
        elif damage == 'average-key':rows[1][0] = '0' * 32
        elif damage == 'missing-row':rows.pop()
        else:rows.append(rows[-1])
        write_lines(average, rows)
    else:
        raw = gzip.decompress(current.read_bytes()).decode()
        data = json.loads(raw)
        if damage == 'current-probability':
            row = next(iter(data['entries'].values()))
            row[1] = [0] * (len(row[1]) - 1) + [1]
        elif damage == 'late-metadata':data['strategy'] = 'average'
        raw = json.dumps(data) + (' trailing' if damage == 'current-trailing' else '')
        current.write_bytes(gzip.compress(raw.encode(), mtime=0))
        spec['sha256'] = file_hash(current)
    with pytest.raises(ValueError):run_sorted(checkpoint, current, average, spec)


def test_unordered_files_are_refused_even_when_reference_can_match_them(tmp_path):
    _, _, checkpoint, current, spec = fixture(tmp_path)
    rows = lines(checkpoint)
    write_lines(checkpoint, [rows[0], *reversed(rows[1:])])
    spec['checkpoint_sha256'] = file_hash(checkpoint)
    current_data = json.loads(gzip.decompress(current.read_bytes()))
    current_data['entries'] = dict(reversed(list(current_data['entries'].items())))
    current.write_bytes(gzip.compress(json.dumps(current_data).encode(), mtime=0))
    spec['sha256'] = file_hash(current)
    average = tmp_path / 'average.gz'
    extract(checkpoint, spec, average)
    assert audit(checkpoint, current, average, spec, file_hash(average))['status'] == 'verified'
    with pytest.raises(ValueError, match='sorted'):
        run_sorted(checkpoint, current, average, spec)


def test_current_gzip_footer_is_read_and_verified(tmp_path):
    _, _, checkpoint, current, spec = fixture(tmp_path)
    average = tmp_path / 'average.gz'
    extract(checkpoint, spec, average)
    current.write_bytes(current.read_bytes()[:-8])
    spec['sha256'] = file_hash(current)
    with pytest.raises(EOFError):run_sorted(checkpoint, current, average, spec)
