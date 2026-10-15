"""Capacity identity permits only the cap; float bytes, keys and counts stay exact."""
import gzip
import json
from pathlib import Path
import pytest
from scripts.hu200_capacity_preflight import verify_rows


def checkpoint(path, cap, rows, *, iteration=7):
    with gzip.open(path, 'wt') as stream:
        stream.write(json.dumps({'config': {'max_entries': cap}, 'iteration': iteration}) + '\n')
        for row in rows:
            stream.write(json.dumps(row) + '\n')


def test_all_rows_match_with_only_cap_changed(tmp_path):
    a, b = tmp_path / 'old.gz', tmp_path / 'large.gz'
    rows = [['0' * 32, ['fold', 'call'], [-0.0, 2.0], [3.0, 5.0], 9]]
    checkpoint(a, 10, rows)
    checkpoint(b, 30, rows)
    checked = verify_rows(a, b, expected_entries=1, old_cap=10, new_cap=30)
    assert checked['entries'] == 1
    assert checked['action_slots'] == 2
    assert checked['visits'] == 9
    checkpoint(b, 30, rows, iteration=8)
    with pytest.raises(ValueError, match='header'):
        verify_rows(a, b, expected_entries=1, old_cap=10, new_cap=30)


def test_signed_zero_or_missing_row_cannot_pass(tmp_path):
    a, b = tmp_path / 'old.gz', tmp_path / 'large.gz'
    row = ['0' * 32, ['call'], [-0.0], [2.0], 0]
    checkpoint(a, 10, [row])
    row[2] = [0.0]
    checkpoint(b, 30, [row])
    with pytest.raises(ValueError, match='row bytes'):
        verify_rows(a, b, expected_entries=1, old_cap=10, new_cap=30)
    checkpoint(b, 30, [])
    with pytest.raises(ValueError):
        verify_rows(a, b, expected_entries=1, old_cap=10, new_cap=30)
