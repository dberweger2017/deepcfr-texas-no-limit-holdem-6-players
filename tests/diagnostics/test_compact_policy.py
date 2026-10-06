"""Compact policy storage returns exactly what the dict-based loader held."""

import gzip
import json
from pathlib import Path
import subprocess

import pytest

from src.diagnostics.compact_policy import CompactBuilder

BINARY = Path("native/hu20-trainer/target/release/hu20-trainer")


def test_builder_sorts_finds_and_keeps_exact_floats():
    rows = {"ff" * 16: (("fold", "call"), [0.1, 0.9], 3, False),
            "00" * 15 + "01": (("check", "min", "pot"), [1 / 3, 1 / 3, 1 / 3], 0, True),
            "ab" * 8 + "00" * 8: (("fold", "call"), [0.30000000000000004, 0.7], 12, False)}  # trailing NUL bytes
    builder = CompactBuilder()
    for key, (names, p, visits, flag) in rows.items():  # deliberately not in key order
        builder.add(key, names, p, visits, flag)
    entries, flagged, counts = builder.build()
    assert len(entries) == 3 and list(entries) == sorted(rows)
    for key, (names, p, visits, flag) in rows.items():
        assert entries[key] == (names, tuple(p)) and entries.get(key) == (names, tuple(p))
        assert counts.get(key) == visits and (key in flagged) == flag
    assert set(flagged) == {"00" * 15 + "01"} and len(flagged) == 1
    for missing in ("ee" * 16, "zz" * 16, "ab", 5, None):
        assert entries.get(missing) is None and missing not in entries and missing not in flagged
    with pytest.raises(KeyError):
        entries["ee" * 16]


def test_builder_rejects_duplicate_keys():
    builder = CompactBuilder()
    builder.add("aa" * 16, ("fold", "call"), [0.5, 0.5], 1, False)
    builder.add("aa" * 16, ("fold", "call"), [0.5, 0.5], 1, False)
    with pytest.raises(ValueError, match="Duplicate"):
        builder.build()


@pytest.mark.skipif(not BINARY.exists(), reason="build native/hu20-trainer first")
@pytest.mark.parametrize("flags", [[], ["--average-rule", "opponent-sampled"]])
def test_diagnostic_average_matches_a_reference_parse(tmp_path, flags):
    from src.diagnostics.cfr_average import DiagnosticAverage
    from src.diagnostics.saved_hu20 import file_hash
    checkpoint, average = tmp_path / "c.json.gz", tmp_path / "a.jsonl.gz"
    subprocess.run([str(BINARY), "train", "--nodes", "300000", "--seed", "8", *flags, "--out", str(checkpoint)],
                   check=True, capture_output=True)
    subprocess.run([str(BINARY), "export", str(checkpoint), "--average", str(average)], check=True, capture_output=True)
    reference, zero, visits = {}, set(), {}
    with gzip.open(average, "rt") as source:
        source.readline()
        for line in source:
            key, names, p, total, count = json.loads(line)
            reference[key], visits[key] = (tuple(names), tuple(p)), count
            if not total:
                zero.add(key)
    policy = DiagnosticAverage(average, file_hash(average))
    assert len(policy.entries) == len(reference) and set(policy.entries) == set(reference)
    assert all(policy.entries.get(k) == v for k, v in reference.items())
    assert set(policy.zero_mass) == zero and dict(policy.visits) == visits
    assert policy.description["entries"] == len(reference) and policy.description["zero_mass_entries"] == len(zero)
