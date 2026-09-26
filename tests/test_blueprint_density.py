"""The large-checkpoint audit accumulates exact counts with bounded memory."""

import gzip
import json
from hashlib import sha256

from scripts.audit_blueprint_density import audit
from src.blueprint.lookup import BUTTON_ZERO_CHECKPOINTS


def test_streamed_density_matches_small_independent_counts(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json.gz"
    header = {
        "kind": "training", "checkpoint_format": "jsonl-v2",
        "abstraction": "blueprint-abstraction-v1", "iteration": 7,
        "table": {"button": 0, "stacks": [10_000] * 6}, "config": {},
    }
    rows = [
        ["0" * 32, ["fold", "check"], [0, 0], [0, 0], 0],
        ["1" * 32, ["fold", "check"], [1, 0], [0, 0], 1],
        ["2" * 32, ["fold", "call"], [0.5, 0.5], [0, 0], 2],
        ["3" * 32, ["check", "pot"], [0, 2], [0, 0], 20],
    ]
    with gzip.open(path, "wt", encoding="utf-8") as output:
        output.write(json.dumps(header) + "\n")
        for row in rows:
            output.write(json.dumps(row) + "\n")
    digest = sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(BUTTON_ZERO_CHECKPOINTS, digest, 7)
    result = audit(path, digest)
    assert result["entries"] == 4
    assert result["visits_total"] == 23
    assert result["visit_histogram"] == {"0": 1, "1": 1, "2": 1, "20": 1}
    assert result["fraction_one_visit"] == 0.25
    assert result["fraction_at_most_two"] == 0.75
    assert result["fraction_at_least_twenty"] == 0.25
    assert result["fold_and_check_entries"] == 2
    assert result["near_pure_current_entries"] == 2
