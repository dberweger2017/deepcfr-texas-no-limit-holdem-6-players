"""A verified successor never grants deletion outside this campaign."""
import json
from pathlib import Path
import pytest
from scripts.run_hu200_capacity_training import pin, retire


def test_retirement_requires_verified_unchanged_recovery_and_owned_path(tmp_path):
    base = tmp_path / 'campaign'
    root = base / 'results'
    root.mkdir(parents=True)
    old, new = root / 'old.gz', root / 'new.gz'
    old.write_bytes(b'old checkpoint')
    new.write_bytes(b'new checkpoint')
    receipt = root / 'audit.json'
    expected = pin(old)
    receipt.write_text(json.dumps(dict(status='verified', audit=dict(status='verified', all_nodes_verified=5),
        native=dict(diagnostics=dict(entries=5)), files=dict(checkpoint=pin(new)))))
    new.write_bytes(b'changed successor')
    with pytest.raises(ValueError, match='Replacement recovery changed'):
        retire(base, old, expected, receipt)
    assert old.exists()
    receipt.write_text(json.dumps(dict(status='verified', audit=dict(status='verified', all_nodes_verified=5),
        native=dict(diagnostics=dict(entries=5)), files=dict(checkpoint=pin(new)))))
    other = tmp_path / 'other-pr.gz'
    other.write_bytes(b'protected')
    with pytest.raises(ValueError, match='outside'):
        retire(base, other, pin(other), receipt)
    retire(base, old, expected, receipt)
    assert not old.exists() and other.exists() and new.exists()
    assert json.loads((base / 'retirements.jsonl').read_text())['removed']['sha256'] == expected['sha256']
