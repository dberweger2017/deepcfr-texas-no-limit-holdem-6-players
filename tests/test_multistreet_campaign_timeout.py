"""Reject invalid time allowances before preparing an experiment."""

import json
from pathlib import Path

import pytest

from scripts.check_multistreet_campaign import run


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_reference_timeout_does_not_create_outputs(tmp_path, seconds):
    plan = json.loads(Path("configs/holdem/multistreet-campaign.json").read_text())
    out, cache = tmp_path / "out", tmp_path / "cache"
    with pytest.raises(ValueError, match="positive and finite"):
        run(plan, out, cache, deadline_seconds=seconds)
    assert not out.exists()
    assert not cache.exists()
