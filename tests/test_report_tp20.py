"""Primary uncertainty treats training seeds and rotations as paired blocks."""

from statistics import variance
from time import time

import pytest

from scripts.evaluate_tp20 import run
from scripts.report_tp20 import read_run, stratified
from tests.test_blueprint_tp20 import plan


def test_stratified_interval_does_not_count_rotations_or_seeds_as_independent():
    groups = [[-3,-1,1,3]*8,[97,99,101,103]*8]
    result = stratified(groups)
    assert result["bb_per_100"] == 50
    assert result["blocks"] == 64
    from scipy.stats import t
    from math import sqrt
    expected_radius = t.ppf(.975,62)*sqrt(variance(groups[0])/32/2)
    assert result["interval"] == pytest.approx([50-expected_radius,50+expected_radius])
    assert stratified([[1]*32,[3]*32])["interval"] == [2,2]


def test_reference_audit_checks_every_hand_and_schedule(tmp_path):
    p = plan(); p["confirmation"]["blocks_per_lineup"] = 3
    out = tmp_path/"run"
    result = run(p,tmp_path,"uniform","tp20_uniform","confirmation",out,time()+30)
    assert result["status"] == "complete"
    failures = []
    item = read_run(out,p,"confirmation","tp20_uniform","uniform",failures)
    assert not failures and len(item["values"]) == 3
    path = out/"hands.jsonl"
    path.write_text(path.read_text().replace('"status": "completed"','"status": "failed"',1))
    failures = []
    broken = read_run(out,p,"confirmation","tp20_uniform","uniform",failures)
    assert failures and broken["values"] is None
    assert broken["hand_attempts"] == 9
