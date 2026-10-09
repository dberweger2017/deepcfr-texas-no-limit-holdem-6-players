"""Critical campaign guard, predeclared inference and storage admission checks."""
import json
import os
import subprocess
import sys

import pytest

from scripts import hu100_qualification_guard as guard
from scripts import run_hu100_seed_qualification as campaign
from scripts.report_hu100_seed_qualification import effect


def test_new_growth_limit_keeps_fixed_baseline_and_inherited_other_guards():
    sample = {'pressure_level': 1, 'free_percent': 86, 'swap_bytes': 1_000_000_000,
        'ac': True, 'disk_free_bytes': 120*guard.GIB}
    assert guard.violation(sample, 1_000_000_000, 0) is None
    assert guard.violation({**sample, 'swap_bytes': 4_000_000_000}, 1_000_000_000, 0) is None
    assert guard.violation({**sample, 'swap_bytes': 4_000_000_001}, 1_000_000_000, 0) == 'swap growth'
    for delta, expected in (({'pressure_level': 2}, 'system pressure/headroom'),
            ({'free_percent': 14}, 'system pressure/headroom'), ({'ac': False}, 'AC power'),
            ({'disk_free_bytes': guard.inherited.DISK_FLOOR}, 'disk floor')):
        assert guard.violation({**sample, **delta}, 1_000_000_000, 0) == expected
    assert guard.violation(sample, 1_000_000_000, 8*guard.GIB) == 'hard whole-family RSS'
    assert guard.inherited.SWAP_GROWTH == 512*1024**2


def test_cleanup_kills_owned_descendant_after_direct_wrapper_exits(tmp_path):
    marker = tmp_path/'child-ready'
    code = '''import os,signal,time,sys
pid=os.fork()
if pid:
    while not os.path.exists(sys.argv[1]): time.sleep(.01)
    sys.exit(0)
signal.signal(signal.SIGTERM,signal.SIG_IGN)
open(sys.argv[1],"w").write(str(os.getpid()))
while True: time.sleep(.1)
'''
    child = subprocess.Popen([sys.executable, '-c', code, str(marker)], start_new_session=True)
    try:
        child.wait(timeout=5)
        assert child.poll() == 0
        assert guard.group_members(child.pid)
        guard.stop_group(child)
        assert guard.group_members(child.pid) == []
    finally:
        if guard.group_members(child.pid):
            os.killpg(child.pid, 9)


def test_absolute_budget_and_closeout_reserve_refuse_child_before_start(tmp_path):
    c = object.__new__(guard.Campaign)
    c.out = tmp_path
    c.remaining = lambda: 1900
    c.tool_quote, c.panel_quote = 200, 300
    c.report_quote = 100
    with pytest.raises(guard.CapacityStop):
        c.run('terminal-export', ['must-not-launch'])
    with pytest.raises(guard.CapacityStop):
        c.run('final-play', ['must-not-launch'])
    assert not (tmp_path/'operations').exists()


def test_multiplicity_families_and_practical_threshold_are_distinct():
    values = [10, 20, 0, 30, 15, 25, 10, 20]*16
    growth, translation = effect(values, .05/6), effect(values, .05/3)
    assert growth['adjusted']['alpha'] == .05/6
    assert translation['adjusted']['alpha'] == .05/3
    assert growth['adjusted']['interval'][0] < translation['adjusted']['interval'][0]
    assert growth['label'] == 'improvement'
    assert growth['practically_supported'] == (growth['adjusted']['interval'][0] > 10)
    assert effect([-v for v in values], .05/6)['label'] == 'decline'


def test_resume_targets_are_total_nodes_and_fresh_seed_is_predeclared(tmp_path, monkeypatch):
    previous = tmp_path/'previous.gz'
    previous.write_bytes(b'recovery-state')
    command = list(map(str, campaign.fixture_command(tmp_path/'new', 2026100901, 1_000_000_000, resume=previous)))
    assert command[command.index('--nodes')+1] == '1000000000'
    assert command[command.index('--resume')+1] == str(previous)
    assert command[command.index('--average-rule')+1] == 'opponent-sampled'
    assert command[command.index('--max-entries')+1] == '57658644'
    assert campaign.SEEDS == (2026100601, 2026100901, 2026100902)


def test_freshness_compares_physical_deals_including_207_and_new_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'OUT', tmp_path)
    model = {'name': 'fixture', 'path': 'fixture', 'sha256': '0'*64,
        'format': 'holdem-hu100-stored-cfr-average-research-v1'}
    campaign.freshness({'model': model})
    r = json.loads((tmp_path/'freshness.json').read_text())
    assert r['all_pairwise_disjoint']
    assert str(2026100810012) in r['roots']
    assert str(2026100820521) in r['roots']
    assert str(2026100820512) in r['roots']
    assert str(campaign.FINAL_ROOT) in r['roots']
