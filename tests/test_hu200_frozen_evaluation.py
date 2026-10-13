"""Campaign admission, fixed sizing and outcome barrier regressions."""
from datetime import datetime, timezone
import pytest

from scripts import run_hu200_frozen_evaluation as campaign


def cost(seconds):
    return {'primary': {'load_or_validation_seconds': 0, 'play_seconds': seconds},
            'repeat': {'load_or_validation_seconds': 0, 'play_seconds': 0},
            'audit_seconds': 0, 'blocks': 1}


def test_fixed_madrid_window():
    assert [datetime.fromtimestamp(x, timezone.utc).isoformat() for x in
            (campaign.OPEN, campaign.LAST_START, campaign.SCIENCE_END, campaign.END)] == [
        '2026-10-13T23:00:00+00:00', '2026-10-14T03:00:00+00:00',
        '2026-10-14T07:30:00+00:00', '2026-10-14T08:00:00+00:00']


def test_reduction_orders_curve_before_primary_before_panel():
    costs = {'primary': cost(1), 'curve': cost(1),
             'panel': {o: cost(1) for o in campaign.direct.PANEL_OPPONENTS}}
    # These fixed budgets straddle the successive authorized configurations.
    expected = [(262144, 131072, 16384), (262144, 32768, 16384),
                (262144, 0, 16384), (131072, 0, 16384), (131072, 0, 8192)]
    for remaining, counts in zip((700000, 520000, 480000, 320000, 240000), expected, strict=True):
        selected, options = campaign.choose_counts(costs, remaining)
        assert tuple(selected[k] for k in ('primary', 'curve', 'panel')) == counts
        assert all(not option['fits'] for option in options[:-1])
    with pytest.raises(ValueError, match='cannot fit'):
        campaign.choose_counts(costs, 1)


def test_readout_refuses_without_global_reproduction_receipt(tmp_path):
    campaign.put(tmp_path/'all-final-verified.json', {'status': 'incomplete'})
    with pytest.raises(ValueError, match='All final hands'):
        campaign.readout(tmp_path, {})
    assert not (tmp_path/'summary.json').exists()


def test_missing_release_statement_cannot_admit(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'time', lambda: campaign.OPEN+1)
    monkeypatch.setattr(campaign, 'gh', lambda *args: '{"body":"M4 reserved; ETA 01:00","comments":[]}')
    with pytest.raises(ValueError, match='Explicit live'):
        campaign.admission({'release_statement': 'M4 released for compute', 'release_confirmed': True}, tmp_path)


def test_pr265_process_blocks_admission_without_signaling(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'time', lambda: campaign.OPEN+1)
    monkeypatch.setattr(campaign, 'gh', lambda *args: '{"body":"M4 released for compute","comments":[]}')
    class Process:
        info = {'pid': 265, 'ppid': 1, 'create_time': 100,
                'cmdline': ['python', 'planning/hu200-k50-frozen/campaign.py', 'audit']}
    monkeypatch.setattr(campaign.psutil, 'process_iter', lambda *args, **kwargs: [Process()])
    with pytest.raises(ValueError, match='still running'):
        campaign.admission({'release_statement': 'M4 released for compute', 'release_confirmed': True}, tmp_path)
    assert campaign.read(tmp_path/'admission.json')['occupancy'][0]['pid'] == 265


def test_shared_interpreter_does_not_imply_pr265_ownership(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'time', lambda: campaign.OPEN+1)
    monkeypatch.setattr(campaign, 'gh', lambda *args: '{"body":"M4 released for compute","comments":[]}')
    class Process:
        info = {'pid': 987654, 'ppid': 1, 'create_time': 100, 'cmdline': [
            '/Users/dberweger/Local/hu200-k50-frozen-20261013/.venv/bin/python',
            '-m', 'scripts.run_hu200_frozen_evaluation', 'run']}
    monkeypatch.setattr(campaign.psutil, 'process_iter', lambda *args, **kwargs: [Process()])
    campaign.admission({'release_statement': 'M4 released for compute', 'release_confirmed': True}, tmp_path)
    assert campaign.read(tmp_path/'admission.json')['occupancy'] == []
