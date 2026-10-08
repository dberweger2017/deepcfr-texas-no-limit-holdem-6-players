"""An unstarted last panel can fill a gap without repeating or extending science."""

import json
from pathlib import Path

import pytest

from src.policies.files import file_hash


def fixture(tmp_path, monkeypatch):
    import scripts.continue_native_hu100_learning_curves as c
    import scripts.run_native_hu100_learning_curves as original
    run = tmp_path / 'run'; run.mkdir()
    monkeypatch.setattr(c, 'ROOT', tmp_path); monkeypatch.setattr(c, 'RUN', run)
    monkeypatch.setattr(c, 'admission', lambda *args: None)
    monkeypatch.setattr(original, 'worker_identity', lambda: 'Apple M4')
    monkeypatch.setattr(original, 'authorized_config', lambda *args: c.CONFIG_HASH)
    def call(args, **kwargs):
        if args[:2] == ['git', 'rev-parse']: return c.SOURCE
        if args[0] == 'git': return ''
        return '999999 1\n'
    monkeypatch.setattr(c.subprocess, 'check_output', call)
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    state = {'phase': 'failed', 'terminal': True, 'worker_pid': 100,
             'failure': "ValueError('Deadline/system headroom admission failure')", 'deadline': 1234567890}
    write(run / 'state.json', state)
    write(run / 'guard/campaign.json', {'status': 'incomplete', 'failure': None,
          'attempts': [{'guard_failure': None}], 'identity': {'pid': 101}})
    write(run / 'frozen-final.json', {'blocks_per_opponent': 2048, 'final_hands': 122880,
          'source': c.SOURCE, 'deadline': state['deadline']})
    monkeypatch.setattr(c, 'FREEZE_HASH', file_hash(run / 'frozen-final.json'))
    for nodes in (100691, 1001382, 5001210, 10001922):
        write(run / f'final/{nodes}/complete.json', {'status': 'complete'})
        write(run / f'final-{nodes}-audit.json', {'status': 'verified'})
        write(run / f'final-reproduction/{nodes}/complete.json', {'reproduced_all_hands_and_decisions': True})
    return c, run


def test_continuation_rejects_any_already_attempted_last_panel(tmp_path, monkeypatch):
    c, run = fixture(tmp_path, monkeypatch)
    assert c.validate()[0]['deadline'] == 1234567890
    (run / 'final/11042440').mkdir()
    with pytest.raises(ValueError, match='already attempted'):
        c.validate()


def test_continuation_preserves_deadline_state_and_durable_single_claim(tmp_path, monkeypatch):
    c, run = fixture(tmp_path, monkeypatch)
    review = tmp_path / 'review.json'
    review.write_text(json.dumps({'status': 'passed', 'continuation_script_sha256': file_hash(Path(c.__file__))}))
    before = (run / 'state.json').read_bytes()
    def supervise(jobs, out, deadline, **limits):
        assert deadline == 1234567890
        assert limits['rss_gib'] == 10 and limits['disk_gib'] == 15.5
        assert limits['swap_gib'] == .5 and limits['system_memory_guard'] and limits['require_ac']
        out.mkdir(); (out / 'campaign.json').write_text('{}')
        assert '--worker' in jobs[0]['command']
        return {'status': 'complete'}
    monkeypatch.setattr(c, 'supervise', supervise)
    assert c.launch(review)
    assert (run / 'state.json').read_bytes() == before
    with pytest.raises(FileExistsError):
        c.launch(review)
