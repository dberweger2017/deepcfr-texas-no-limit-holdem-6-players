import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts import m4_parallel_takeover as takeover


@pytest.mark.parametrize('exceed_memory', [False, True])
def test_handoff_preserves_existing_worker_and_enforces_combined_limit(tmp_path, monkeypatch, exceed_memory):
    out = tmp_path
    worker_script = out/'fake_worker.py'
    worker_script.write_text('''import json, pathlib, sys, time
out = pathlib.Path(sys.argv[1])
out.mkdir(exist_ok=True)
(out/'started').write_text('yes')
while not (out.parent/'release').exists():
    time.sleep(.02)
(out/'result.json').write_text(json.dumps({'complete': True}))
''')
    parent_script = out/'fake_parent.py'
    parent_script.write_text('''import pathlib, subprocess, sys, time
p = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]], start_new_session=True)
pathlib.Path(sys.argv[3]).write_text(str(p.pid))
p.wait()
pathlib.Path(sys.argv[3]+'.duplicate').write_text('bad')
''')
    pid_file = out/'worker.pid'
    parent = subprocess.Popen([sys.executable, str(parent_script), str(worker_script),
                               str(out/'seed-2026091802'), str(pid_file)])
    original_process = takeover.process
    real_sleep = time.sleep
    worker = None
    second = None
    try:
        deadline = time.monotonic()+5
        while not pid_file.exists():
            assert time.monotonic() < deadline
            real_sleep(.02)
        worker = int(pid_file.read_text())
        (out/'manifest.json').write_text(json.dumps({'source_sha256': 'same'}))
        (out/'status.json').write_text('{}')
        (out/'seed-2026091802.json').write_text(json.dumps({'pid': worker, 'status': 'running'}))
        monkeypatch.setattr(takeover, 'source_fingerprint', lambda: 'same')
        monkeypatch.setattr(takeover.shutil, 'disk_usage', lambda p: SimpleNamespace(free=30*takeover.GIB))
        monkeypatch.setattr(takeover, 'train_command', lambda p, s: [sys.executable, str(worker_script), str(p/f'seed-{s}')])
        def observed(pid):
            row = original_process(pid)
            if row and pid == parent.pid:
                row['command'] = '-m scripts.local_fullgame --plan test'
            if row and pid == worker:
                row['command'] = '--seed 2026091802'
            if row and exceed_memory:
                row['rss'] = 5*takeover.GIB
            return row
        monkeypatch.setattr(takeover, 'process', observed)
        def short_sleep(seconds):
            if not exceed_memory and (out/'parallel-handoff.json').exists():
                record = json.loads((out/'parallel-handoff.json').read_text())
                if record['state'] == 'transferred':
                    assert original_process(worker) is not None
                    assert original_process(record['second_worker_pid']) is not None
                    (out/'release').touch()
            real_sleep(min(seconds, .05))
        monkeypatch.setattr(takeover.time, 'sleep', short_sleep)
        def verify(command, output, **kwargs):
            destination = Path(command[-1])
            destination.mkdir()
            (destination/'final-summary.json').write_text(json.dumps({'competence_check_passed': False}))
            return {'seconds': 1}
        monkeypatch.setattr(takeover, 'guarded_run', verify)
        if exceed_memory:
            with pytest.raises(RuntimeError, match='combined_process_memory_limit'):
                takeover.run(out, parent.pid, worker)
        else:
            takeover.run(out, parent.pid, worker)
        record = json.loads((out/'parallel-handoff.json').read_text())
        second = record['second_worker_pid']
        assert record['existing_worker_pid'] == worker
        assert not Path(str(pid_file)+'.duplicate').exists()
        assert json.loads((out/'status.json').read_text())['state'] == ('failed' if exceed_memory else 'complete')
        parent.wait(timeout=5)
        assert parent.returncode == -signal.SIGKILL
        if exceed_memory:
            for pid in (worker, second):
                row = original_process(pid)
                assert row is None or 'Z' in row['state']
    finally:
        parent.kill() if parent.poll() is None else None
        parent.wait()
        if (out/'parallel-handoff.json').exists():
            second = json.loads((out/'parallel-handoff.json').read_text()).get('second_worker_pid')
        for pid in (worker, second):
            row = original_process(pid) if pid else None
            if row and 'Z' not in row['state']:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
