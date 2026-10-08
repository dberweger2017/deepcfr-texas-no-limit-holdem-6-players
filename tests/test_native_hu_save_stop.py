"""Bounded native CLI stop-during-save fixture; no campaign or external worker use."""
import json
from pathlib import Path
import subprocess
import threading
import time

import pytest

BINARY=Path('native/hu20-trainer/target/release/hu20-trainer')


def test_stop_during_atomic_save_does_not_serialize_the_same_state_again(tmp_path):
    if not BINARY.exists(): pytest.skip('build the native trainer')
    stop=tmp_path/'stop.json'; telemetry=tmp_path/'checkpoints.jsonl'
    first=tmp_path/'checkpoint-100000.json.tmp'
    observed=threading.Event(); done=threading.Event()
    def request():
        deadline=time.monotonic()+20
        while not done.is_set() and time.monotonic()<deadline:
            if first.exists():
                stop.write_text('{"reason":"fixture stop during first save"}\n')
                observed.set(); return
            time.sleep(.0005)
    thread=threading.Thread(target=request); thread.start()
    try:
        result=subprocess.run([str(BINARY),'train','--seed','123','--average-rule','opponent-sampled',
            '--nodes','1000000','--milestones','100000','--max-seconds','15','--out',
            str(tmp_path/'checkpoint-{nodes}.json.gz'),'--stop-file',str(stop),'--telemetry',str(telemetry)],
            capture_output=True,text=True,timeout=25,env={**__import__('os').environ,'RAYON_NUM_THREADS':'1'})
    finally:
        done.set(); thread.join(timeout=2)
    assert observed.is_set(),'fixture never observed the first atomic save'
    assert result.returncode==3,result.stderr
    rows=[json.loads(line) for line in telemetry.read_text().splitlines()]
    assert len(rows)==1 and rows[0]['requested_nodes']==100000
    assert (tmp_path/'checkpoint-100000.json.gz').is_file()
    assert not (tmp_path/'checkpoint-1000000.json.gz').exists()
    assert not (tmp_path/'checkpoint-1000000.json.tmp').exists()


def test_preexisting_stop_request_refuses_a_fresh_native_attempt(tmp_path):
    if not BINARY.exists(): pytest.skip('build the native trainer')
    stop=tmp_path/'stop.json'; stop.write_text('{}\n')
    out=tmp_path/'never.json.gz'
    result=subprocess.run([str(BINARY),'train','--nodes','1','--stop-file',str(stop),'--out',str(out)],
        capture_output=True,text=True,timeout=5)
    assert result.returncode!=0 and 'fresh stop-request path' in result.stderr
    assert not out.exists()
