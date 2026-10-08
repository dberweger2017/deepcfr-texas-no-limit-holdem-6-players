"""Bounded cancellation fixtures use disposable sleeping children, never training."""
import json
import os
import signal
import subprocess
import sys
from time import time

import pytest
from scripts import hu20_scaling_supervise as supervisor


def environment(monkeypatch):
    monkeypatch.setattr(supervisor, 'identity', lambda: {'fixture': True})
    monkeypatch.setattr(supervisor, 'system', lambda _: 'AC Power')
    monkeypatch.setattr(supervisor, 'swap_bytes', lambda _: 0)
    monkeypatch.setattr(supervisor, 'TERMINATION_GRACE_SECONDS', .2)
    monkeypatch.setattr(supervisor, 'sleep', lambda _: None)


@pytest.mark.parametrize('failure', [signal.SIGINT, signal.SIGTERM, KeyboardInterrupt, RuntimeError])
def test_interrupts_and_monitor_failures_stop_owned_child_and_publish_receipt(monkeypatch, tmp_path, failure):
    environment(monkeypatch)
    children = []; real_popen = subprocess.Popen
    unrelated = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    def spawn(*args, **kwargs):
        child = real_popen(*args, **kwargs); children.append(child); return child
    monkeypatch.setattr(supervisor.subprocess, 'Popen', spawn)
    def monitor(*_, **__):
        if isinstance(failure, type): raise failure('fixture monitor failure')
        os.kill(os.getpid(), failure)
        return ''
    monkeypatch.setattr(supervisor.subprocess, 'check_output', monitor)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    command = [sys.executable, '-c', 'import time; time.sleep(30)']
    jobs = [{'name':'sleeper', 'command':command}, {'name':'must-not-start', 'command':command}]
    out = tmp_path/'attempt'
    try:
        result = supervisor.run(jobs,out,time()+10,disk_gib=.001)
        assert len(children) == 1 and children[0].poll() is not None
        assert unrelated.poll() is None
        with pytest.raises(ProcessLookupError): os.kill(children[0].pid,0)
        assert result['status'] == ('incomplete' if failure is RuntimeError else 'interrupted')
        assert result['attempts'][0]['status'] != 'running'
        assert result['attempts'][0]['finished'] and result['attempts'][0]['exit_code'] != 0
        assert json.loads((out/'campaign.json').read_text()) == result
        assert out.with_name(out.name+'-inventory.json').exists()
        assert handlers == {s: signal.getsignal(s) for s in handlers}
    finally:
        for child in children: supervisor.terminate_child(child)
        supervisor.terminate_child(unrelated)


def test_termination_escalates_for_a_child_that_ignores_term(tmp_path):
    ready = tmp_path/'ready'
    code = "import signal,time,pathlib,sys; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(sys.argv[1]).touch(); time.sleep(30)"
    child = subprocess.Popen([sys.executable,'-c',code,str(ready)],start_new_session=True)
    from time import sleep
    try:
        for _ in range(100):
            if ready.exists(): break
            sleep(.01)
        assert ready.exists()
        old = supervisor.TERMINATION_GRACE_SECONDS
        supervisor.TERMINATION_GRACE_SECONDS = .2
        try: supervisor.terminate_child(child)
        finally: supervisor.TERMINATION_GRACE_SECONDS = old
        assert child.returncode == -signal.SIGKILL
    finally:
        supervisor.terminate_child(child)


def test_resource_refusal_never_spawns_a_child(monkeypatch,tmp_path):
    environment(monkeypatch)
    monkeypatch.setattr(supervisor.subprocess,'Popen',lambda *_args,**_kw: pytest.fail('child started after admission refusal'))
    result=supervisor.run([{'name':'never','command':['unused']}],tmp_path/'refused',time()-1,disk_gib=.001)
    assert result['status']=='incomplete'
    assert result['attempts'][0]['guard_failure']=='Absolute phase/deadline guard'


def test_cleanup_timeout_still_publishes_terminal_receipt(monkeypatch,tmp_path):
    environment(monkeypatch)
    real_popen = subprocess.Popen; real_cleanup = supervisor.terminate_child; children=[]
    def spawn(*args,**kw):
        child=real_popen(*args,**kw); children.append(child); return child
    monkeypatch.setattr(supervisor.subprocess,'Popen',spawn)
    def failed_monitor(*_args,**_kw): raise RuntimeError('monitor failed')
    monkeypatch.setattr(supervisor.subprocess,'check_output',failed_monitor)
    def timeout_cleanup(child): raise subprocess.TimeoutExpired('fixture cleanup',.2)
    monkeypatch.setattr(supervisor,'terminate_child',timeout_cleanup)
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    out=tmp_path/'cleanup-timeout'
    try:
        result=supervisor.run([{'name':'sleeper','command':[sys.executable,'-c','import time; time.sleep(30)']}],out,time()+10,disk_gib=.001)
        assert result['status']=='incomplete'
        assert result['attempts'][0]['status']=='failed'
        assert result['attempts'][0]['pid']==children[0].pid
        assert 'TimeoutExpired' in result['cleanup_failure']
        assert json.loads((out/'campaign.json').read_text())==result
        assert handlers=={s:signal.getsignal(s) for s in handlers}
    finally:
        for child in children: real_cleanup(child)


def test_signal_during_popen_cannot_lose_child_ownership(monkeypatch,tmp_path):
    environment(monkeypatch)
    children=[];real_popen=subprocess.Popen
    def spawn(*args,**kwargs):
        child=real_popen(*args,**kwargs);children.append(child)
        os.kill(os.getpid(),signal.SIGTERM)
        return child
    monkeypatch.setattr(supervisor.subprocess,'Popen',spawn)
    out=tmp_path/'popen-interrupt'
    try:
        result=supervisor.run([{'name':'sleeper','command':[sys.executable,'-c','import time; time.sleep(30)']}],out,time()+10,disk_gib=.001)
        assert result['status']=='interrupted'
        assert result['attempts'][0]['pid']==children[0].pid
        assert children[0].poll() is not None
        assert json.loads((out/'campaign.json').read_text())==result
    finally:
        for child in children: supervisor.terminate_child(child)


def test_successful_child_keeps_complete_receipt(monkeypatch,tmp_path):
    environment(monkeypatch)
    monkeypatch.setattr(supervisor.subprocess,'check_output',lambda *_a,**_k:'')
    result=supervisor.run([{'name':'empty-fixture','command':[sys.executable,'-c','pass']}],tmp_path/'success',time()+10,disk_gib=.001)
    assert result['status']=='complete' and result['attempts'][0]['exit_code']==0
