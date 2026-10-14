from scripts.guard_m4_research import GIB, sustained


def sample(at=0, **changes):
    return dict({'monotonic': at, 'family_rss_bytes': 0, 'pressure_level': 1,
                 'free_percent': 30, 'swap_bytes': 0, 'disk_free_bytes': 20*GIB}, **changes)


def test_transient_and_sustained_each_memory_guard():
    for change, key in [({'family_rss_bytes':13*GIB+1},'family_rss'),
                         ({'pressure_level':4},'critical_pressure'),
                         ({'free_percent':7},'free_memory'),
                         ({'swap_bytes':4_000_000_001},'swap_growth')]:
        held = {}
        assert sustained(sample(10, **change), sample(), held)[1] == []
        assert sustained(sample(69, **change), sample(), held)[1] == []
        assert sustained(sample(70, **change), sample(), held)[1] == [key]
        assert sustained(sample(71), sample(), held)[1] == []
        assert held == {}
        assert sustained(sample(72, **change), sample(), held)[1] == []


def test_warning_thresholds_and_swap_baseline():
    base = sample(swap_bytes=2_000_000_000)
    reading = sample(90, family_rss_bytes=13*GIB, pressure_level=2, free_percent=8,
                     swap_bytes=6_000_000_000, disk_free_bytes=16*GIB)
    assert not any(sustained(reading, base, {})[0].values())


def test_disk_is_immediate_and_breaches_are_independent():
    held = {}
    assert sustained(sample(0, free_percent=7), sample(), held)[1] == []
    assert sustained(sample(59, pressure_level=4), sample(), held)[1] == []
    assert sustained(sample(60, pressure_level=4, disk_free_bytes=16*GIB-1), sample(), held)[1] == ['free_disk']


def test_privileged_helper_does_not_break_family(monkeypatch):
    import os
    from types import SimpleNamespace
    from scripts import guard_m4_research as guard
    real_root = guard.psutil.Process()
    monkeypatch.setattr(guard.psutil, 'process_iter', lambda *a, **k: iter([
        SimpleNamespace(info={'pid':99112233,'ppid':None,'create_time':None})]))
    monkeypatch.setattr(guard.psutil, 'Process', lambda pid=None: real_root)
    _, readings, denied = guard.family({})
    assert denied == [99112233]
    assert [row['pid'] for row in readings] == [os.getpid()]


def test_reused_pid_cannot_adopt_unowned_children(monkeypatch):
    from types import SimpleNamespace
    from scripts import guard_m4_research as guard
    real_root = guard.psutil.Process()
    reused = SimpleNamespace(create_time=lambda:2)
    monkeypatch.setattr(guard.psutil, 'process_iter', lambda *a, **k: iter([
        SimpleNamespace(info={'pid':99112234,'ppid':99112233,'create_time':3})]))
    monkeypatch.setattr(guard.psutil, 'Process', lambda pid=None: reused if pid==99112233 else real_root)
    known = {99112233:1}
    guard.family(known)
    assert not known


def test_launch_identity_requires_own_parent(monkeypatch):
    import os
    import pytest
    from types import SimpleNamespace
    from scripts import guard_m4_research as guard
    known = {}
    process = SimpleNamespace(ppid=lambda:-1,create_time=lambda:17.0)
    monkeypatch.setattr(guard.psutil,'Process',lambda pid:process)
    with pytest.raises(RuntimeError,match='identity changed'):
        guard.remember_child(99112233,known)
    assert not known
    process.ppid = os.getpid
    guard.remember_child(99112233,known)
    assert known == {99112233:17.0}


def test_deadline_terminates_launched_family_and_records_stop(tmp_path,monkeypatch):
    import json
    import sys
    import time
    import pytest
    from scripts import guard_m4_research as guard
    real_output=guard.subprocess.check_output
    monkeypatch.setattr(guard.subprocess,'check_output',lambda cmd,**kw:
        'Apple M4\n' if cmd[:1]==['sysctl'] else real_output(cmd,**kw))
    calls=[]
    def host(path):
        calls.append(path)
        return dict(sample(time.monotonic()),at=0 if len(calls)<=2 else 10)
    monkeypatch.setattr(guard,'host',host)
    real_sleep=time.sleep
    monkeypatch.setattr(guard.time,'sleep',lambda seconds:real_sleep(min(seconds,.1)))
    deadline=10
    monkeypatch.setattr(sys,'argv',['guard','--stage',str(tmp_path),'--name','deadline',
        '--deadline',str(deadline),'--',sys.executable,'-c','import time; time.sleep(60)'])
    with pytest.raises(SystemExit) as stopped:
        guard.main()
    assert stopped.value.code==3
    receipt=json.loads((tmp_path/'operations/deadline/receipt.json').read_text())
    assert receipt['reasons']==['campaign_deadline'] and receipt['returncode'] is not None
    readings=[json.loads(s) for s in (tmp_path/'operations/deadline/resources.jsonl').read_text().splitlines()]
    assert not any(any(r['breaches'].values()) for r in readings)
    assert readings[-1]['stop']==['campaign_deadline']
