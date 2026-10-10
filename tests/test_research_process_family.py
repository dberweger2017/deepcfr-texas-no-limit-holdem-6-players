import pytest
import psutil
import subprocess
import sys
from scripts.research_process_family import owned_processes

class Process:
    def __init__(self,pid,created):self.pid,self.created=pid,created
    def create_time(self):return self.created

def test_only_owned_descendants_and_reparented_identities():
    times={1:1,2:2,3:3,99:99};calls=[]
    def factory(pid):
        calls.append(pid)
        if pid==99:raise psutil.AccessDenied(pid)
        return Process(pid,times[pid])
    known={}
    assert [p.pid for p in owned_processes(known,root=1,parent_rows=[(2,1),(3,2),(99,0)],process_factory=factory)]==[1,2,3]
    assert 99 not in calls
    # A new session does not alter ownership; after reparenting it stays owned.
    assert [p.pid for p in owned_processes(known,root=1,parent_rows=[(2,1),(3,0),(99,0)],process_factory=factory)]==[1,2,3]
    times[3]=30
    assert [p.pid for p in owned_processes(known,root=1,parent_rows=[(2,1),(3,0)],process_factory=factory)]==[1,2]
    assert known=={2:2}

def test_missing_owned_identity_is_pruned_but_access_denied_stops():
    def factory(pid):
        if pid==2:raise psutil.NoSuchProcess(pid)
        if pid==3:raise psutil.AccessDenied(pid)
        return Process(pid,pid)
    known={2:2}
    assert [p.pid for p in owned_processes(known,root=1,parent_rows=[],process_factory=factory)]==[1]
    assert known=={}
    with pytest.raises(psutil.AccessDenied):owned_processes(known,root=1,parent_rows=[(3,1)],process_factory=factory)


def test_new_descendants_of_creation_verified_orphan_are_owned():
    times={1:1,3:3,4:4}
    def factory(pid):return Process(pid,times[pid])
    known={3:3}
    assert [p.pid for p in owned_processes(known,root=1,parent_rows=[(3,0),(4,3)],process_factory=factory)]==[1,3,4]
    assert known=={3:3,4:4}
    # A reused orphan PID must not pull its unrelated new children into ownership.
    times[3]=30;known={3:3}
    assert [p.pid for p in owned_processes(known,root=1,parent_rows=[(3,0),(4,3)],process_factory=factory)]==[1]
    assert known=={}


def test_discovery_starts_no_helper_and_rechecks_unreadable_owned_identity(monkeypatch):
    class Entry:
        def __init__(self,pid,parent,created):
            self.info={'pid':pid,'ppid':parent,'create_time':created}
    def iterator(attrs):
        assert attrs==['pid','ppid','create_time']
        return [Entry(1,0,1),Entry(2,1,None),Entry(99,None,None)]
    monkeypatch.setattr(psutil,'process_iter',iterator)
    def forbidden(*args,**kwargs):raise AssertionError('Discovery spawned a helper')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    def factory(pid):
        if pid==2:raise psutil.AccessDenied(pid)
        if pid==99:raise AssertionError('Unowned identity inspected')
        return Process(pid,pid)
    with pytest.raises(psutil.AccessDenied):
        owned_processes({},root=1,process_factory=factory)
    # A retained child stays owned even when snapshot ancestry is unreadable.
    monkeypatch.setattr(psutil,'process_iter',lambda attrs:[Entry(2,None,None)])
    with pytest.raises(psutil.AccessDenied):
        owned_processes({2:2},root=1,process_factory=factory)


def test_discovery_rejects_pid_reused_after_snapshot(monkeypatch):
    class Entry:
        info={'pid':2,'ppid':1,'create_time':2}
    monkeypatch.setattr(psutil,'process_iter',lambda attrs:[Entry()])
    with pytest.raises(RuntimeError,match='PID changed'):
        owned_processes({},root=1,process_factory=lambda pid:Process(pid,20 if pid==2 else pid))


def test_owned_workload_rss_access_denied_still_stops(monkeypatch):
    from scripts import guard_hu20_equity_scoring as guard
    class Unreadable:
        def memory_info(self):raise psutil.AccessDenied(2)
    monkeypatch.setattr(guard,'owned_processes',lambda known:[Unreadable()])
    with pytest.raises(psutil.AccessDenied):guard.family_rss({2:2})


@pytest.mark.skipif(sys.platform!='darwin',reason='macOS setuid ps regression')
def test_ps_child_discovery_to_exit_starts_no_sampling_helper(monkeypatch):
    # The explicit workload child may itself be privileged; discovery must not
    # launch another ps, and unreadable workload RSS must still fail closed.
    child=subprocess.Popen(['/bin/ps','-o','rss=','-p','1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    known={child.pid:psutil.Process(child.pid).create_time()}
    def forbidden(*args,**kwargs):raise AssertionError('Discovery spawned a helper')
    try:
        monkeypatch.setattr(subprocess,'Popen',forbidden)
        for _ in range(20):
            owned_processes(known)
            if child.poll() is not None:break
        child.wait(timeout=5)
        assert [p.pid for p in owned_processes(known) if p.pid==child.pid]==[]
        assert child.returncode==0
    finally:
        if child.poll() is None:child.kill()
        child.communicate(timeout=5)
