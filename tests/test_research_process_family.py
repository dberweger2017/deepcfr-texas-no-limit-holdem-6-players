import pytest
import psutil
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
