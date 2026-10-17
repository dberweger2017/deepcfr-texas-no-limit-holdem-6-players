"""Exercise the production dispatcher without consuming final deals."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
import pytest

from scripts.evaluate_hu200_river_final import execute_pipeline


def test_search_storage_overlap_is_bounded_and_every_job_ships_once():
    lock=threading.Lock();active=set();resident=set();completed=[];overlap=[];peaks=[];shipping=[False]
    def produce(job):
        with lock:
            active.add(job);resident.add(job);peaks.append((len(active),len(resident)))
            overlap.append(shipping[0])
        time.sleep(.005)
        with lock:active.remove(job)
        return job
    def ship(panel,index,summary):
        job=(panel,index);assert summary==job
        with lock:overlap.append(bool(active));shipping[0]=True
        time.sleep(.015)
        with lock:resident.remove(job);shipping[0]=False
        return job
    def progress(job,receipt):assert receipt==job;completed.append(job)
    jobs=[('fixture',i) for i in range(24)]
    with ThreadPoolExecutor(max_workers=3) as search,ThreadPoolExecutor(max_workers=1) as storage:
        execute_pipeline(iter(jobs),search,storage,produce,ship,lambda:3,lambda:None,progress)
    assert sorted(completed)==jobs and len(set(completed))==24
    assert max(n for n,_ in peaks)<=3 and max(n for _,n in peaks)<=6
    assert any(overlap) and not resident


def test_failed_search_is_not_resubmitted_or_shipped():
    attempts=[];shipped=[]
    def produce(job):attempts.append(job);raise ValueError('fixture accounting failure')
    def ship(*args):shipped.append(args)
    with ThreadPoolExecutor(max_workers=3) as search,ThreadPoolExecutor(max_workers=1) as storage:
        with pytest.raises(ValueError,match='accounting'):
            execute_pipeline(iter([('fixture',i) for i in range(9)]),search,storage,produce,ship,lambda:3,lambda:None,lambda *a:None)
    assert len(attempts)==len(set(attempts)) and not shipped
