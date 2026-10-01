"""Operational priority must preserve every frozen model/stage/count."""

from copy import deepcopy
import pytest

from scripts.queue_hu20_500m_endpoint_first import pending_jobs


def fixture():
    plan = dict(parents=[{'seed':s} for s in (1,2,3)],
                light_totals=[],
                broad_totals=[100000000,150000000,200000000,300000000,400000000,500000000],
                heldout_totals=[100000000,500000000])
    tasks = [dict(destination='M4',spec=dict(seed=s,milestone=m,hash=f'{s}-{m}'))
             for s in (3,2,1) for m in reversed(plan['broad_totals'])]
    return plan,tasks


def test_complete_endpoint_groups_before_intermediates_and_confirmation():
    plan,tasks=fixture()
    jobs=pending_jobs(plan,tasks,set())
    assert [(s,m['milestone'],m['seed']) for s,m,i in jobs[:6]] == [
        ('broad',m,s) for m in (100000000,500000000) for s in (1,2,3)]
    assert [(s,m['milestone'],m['seed']) for s,m,i in jobs[6:18]] == [
        ('broad',m,s) for m in (150000000,200000000,300000000,400000000) for s in (1,2,3)]
    assert [(s,m['milestone'],m['seed']) for s,m,i in jobs[18:]] == [
        ('heldout',m,s) for s in (1,2,3) for m in (100000000,500000000)]


def test_no_specification_changes_and_no_completed_task_repetition():
    plan,tasks=fixture(); before=deepcopy((plan,tasks))
    all_jobs=pending_jobs(plan,tasks,set())
    remaining=pending_jobs(plan,tasks,{'broad-1-100000000'})
    assert len(all_jobs)==24 and len(remaining)==23
    assert {i for _,_,i in remaining}=={i for _,_,i in all_jobs}-{'broad-1-100000000'}
    assert (plan,tasks)==before
    assert all(any(m is t['spec'] for t in tasks) for _,m,_ in remaining)


def test_duplicate_tasks_are_rejected():
    plan,tasks=fixture()
    with pytest.raises(ValueError,match='Duplicate'):
        pending_jobs(plan,tasks+[tasks[0]],set())


def test_unretrieved_policy_cannot_enter_queue():
    plan,tasks=fixture();tasks[0]['destination']='M1'
    with pytest.raises(ValueError,match='retrieved'):
        pending_jobs(plan,tasks,set())
