"""Disposable child tests for serialization headroom; never invoke a trainer."""
import json
import os
import sys
from time import time

import pytest

from scripts import hu20_scaling_supervise as supervisor
from tests.test_native_hu_supervisor import environment


def jobs(stop):
    code = 'import pathlib,sys,time; p=pathlib.Path(sys.argv[2]); end=time.time()+3\nwhile not p.exists() and time.time()<end: time.sleep(.01)\nsys.exit(3 if p.exists() else 1)'
    return [{'name':'controlled-fixture','command':[sys.executable,'-c',code,'--stop-file',str(stop)]}]


def test_soft_rss_stop_requests_save_without_hard_kill(monkeypatch,tmp_path):
    environment(monkeypatch)
    stop=tmp_path/'stop.json'
    # Sample RSS between soft and hard limits. The real child's polling observes
    # the request; its exit3 is retained as an incomplete capacity result.
    monkeypatch.setattr(supervisor.subprocess,'check_output',lambda *_a,**_k:f'{os.getpid()} 1 2000\n')
    result=supervisor.run(jobs(stop),tmp_path/'guard',time()+5,disk_gib=.001,
                          rss_gib=.01,stop_file=stop,soft_rss_gib=.001,save_reserve_seconds=1)
    assert result['status']=='incomplete'
    assert result['attempts'][0]['exit_code']==3
    assert result['attempts'][0]['guard_failure'] is None
    assert result['controlled_stop']['reason']=='Serialization headroom RSS stop'
    assert json.loads(stop.read_text())==result['controlled_stop']


def test_no_child_starts_when_save_reserve_is_already_exhausted(monkeypatch,tmp_path):
    environment(monkeypatch)
    monkeypatch.setattr(supervisor.subprocess,'Popen',lambda *_a,**_k:pytest.fail('spawned after reserve refusal'))
    stop=tmp_path/'stop.json'
    result=supervisor.run(jobs(stop),tmp_path/'guard',time()+.2,disk_gib=.001,
                          rss_gib=.01,stop_file=stop,soft_rss_gib=.001,save_reserve_seconds=1)
    assert result['status']=='incomplete'
    assert result['attempts'][0]['guard_failure']=='Insufficient save time reserve'
    assert not stop.exists()


def test_controlled_stop_cannot_disable_hard_rss_guard(monkeypatch,tmp_path):
    environment(monkeypatch)
    monkeypatch.setattr(supervisor.subprocess,'check_output',lambda *_a,**_k:f'{os.getpid()} 1 20000\n')
    stop=tmp_path/'stop.json'
    result=supervisor.run(jobs(stop),tmp_path/'guard',time()+5,disk_gib=.001,
                          rss_gib=.01,stop_file=stop,soft_rss_gib=.001,save_reserve_seconds=1)
    assert result['status']=='incomplete'
    assert result['attempts'][0]['guard_failure']=='Aggregate job RSS guard'


def test_stop_request_must_match_the_only_owned_training_command(tmp_path):
    stop=tmp_path/'stop.json'
    with pytest.raises(ValueError,match='matching trainer'):
        supervisor.run([{'name':'wrong','command':['unused']}],tmp_path/'never',time()+5,
                        stop_file=stop,soft_rss_gib=4,save_reserve_seconds=1)
    assert not (tmp_path/'never').exists()
