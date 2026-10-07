"""Durable launch/liveness fixtures mock system reads and never start a process."""
import json
from time import time
import pytest
from scripts import verify_native_hu_launch as launch
from scripts.prepare_native_hu_campaign import file_hash,digest


def put(path,value): path.write_text(json.dumps(value)+'\n')


def fixture(tmp_path,monkeypatch,pr_state='MERGED',busy=False):
    binary=tmp_path/'binary'; binary.write_bytes(b'fixture')
    q=tmp_path/'q.json'; put(q,{})
    state=tmp_path/'campaign.json'; put(state,{'active_attempt':None})
    closeout=tmp_path/'pr188-closeout.json'; put(closeout,{'status':'complete'})
    p={'source':'source','binary':str(binary),'binary_sha256':file_hash(binary),
       'qualification_path':str(q),'qualification_sha256':file_hash(q),'campaign_swap_baseline':'used = 0M',
       'hard_deadline':launch.DEADLINE,'phase_jobs':{},'stage':'hu20'}
    p['plan_sha256']=digest(p); plan=tmp_path/'plan.json'; put(plan,p)
    put(tmp_path/'admission.json',{'status':'admitted','m4_idle':True,'pr188_state':'MERGED',
        'worker_closeout':'complete','observed_at':time(),'campaign_state_path':str(state),
        'closeout_evidence':{str(closeout):{'bytes':closeout.stat().st_size,'sha256':file_hash(closeout)}}})
    monkeypatch.setattr(launch.platform,'system',lambda:'Darwin')
    monkeypatch.setattr(launch,'time',lambda:launch.DEADLINE-1000)
    a=json.loads((tmp_path/'admission.json').read_text()); a['observed_at']=launch.DEADLINE-1000
    put(tmp_path/'admission.json',a)
    monkeypatch.setattr(launch,'qualification',lambda *_a:('source',{}))
    def command(argv,**_kwargs):
        if argv[0]=='sysctl':return 'Apple M4\n'
        if argv[0]=='gh':return json.dumps({'state':pr_state})
        if argv[0]=='ps':return '42 1 python hu20-o-10b-lbr-20261007/archive.py\n' if busy else ''
        raise AssertionError(argv)
    monkeypatch.setattr(launch.subprocess,'check_output',command)
    return plan,state,binary


def test_exclusive_claim_survives_duplicate_launch_and_lost_ack(monkeypatch,tmp_path):
    plan,state,_=fixture(tmp_path,monkeypatch)
    first=launch.verify(plan)
    assert json.loads(state.read_text())['active_attempt']==first
    with pytest.raises(ValueError,match='active or uncertain'): launch.verify(plan)
    assert json.loads(state.read_text())['launch_attempts']==[first]
    assert state.with_name(state.name+'.launch-lock').exists()


@pytest.mark.parametrize('pr_state,busy',[('OPEN',False),('MERGED',True)])
def test_pr188_gate_prevents_claim(monkeypatch,tmp_path,pr_state,busy):
    plan,state,_=fixture(tmp_path,monkeypatch,pr_state,busy)
    with pytest.raises(ValueError):launch.verify(plan)
    assert json.loads(state.read_text())['active_attempt'] is None
    assert not (tmp_path/'LAUNCH.json').exists()


def test_binary_substitution_prevents_launch(monkeypatch,tmp_path):
    plan,state,binary=fixture(tmp_path,monkeypatch); binary.write_bytes(b'substituted')
    with pytest.raises(ValueError,match='source/binary'):launch.verify(plan)
    assert json.loads(state.read_text())['active_attempt'] is None
