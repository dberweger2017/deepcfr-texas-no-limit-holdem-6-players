"""Ownership, spending, destination ACK and exact same-work campaign recovery."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
from scripts.dr2x2_control import estimated_cost, budget_action
from scripts.dr2x2_worker import acknowledged, rotate, run, make_trainer, scientific_config
from scripts.hu20_platform_pilot import write
from scripts.mature_cpu_rental_guard import owned_pods, check_quote
from src.blueprint.artifact import save_training

PLAN=Path('configs/blueprint/dr2x2-cd-campaign.json')


def test_approved_work_shapes_waves_and_no_measurement_parents():
    plan=json.loads(PLAN.read_text())
    assert plan['owner_approved_budget']['approval'].startswith('APPROVED')
    assert plan['target_total_nodes']==100000000
    assert len(plan['jobs'])==6 and len(plan['waves'][0])==4 and len(plan['waves'][1])==2
    assert {(j['cell'],j['seed']) for j in plan['jobs']}=={(c,s) for c in ('C','D') for s in plan['seeds']}
    assert all(j['completed_nodes']==0 and j['checkpoint_sha256']=='from-zero' for j in plan['jobs'])
    assert plan['cells']['C']['max_entries']==8000000 and plan['cells']['D']['max_entries']==24000000
    assert plan['recovery_totals']==[10000000,20000000,25000000,30000000,40000000,50000000,60000000,70000000,80000000,90000000,100000000]
    assert plan['export_totals']==plan['permanent_totals']==[25000000,50000000,100000000]


def test_cost_retains_terminated_attempts_and_training_reserve():
    rows=[dict(created_epoch=0,terminated=3600,upper_rate=.57),dict(created_epoch=1,terminated=3601,upper_rate=.18),dict(created_epoch=3600,upper_rate=.57)]
    # Zero is a real epoch, not a missing creation. Production uses positive epochs.
    rows[0]['created_epoch']=1;rows[0]['terminated']=3601
    assert estimated_cost(rows,7200)==pytest.approx(1.32)
    assert budget_action(9.99,12,2)=='continue'
    assert budget_action(10,12,2)=='stop'
    assert budget_action(14,16,2)=='stop'


def test_exact_owned_names_never_admit_neighbor_and_quote():
    pods=[dict(id='ours',name='dr2x2-C-3001'),dict(id='other',name='new-guy-B-3001')]
    assert [p['id'] for p in owned_pods(pods,{'dr2x2-C-3001'})]==['ours']
    config=dict(cpu_id='cpu5m',vcpus=2,ram_gb=16)
    pod=dict(cpu=dict(id='cpu5m',vcpuCount=2,memory=16),cost=.13)
    assert check_quote(config,pod,.13)
    assert not check_quote(config,{**pod,'cost':.14},.13)
    assert not check_quote(config,{**pod,'cpu':{**pod['cpu'],'memory':8}},.13)


def test_rotation_requires_matching_ack_and_keeps_permanent(tmp_path):
    (tmp_path/'ack').mkdir();rows=[]
    for i in range(5):
        p=tmp_path/f'checkpoint-{i}.gz';p.write_bytes(str(i).encode())
        rows.append(dict(id=str(i),permanent=i==0,files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha256(p.read_bytes()).hexdigest())]))
    # Forged/partial destination acknowledgements cannot authorize source deletion.
    write(tmp_path/'ack/1.json',dict(id='1',files=[]));assert not acknowledged(rows[1],tmp_path)
    for row in (rows[0],rows[2],rows[3],rows[4]):write(tmp_path/'ack'/(row['id']+'.json'),dict(id=row['id'],files=row['files']))
    rotate(rows,tmp_path)
    assert (tmp_path/'checkpoint-0.gz').exists() and (tmp_path/'checkpoint-1.gz').exists()
    assert not (tmp_path/'checkpoint-2.gz').exists()
    assert (tmp_path/'checkpoint-3.gz').exists() and (tmp_path/'checkpoint-4.gz').exists()


@pytest.mark.parametrize('cell',['C','D'])
def test_direct_fresh_resume_trace_and_state_bytes(tmp_path,cell):
    plan=json.loads(PLAN.read_text());plan['preflight_nodes']=2500
    pp=tmp_path/'plan.json';write(pp,plan)
    parent=next(j for j in plan['jobs'] if j['cell']==cell);jp=tmp_path/'parent.json';write(jp,parent)
    for name in ('direct','resume'):
        args=[sys.executable,'-m','scripts.dr2x2_worker','preflight','--plan',str(pp),'--parent',str(jp),'--out',str(tmp_path/name)]
        if name=='resume':args+=['--resume',str(tmp_path/'direct')]
        subprocess.run(args,check=True,capture_output=True)
    a=json.loads((tmp_path/'direct/result.json').read_text());b=json.loads((tmp_path/'resume/result.json').read_text())
    for key in ('added_nodes','iteration','next_streams','suffix_trace','suffix_work_sha256','final','current'):assert a[key]==b[key]
    assert a['suffix_trace']['counts']['observations']>0 and a['suffix_trace']['counts']['keys_menus']>0


def test_training_waits_for_hash_ack_and_matches_plain_trainer(tmp_path):
    plan=json.loads(PLAN.read_text());plan.update(target_total_nodes=4000,recovery_totals=[2000,4000],export_totals=[4000],permanent_totals=[4000])
    parent=plan['jobs'][0];control=tmp_path/'control.json';write(control,dict(lease_until=time.time()+120,stop=None))
    out=tmp_path/'training';closed=threading.Event()
    def acknowledger():
        while not closed.wait(.02):
            saved=out/'saved.jsonl'
            if not saved.exists():continue
            for line in saved.read_text().splitlines():
                row=json.loads(line)
                for item in row['files']:assert sha256((out/item['name']).read_bytes()).hexdigest()==item['sha256']
                write(out/'ack'/(row['id']+'.json'),dict(id=row['id'],files=row['files']))
    thread=threading.Thread(target=acknowledger,daemon=True);thread.start()
    try:r=run(plan,parent,out,control)
    finally:closed.set();thread.join()
    assert r['status']=='complete' and r['completed_nodes']>=4000
    trainer=make_trainer(plan,parent);nodes=0
    while nodes<4000:nodes+=trainer.step().nodes
    cp=tmp_path/'plain.gz';save_training(trainer,cp)
    last=json.loads((out/'saved.jsonl').read_text().splitlines()[-1])
    assert cp.read_bytes()==(out/last['files'][0]['name']).read_bytes()
    assert sum(n.visits for n in trainer.nodes.values())==r['cumulative_work']['raw_traverser_visits']


def test_final_archive_checks_every_closed_byte_without_extraction(tmp_path):
    import tarfile
    from scripts.dr2x2_control import verify_archive
    worker=tmp_path/'results/worker';worker.mkdir(parents=True)
    (worker/'result.json').write_bytes(b'closed result')
    write(worker/'manifest.json',dict(files={'result.json':dict(bytes=13,sha256=sha256(b'closed result').hexdigest())}))
    archive=tmp_path/'archive.tar'
    with tarfile.open(archive,'w') as target:target.add(tmp_path/'results',arcname='results')
    assert verify_archive(archive)==dict(passed=True,verified_files=1)
    (worker/'result.json').write_bytes(b'changed bytes')
    with tarfile.open(archive,'w') as target:target.add(tmp_path/'results',arcname='results')
    with pytest.raises(ValueError,match='hash mismatch'):verify_archive(archive)
