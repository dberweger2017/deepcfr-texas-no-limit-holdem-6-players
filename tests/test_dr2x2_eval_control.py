"""Exact scope, cross-host gate and retrieval proof before rental teardown."""
import io
import json
import tarfile
from hashlib import sha256
from copy import deepcopy
import pytest
from scripts.dr2x2_eval_control import valid_eval_lease, verify_closed_archive
from scripts.dr2x2_eval_pod import same_reference


def test_watchdog_rejects_other_campaign_names_counts_and_budgets():
    lease={'names':[f'dr2x2-EVAL-{s}-123' for s in (2026093001,2026093002,2026093003)],'subcap_usd':4,'reserve_usd':1}
    assert valid_eval_lease(lease)
    other=deepcopy(lease);other['names'][0]='dr2x2-C-2026093001-123';assert not valid_eval_lease(other)
    other=deepcopy(lease);other['names'][0]=other['names'][1];assert not valid_eval_lease(other)
    other=deepcopy(lease);other['subcap_usd']=16;assert not valid_eval_lease(other)


def test_linux_admission_needs_all_same_readouts_counts_and_fingerprints():
    reference={'status':'complete','seed':2026093001,'closed_hands':160,'plan_sha256':'plan','tasks':[{'cell':c,'readout':r,'hands':40,'scientific_fingerprints':['a','b']}
               for c in ('A','C') for r in ('current','average')]}
    assert same_reference(reference,deepcopy(reference))
    damaged=deepcopy(reference);damaged['tasks'][1]['scientific_fingerprints'][0]='different';assert not same_reference(reference,damaged)
    damaged=deepcopy(reference);damaged['tasks'].pop();assert not same_reference(reference,damaged)
    damaged=deepcopy(reference);damaged['status']='failed-retained';assert not same_reference(reference,damaged)
    damaged=deepcopy(reference);damaged['tasks'].append(damaged['tasks'][0]);assert not same_reference(reference,damaged)


def test_archive_requires_every_closed_member_hash_and_safe_paths(tmp_path):
    def make(path,damaged=False,unsafe=False):
        data=b'closed replay evidence\n'
        manifest=json.dumps({'files':{'evidence.txt':{'bytes':len(data),'sha256':'0'*64 if damaged else sha256(data).hexdigest()}}}).encode()
        with tarfile.open(path,'w') as tar:
            for name,body in [('results/dr2x2-eval/evidence.txt',data),('results/dr2x2-eval/manifest.json',manifest)]:
                m=tarfile.TarInfo(name);m.size=len(body);tar.addfile(m,io.BytesIO(body))
            if unsafe:
                m=tarfile.TarInfo('../unrelated');m.size=0;tar.addfile(m,io.BytesIO(b''))
    good=tmp_path/'good.tar';make(good);assert verify_closed_archive(good)=={'passed':True,'verified_files':1}
    bad=tmp_path/'bad.tar';make(bad,damaged=True)
    with pytest.raises(ValueError,match='hash'):verify_closed_archive(bad)
    unsafe=tmp_path/'unsafe.tar';make(unsafe,unsafe=True)
    with pytest.raises(ValueError,match='Unsafe'):verify_closed_archive(unsafe)


def test_capacity_projection_keeps_both_full_work_schedules_and_readouts():
    from scripts.dr2x2_eval_pod import projected_playing_seconds
    plan={'panels':[{'name':'lbr','blocks':1024,'readout_blocks':256}]}
    actual={'tasks':[{'readout':r,'load_seconds':2,'panels':[{'panel':'lbr','max_hand_seconds':.1}]}
                      for r in ('current','average')]}
    assert projected_playing_seconds(plan,actual)==pytest.approx(8+4*.1*(1280+256))
