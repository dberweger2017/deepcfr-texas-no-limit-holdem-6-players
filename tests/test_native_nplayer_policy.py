"""Pinned cash export/loader checks, including independent engine replay."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import pytest
from scripts.check_nplayer_policy_replay import check
from scripts.audit_nplayer_policy import audit

BINARY=Path('native/hu20-trainer/target/release/examples/nplayer_policy')
TRAIN=Path('native/hu20-trainer/target/release/examples/nplayer_train')
pytestmark=pytest.mark.skipif(not BINARY.exists(),reason='build nplayer_policy')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(*args,code=0):
    r=subprocess.run([str(BINARY),*map(str,args)],capture_output=True,text=True,timeout=120)
    assert r.returncode==code,r.stdout+r.stderr
    return r


@pytest.fixture(scope='module')
def model(tmp_path_factory):
    tables=os.environ.get('NPLAYER_K50_TABLES')
    if not tables:pytest.skip('set NPLAYER_K50_TABLES to pinned #163 inputs')
    root=tmp_path_factory.mktemp('nplayer-policy');checkpoint=root/'checkpoint.gz'
    r=subprocess.run([str(TRAIN),'--players','3','--stack-bb','20','--card-buckets',tables,'--nodes','10000','--max-entries','1000000','--max-seconds','60','--last-average-street','0','--out',str(checkpoint)],capture_output=True,text=True,timeout=90)
    assert r.returncode==0,r.stderr
    export=root/'scoped.gz';run('export',checkpoint,sha(checkpoint),'scoped-average',export)
    return root,Path(tables),checkpoint,export


def test_export_probabilities_match_checkpoint_and_name_continuation(model):
    root,tables,checkpoint,export=model
    with gzip.open(checkpoint,'rt') as f:source=[json.loads(s) for s in f]
    with gzip.open(export,'rt') as f:policy=[json.loads(s) for s in f]
    assert policy[0]['strategy']=='scoped-average'
    assert policy[0]['continuation']=='current-after-average-scope'
    assert policy[0]['source_checkpoint_sha256']==sha(checkpoint)
    assert len(source)==len(policy)
    for a,b in zip(source[1:],policy[1:]):
        assert a[:4]==b[:4]
        weights=a[6] if a[2]==0 else [max(0,x) for x in a[5]]
        total=sum(weights);expected=[w/total for w in weights] if total else [1/len(weights)]*len(weights)
        assert b[4]==pytest.approx(expected,abs=1e-14)
    r=run('export',checkpoint,sha(checkpoint),'average',root/'invalid-full.gz',code=2)
    assert 'full average unavailable' in r.stderr
    assert not (root/'invalid-full.gz').exists()
    assert audit(checkpoint,export)['rows']==source[0]['entries']
    corrupted=root/'wrong-probabilities.gz'
    policy[1][4]=[0.0]*len(policy[1][4])
    with gzip.open(corrupted,'wt') as f:
        for row in policy:f.write(json.dumps(row)+'\n')
    with pytest.raises(AssertionError):audit(checkpoint,corrupted)


def test_three_distinct_policy_instances_reproduce_and_match_pinned_engine(model):
    root,tables,checkpoint,export=model
    traces=[]
    for i in range(2):
        path=root/f'play-{i}.jsonl';r=run('play',tables,32,2026101532,path,*([export,sha(export)]*3))
        summary=json.loads(r.stdout);assert summary['distinct_policy_instances']==3
        assert summary['loaded_weight_tables']==1
        assert all(n>0 for n in summary['seat_decisions'])
        assert check(path)['hands']==32
        traces.append(path.read_bytes())
    assert traces[0]==traces[1]


def test_loader_rejects_hash_or_seat_count_before_play(model):
    root,tables,checkpoint,export=model
    run('play',tables,1,7,root/'wrong-hash.jsonl',*([export,'0'*64]*3),code=2)
    run('play',tables,1,7,root/'wrong-seats.jsonl',*([export,sha(export)]*2),code=2)
    assert not (root/'wrong-hash.jsonl').exists()
    assert not (root/'wrong-seats.jsonl').exists()
