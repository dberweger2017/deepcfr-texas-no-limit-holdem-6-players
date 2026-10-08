"""All-row recovery comparison fixtures; no training, policy downloads or worker use."""
import gzip
import json
import subprocess

import pytest

from scripts import compare_native_hu_recovery as recovery


def write_gz(path, rows):
    with gzip.open(path,'wt') as f:
        for row in rows: f.write(json.dumps(row)+'\n')


def fixture(tmp_path, monkeypatch):
    parent, reference, resumed = (tmp_path/name for name in ('parent.gz','reference.gz','resumed.gz'))
    reference_current, resumed_current = (tmp_path/name for name in ('current.gz','resumed-current.gz'))
    reference_average, resumed_average = (tmp_path/name for name in ('average.gz','resumed-average.gz'))
    reference_telemetry, resumed_telemetry = (tmp_path/name for name in ('reference.jsonl','resumed.jsonl'))
    h = {'iteration':2126271,'config':{'seed':recovery.SEED},'average_rule':'opponent-sampled'}
    ph = {**h,'iteration':1095942}
    state = {'version':1,'completed_nodes':1000000017,'coverage_start':[1095942,500000012,1],
             'decisions_by_street':[2,2,2,2],'traverser_visits_by_street':[0,1,0,1]}
    rh = {**h,'native_state':state}
    rows = [['0'*32,['check'],[-0.0],[0.0],1], ['1'*32,['check'],[2.0],[2.0],2]]
    write_gz(parent,[ph,rows[0]])
    write_gz(reference,[h,*rows]); write_gz(resumed,[rh,*rows])
    reference_current.write_bytes(b'identical current bytes')
    resumed_current.write_bytes(reference_current.read_bytes())
    def d(visits, decisions, streets, entries=2, baseline=None):
        return {'entries':entries,'traverser_visits':visits,'decisions_by_street':decisions,
                'traverser_visits_by_street':streets,'coverage_start':baseline or [0,0,0]}
    pd = d(1,[1,1,1,1],[1,0,0,0],1)
    rd = d(3,[3,3,3,3],[1,1,0,1])
    sd = d(3,[2,2,2,2],[0,1,0,1],baseline=state['coverage_start'])
    def r(path, requested, completed, iteration, diagnostics):
        return {'path':str(path),'status':'saved','checkpoint_sha256':recovery.file_hash(path),
            'checkpoint_bytes':path.stat().st_size,'requested_nodes':requested,'completed_nodes':completed,
            'iteration':iteration,'diagnostics':diagnostics,'binary_sha256':'b'*64}
    reference_telemetry.write_text('\n'.join(json.dumps(x) for x in
        [r(parent,500000000,500000012,1095942,pd),r(reference,1000000000,1000000017,2126271,rd)])+'\n')
    resumed_telemetry.write_text(json.dumps(r(resumed,1000000000,1000000017,2126271,sd))+'\n')
    (tmp_path/'plan.json').write_text(json.dumps({'campaign_swap_baseline':'used = 0M',
        'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'binary_sha256':'b'*64})+'\n')
    def meta(header,path):
        return {'format':'fixture','checkpoint_header':header,'source_checkpoint_sha256':recovery.file_hash(path),
                'zero_mass_rule':'uniform'}
    average_rows = [[row[0],row[1],[1.0],row[3][0],row[4]] for row in rows]
    write_gz(reference_average,[meta(h,reference),*average_rows])
    write_gz(resumed_average,[meta(rh,resumed),*average_rows])
    monkeypatch.setattr(recovery,'PARENT_SHA',recovery.file_hash(parent))
    # Header validation and full accumulator audits have existing separate fixtures;
    # these fixtures isolate the equivalence gates and narrow metadata differences.
    monkeypatch.setattr(recovery,'read_header',lambda p:json.loads(gzip.open(p,'rt').readline()))
    monkeypatch.setattr(recovery,'inspect',lambda *_a,**_kw:{'status':'verified'})
    return dict(reference=reference,resumed=resumed,parent=parent,reference_telemetry=reference_telemetry,
        resumed_telemetry=resumed_telemetry,reference_current=reference_current,reference_average=reference_average,
        resumed_current=resumed_current,resumed_average=resumed_average)


def test_full_recovery_accepts_only_explained_metadata_delta(monkeypatch,tmp_path):
    args = fixture(tmp_path,monkeypatch)
    result = recovery.compare(**args)
    assert result['status'] == 'verified' and result['entries'] == 2
    assert result['metadata_differences']['checkpoint'] == ['native_state']
    assert result['completed_nodes'] == 1000000017


@pytest.mark.parametrize('change',['regret','accumulator','visits','signed-zero','key','length','config','coverage'])
def test_state_changes_block_equivalence(monkeypatch,tmp_path,change):
    args = fixture(tmp_path,monkeypatch)
    path = args['resumed']
    with gzip.open(path,'rt') as f: rows = [json.loads(line) for line in f]
    if change == 'regret': rows[2][2][0] += 1
    elif change == 'accumulator': rows[2][3][0] += 1
    elif change == 'visits': rows[2][4] += 1
    elif change == 'signed-zero': rows[1][2][0] = 0.0
    elif change == 'key': rows[2][0] = '2'*32
    elif change == 'length': rows.pop()
    elif change == 'config': rows[0]['config']['seed'] += 1
    else: rows[0]['native_state']['decisions_by_street'][0] += 1
    write_gz(path,rows)
    # Keep the receipt bound to the changed bytes, so mismatch detection exercises
    # complete-state comparison rather than merely rejecting an outdated hash.
    rp = args['resumed_telemetry']; r=json.loads(rp.read_text())
    r['checkpoint_sha256']=recovery.file_hash(path); r['checkpoint_bytes']=path.stat().st_size
    rp.write_text(json.dumps(r)+'\n')
    with pytest.raises(ValueError): recovery.compare(**args)


@pytest.mark.parametrize('change',['probability','metadata','current','binary','actual-nodes'])
def test_export_or_execution_changes_block_equivalence(monkeypatch,tmp_path,change):
    args = fixture(tmp_path,monkeypatch)
    if change == 'current': args['resumed_current'].write_bytes(b'different')
    elif change in ('binary','actual-nodes'):
        p=args['resumed_telemetry']; r=json.loads(p.read_text())
        if change == 'binary': r['binary_sha256']='c'*64
        else: r['completed_nodes']+=1
        p.write_text(json.dumps(r)+'\n')
    else:
        p=args['resumed_average']
        with gzip.open(p,'rt') as f: rows=[json.loads(line) for line in f]
        if change == 'probability': rows[2][2][0]=.5
        else: rows[0]['zero_mass_rule']='current'
        write_gz(p,rows)
    with pytest.raises(ValueError): recovery.compare(**args)


def test_float_equality_is_exact_and_preserves_signed_zero():
    assert recovery.exact([-0.0,{'a':1.0}],[-0.0,{'a':1.0}])
    assert not recovery.exact([-0.0],[0.0])
    assert not recovery.exact([1.0],[1.0000000000000002])
