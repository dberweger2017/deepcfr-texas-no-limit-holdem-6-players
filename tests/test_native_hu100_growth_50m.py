"""Normal audit metadata/report path, capacity admission and measured stops."""
import gzip
import json
from pathlib import Path
import subprocess
from time import time
import pytest

from scripts import run_native_hu100_growth_50m as growth
from scripts.native_hu100_model_metadata import audited_average_spec
from src.policies.files import file_hash


def audit_fixture(path, nodes=10000, iteration=2, checkpoint='checkpoint'):
    expected = {'bytes': path.stat().st_size, 'sha256': file_hash(path)}
    return {'status': 'verified', 'entries': 1, 'iteration': iteration,
            'native_state': {'completed_nodes': nodes}, 'files': {'old/root/average.gz': expected},
            'audit': {'status': 'verified', 'checkpoint_sha256': checkpoint,
                      'average_sha256': expected['sha256'], 'all_nodes_verified': 1, 'counts': {'entries': 1}}}


def test_audited_metadata_uses_counts_without_native_header_entries(tmp_path):
    p = tmp_path / 'average.gz'
    with gzip.open(p, 'wt') as f:
        f.write(json.dumps({'format': 'fixture', 'source_checkpoint_sha256': 'checkpoint',
            'checkpoint_header': {'iteration': 2, 'native_state': {'completed_nodes': 10000}}})+'\n')
    a = audit_fixture(p)
    assert audited_average_spec(p, a, checkpoint_sha256='checkpoint', actual_nodes=10000)['entries'] == 1
    a['audit']['counts']['entries'] = 2
    with pytest.raises(ValueError, match='consistent'): audited_average_spec(p, a, checkpoint_sha256='checkpoint', actual_nodes=10000)
    a = audit_fixture(p); p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes changed'): audited_average_spec(p, a, checkpoint_sha256='checkpoint', actual_nodes=10000)


def test_fresh_capacity_uses_measured_save_peak_and_only_tightens():
    a = json.loads(Path('docs/reports/native-hu100-growth-artifacts/terminal-audit.json').read_text())
    r = json.loads(Path('docs/reports/native-hu100-growth-artifacts/stage1-resources.json').read_text())
    q = growth.capacity(a, r, 57*1024**3)
    assert q['entry_stop'] == 7933918 and q['entry_stop'] != 6510774
    assert q['entry_stop'] <= 2*a['entries'] and q['hard_rss_gib'] == 10
    assert growth.capacity(a, r, 57*1024**3, pilot_peak=3*1024**3, pilot_entries=a['entries'])['entry_stop'] < q['entry_stop']
    assert growth.capacity(a, r, 19*1024**3)['entry_stop'] == 0


def test_cost_quote_reserves_tools_save_and_closeout_without_winnings():
    p = {'diagnostics': {'entries': 5000000}, 'completed_nodes': growth.PARENT_NODES+100000,
         'elapsed_seconds_including_writes': 21, 'write_seconds': 20, 'profit': -100000}
    ops = {'pilot-train': {'seconds': 24}, 'pilot-export': {'seconds': 50}, 'pilot-audit': {'seconds': 110}}
    m = {'sampled_peak_family_rss_bytes': 1700000000}
    q = growth.quote(p,m,ops,7000000,1800)
    assert q['required_seconds'] >= q['training_seconds']+39.1+670.5+180
    p['profit'] = 100000
    assert growth.quote(p,m,ops,7000000,1800) == q
    assert growth.quote(p,m,ops,7000000,300)['status'] == 'refused'


def test_ignored_runtime_binary_normal_report_and_full_reproduction(tmp_path, monkeypatch):
    from tests.test_native_hu100_baseline import model_fixture
    from scripts.evaluate_native_hu100_baseline import execute
    from scripts.audit_native_hu100_baseline import audit
    from scripts.report_native_hu100_learning_curves import frozen_schedule
    from scripts.report_native_hu100_growth import report
    from src.arena import artifacts
    settings = json.loads(Path('configs/arena/hu100-playing-baseline-v1.json').read_text())
    repo = tmp_path / 'repo'; repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    (repo/'.gitignore').write_text('results/\n'); (repo/'fixture.py').write_text('# committed fixture source\n')
    subprocess.run(['git','-C',str(repo),'add','.'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],check=True)
    source = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    runtime = repo/'results/runtime/hu20-trainer'; runtime.parent.mkdir(parents=True); runtime.write_bytes(b'pinned fixture binary')
    run = repo/'results/stage2'; run.mkdir()
    specs=[]
    for nodes in (10000,20000):
        modeldir=repo/'results'/str(nodes); modeldir.mkdir()
        _, _, path = model_fixture(modeldir)
        with gzip.open(path,'rt') as f: rows=f.readlines()
        h=json.loads(rows[0]);h['checkpoint_header']['native_state']={'completed_nodes':nodes}
        with gzip.open(path,'wt') as f: f.write(json.dumps(h)+'\n'+''.join(rows[1:]))
        receipt=audit_fixture(path,nodes,checkpoint=h['source_checkpoint_sha256'])
        specs.append(audited_average_spec(path,receipt,checkpoint_sha256=h['source_checkpoint_sha256'],actual_nodes=nodes))
    settings.pop('model');settings.update(models=specs,final_root=7654321)
    (run/'settings.json').write_text(json.dumps(settings))
    schedule=run/'frozen-schedule.json';schedule.write_text(json.dumps(frozen_schedule(settings,2,settings['final_root'])))
    (run/'frozen-final.json').write_text(json.dumps({'source':source,'blocks_per_opponent':2,'final_root':settings['final_root'],'schedule_sha256':file_hash(schedule)}))
    monkeypatch.setattr(artifacts,'ROOT',repo);monkeypatch.chdir(repo)
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    for i,spec in enumerate(specs):
        n=str(spec['actual_nodes']);config=run/(n+'.config.json')
        single={k:v for k,v in settings.items() if k!='models'};single['model']=spec;config.write_text(json.dumps(single))
        target=run/'final'/n;reference=run/'final'/str(specs[0]['actual_nodes'])
        execute(config,target,2,settings['final_root'],source,reference_run=reference if i else None)
        audit(target,run/f'final-{n}-audit.json')
        repeat=run/'final-reproduction'/n;repeat_ref=run/'final-reproduction'/str(specs[0]['actual_nodes'])
        execute(config,repeat,2,settings['final_root'],source,reproduce=target,reference_run=repeat_ref if i else None)
        assert not json.loads((target/'random/manifest.json').read_text())['dirty']
    result=report(run)  # Default strict path: no verified_source_fingerprint exception.
    assert result['status']=='verified' and result['unique_final_hands']==60
    assert result['formal_family_size']==2
    # An ignored binary stays outside Git dirty metadata but must retain its pin.
    monkeypatch.setattr(growth,'ROOT',repo);monkeypatch.setattr(growth.g,'identity',lambda *_:source)
    c=object.__new__(growth.Campaign);c.source=source;c.binary=runtime;c.pins={'binary_sha256':file_hash(runtime)}
    runtime.write_bytes(b'changed binary')
    monkeypatch.setattr(growth.g.Campaign,'operation',lambda *a,**kw:pytest.fail('changed binary spawned child'))
    with pytest.raises(ValueError,match='unchanged ignored'):c.operation('fixture',['native'])


def test_measure_preserves_controlled_native_capacity_exit(tmp_path, monkeypatch):
    from scripts import benchmark_hu_export_audit as b
    class Child:
        pid=123;returncode=3
        def poll(self):return 3
    def popen(cmd,**kw):kw['stdout'].write(b'1234 maximum resident set size\n');kw['stdout'].flush();return Child()
    monkeypatch.setattr(b.subprocess,'Popen',popen)
    monkeypatch.setattr(b,'memory_snapshot',lambda:{'physical_bytes':16*1024**3,'free_percent':82,'pressure_level':1})
    r=b.measure(['fixture'],tmp_path,tmp_path,'accepted',1,time()+30,accepted_returncodes=(0,3))
    assert r['returncode']==3 and r['kernel_command_peak_rss_bytes']==1234
    with pytest.raises(subprocess.CalledProcessError):b.measure(['fixture'],tmp_path,tmp_path,'strict',1,time()+30)
