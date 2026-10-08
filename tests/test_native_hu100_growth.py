import json
from pathlib import Path
import pytest

from scripts import run_native_hu100_growth as g
from scripts.report_native_hu100_learning_curves import interval


def test_growth_quote_includes_save_tools_and_closeout():
    pilot = {'completed_nodes': g.PARENT_NODES + 100000, 'write_seconds': 10, 'elapsed_seconds_including_writes': 20,
             'diagnostics': {'entries': 3300000}}
    operations = {'pilot-train': {'seconds': 20, 'sampled_peak_family_rss_bytes': 1024**3},
                  'pilot-export': {'seconds': 60}, 'pilot-audit': {'seconds': 80}}
    q = g.growth_quote(pilot, operations, 1800)
    assert q['required_seconds'] >= q['training_seconds'] + 39.1 + 670.5 + 180
    assert q['status'] == 'refused'
    assert q['entry_stop'] == 6510774
    assert not q['outcomes_inspected']


def campaign(tmp_path):
    c = object.__new__(g.Campaign)
    c.out = tmp_path; c.deadline = g.time() + 1800
    c.state_path = tmp_path / 'state.json'
    c.state = {'operations': {}, 'status': 'running'}
    return c


def test_admission_refusal_saves_snapshot_and_never_starts(tmp_path, monkeypatch):
    c = campaign(tmp_path)
    monkeypatch.setattr(g, 'snapshot', lambda *_: {'refused': True, 'memory': {'free_percent': 72}})
    monkeypatch.setattr(g, 'supervise', lambda *a, **k: pytest.fail('refused child spawned'))
    with pytest.raises(g.AdmissionRefused): c.operation('play', ['play'])
    files = list(tmp_path.glob('play-admission-*.json'))
    assert len(files) == 1 and json.loads(files[0].read_text())['memory']['free_percent'] == 72
    assert not (tmp_path / 'play-intent.json').exists()
    assert c.state['operations'] == {}
    assert c.state['status'] == 'admission-refused'


def test_partial_and_changed_operations_never_retry(tmp_path):
    c = campaign(tmp_path)
    c.state['operations']['play'] = {'status': 'attempted', 'command': ['play']}
    with pytest.raises(ValueError, match='never repeat'): c.operation('play', ['play'])
    c.state['operations']['play']['status'] = 'complete'
    with pytest.raises(ValueError, match='never repeat'): c.operation('play', ['changed'])


def test_primary_interval_widens_for_two_comparisons():
    values = [1., 7., -2., 9., 3.]
    ordinary, primary = interval(values), interval(values, .025)
    assert primary['alpha'] == .025
    assert primary['interval'][0] < ordinary['interval'][0]
    assert primary['interval'][1] > ordinary['interval'][1]


def test_stage2_rejects_average_changed_after_full_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(g, 'ROOT', tmp_path)
    stage = tmp_path / 'results/stage1'; folder = stage / 'terminal'
    folder.mkdir(parents=True)
    (stage / 'state.json').write_text(json.dumps({'status': 'complete'}))
    (stage / 'result.json').write_text(json.dumps({'terminal_folder': str(folder)}))
    model = folder / 'average.gz'; model.write_bytes(b'changed')
    (folder / 'audit.json').write_text(json.dumps({'status': 'verified', 'files': {
        str(model): {'bytes': 7, 'sha256': 'original-audited-hash'}}}))
    monkeypatch.setattr(g, 'telemetry', lambda _: {'completed_nodes': g.PARENT_NODES + 1})
    with pytest.raises(ValueError, match='audited member changed'): g.models()


def test_archive_excludes_live_supervisor_files(tmp_path, monkeypatch):
    monkeypatch.setattr(g, 'ROOT', tmp_path)
    root = tmp_path / 'results/stage1'; root.mkdir(parents=True)
    (root / 'state.json').write_text(json.dumps({'status': 'running', 'operations': {'archive': {'status': 'attempted'}}}))
    (root / 'evidence.txt').write_text('immutable')
    (root / 'archive-guard').mkdir(); (root / 'archive-guard/resources.jsonl').write_text('mutable')
    (tmp_path / 'source.tar').write_bytes(b'source')
    (tmp_path / 'bin').mkdir(); (tmp_path / 'bin/hu20-trainer').write_bytes(b'binary')
    for name in ('qualification', 'source-review', 'environment'):
        (tmp_path / 'results' / (name + '.json')).write_text('{}')
    destination = tmp_path / 'cloud/archive.zip'
    g.seal(root, destination)
    from zipfile import ZipFile
    with ZipFile(destination) as z:
        assert 'research/evidence.txt' in z.namelist()
        assert 'research/science-closeout.json' in z.namelist()
        assert not any('archive-guard' in p or p == 'research/state.json' for p in z.namelist())
        frozen = json.loads(z.read('research/science-closeout.json'))
        assert 'archive' not in frozen['operations']
