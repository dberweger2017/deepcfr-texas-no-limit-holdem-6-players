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


def test_model_specs_use_verified_counts_without_header_entries(tmp_path, monkeypatch):
    import gzip
    from src.policies.files import file_hash
    monkeypatch.setattr(g, 'ROOT', tmp_path)
    stage = tmp_path / 'results/stage1'; folder = stage / 'terminal'; folder.mkdir(parents=True)
    (stage / 'state.json').write_text(json.dumps({'status': 'complete'}))
    (stage / 'result.json').write_text(json.dumps({'terminal_folder': str(folder)}))
    hashes = {}
    for name, nodes in [('parent-average.gz', g.PARENT_NODES), ('terminal/average.gz', g.PARENT_NODES + 1)]:
        p = stage / name
        with gzip.open(p, 'wt') as f:
            json.dump({'format': 'research', 'checkpoint_header': {'iteration': 8,
                'native_state': {'completed_nodes': nodes}}, 'source_checkpoint_sha256': 'checkpoint'}, f)
            f.write('\n')
        hashes[str(p)] = {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
    average = folder / 'average.gz'
    (folder / 'audit.json').write_text(json.dumps({'status': 'verified', 'entries': 7,
        'files': {str(average): hashes[str(average)]}}))
    parent = stage / 'parent-average.gz'
    (stage / 'retrieval.json').write_text(json.dumps({'members': [{'local_path': str(parent), **hashes[str(parent)]}]}))
    index = tmp_path / 'docs/reports/native-recovery-hu100-artifacts/followup-model-index.json'
    index.parent.mkdir(parents=True); index.write_text(json.dumps({'models': [{'entries': 5}]}))
    monkeypatch.setattr(g, 'telemetry', lambda _: {'completed_nodes': g.PARENT_NODES + 1, 'checkpoint_sha256': 'checkpoint'})
    specs = g.models()
    assert [s['entries'] for s in specs] == [5, 7]


def test_frozen_source_digest_matches_manifest_on_committed_source():
    import subprocess
    from scripts.verify_native_hu100_growth_closeout import git_source_fingerprint
    from src.arena.artifacts import source_fingerprint
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    # This fixture runs after the source commit, as the evaluator also requires.
    assert git_source_fingerprint(source) == source_fingerprint()


def test_postprocessing_rejects_changed_model_order_before_reporting(tmp_path):
    from scripts.verify_native_hu100_growth_closeout import validate, FROZEN_SOURCE
    deadline = g.time() + 120
    (tmp_path / 'state.json').write_text(json.dumps({'status': 'failed',
        'error': "ValueError('Terminal child/guard failure: report')", 'pins': {'source': FROZEN_SOURCE},
        'deadline': deadline, 'operations': {'report': {'status': 'failed'}}}))
    (tmp_path / 'frozen-final.json').write_text(json.dumps({'source': FROZEN_SOURCE,
        'deadline': deadline, 'settings_sha256': 'the-original-fixed-order-hash'}))
    (tmp_path / 'settings.json').write_text('{"models": ["terminal", "parent"]}')
    with pytest.raises(ValueError, match='Fixed model settings changed'): validate(tmp_path)
