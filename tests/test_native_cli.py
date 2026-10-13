"""Native argument errors fail before creating output; stderr identifies the resolved run."""
import json
from pathlib import Path
import subprocess

import pytest

BINARY = Path('native/hu20-trainer/target/release/hu20-trainer')
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason='build the native trainer')


@pytest.mark.parametrize('args,message', [
    (['train','--dcfr','1.5,0,2'], '--dcfr'),
    (['bench-train','--menu','v2'], '--menu'),
    (['train','--unknown'], '--unknown'),
    (['train','--seed','1','--seed','2'], 'duplicate'),
    (['train','--seed'], 'missing value'),
    (['train','--seed','--nodes','1'], 'missing value'),
    (['train','--nodes','nope'], 'unsigned integer'),
    (['train','--max-seconds','NaN'], 'nonfinite'),
    (['train','--regret-floor','inf'], 'nonfinite'),
    (['train','--menu','v3'], '--menu'),
    (['train','--average-rule','dcfr'], '--average-rule'),
    (['train','--stack-bb','50'], '--stack-bb'),
    (['train','--milestones','1,,2'], '--milestones'),
    (['train','--resume','missing'], '--resume-sha256'),
    (['train','--completed-nodes','1'], '--resume'),
    (['train','--card-tables-unpinned'], '--card-buckets'),
    (['train','stray'], 'positional'),
    (['parity','--menu','v2','fixture'], 'misplaced'),
    (['traversal-parity','fixture','--menu','v2'], '--menu'),
    (['run-parity','fixture','--menu','v2'], '--menu'),
    (['export','checkpoint','--seed','1'], '--seed'),
    (['export','checkpoint','--zero-mass','nope'], '--zero-mass'),
    (['engine-deal','nope'], 'unsigned integer'),
    (['bench-train','--dcfr','1,2'], 'alpha,beta,gamma'),
])
def test_invalid_flags_are_clear_errors_without_panic_or_outputs(tmp_path, args, message):
    # Every invalid command must fail argument admission, before trying absent inputs.
    result = subprocess.run([str(BINARY.resolve()), *args], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert message in result.stderr
    assert 'panicked' not in result.stderr and 'backtrace' not in result.stderr
    assert not result.stdout and not list(tmp_path.iterdir())


def test_resolved_json_and_resume_inherited_values(tmp_path):
    from src.policies.files import file_hash
    checkpoint = tmp_path/'initial.gz'
    result = subprocess.run([str(BINARY),'train','--nodes','1','--seed','17','--roots-per-seat','2',
                             '--average-rule','opponent-sampled','--stack-bb','100','--menu','v2',
                             '--out',str(checkpoint)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stderr)
    assert config['command'] == 'train'
    assert config['options']['menu'] == 'v2' and config['options']['stack_bb'] == 100
    assert config['options']['schema'] == 'hu100-native-reopening-ordered-history-card-menu-v2'
    assert config['options']['recovery']
    assert result.stdout.startswith('milestone 1 iterations ')
    resumed = tmp_path/'resumed.gz'
    result = subprocess.run([str(BINARY),'train','--nodes','10000','--stack-bb','100','--menu','v2',
                             '--resume',str(checkpoint),'--resume-sha256',file_hash(checkpoint),
                             '--out',str(resumed)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stderr)['options']
    assert (config['seed'],config['roots_per_seat'],config['average_rule']) == (17,2,'opponent-sampled')


def test_engine_deal_stdout_is_unchanged_and_json_goes_to_stderr():
    result = subprocess.run([str(BINARY),'engine-deal','1','2'],capture_output=True,text=True,check=True)
    assert len(result.stdout.splitlines()) == 2
    assert json.loads(result.stderr) == {'command':'engine-deal','options':{},'positionals':['1','2']}


@pytest.mark.parametrize('lineage,expected', [('3', 3), ('"3"', '3'), ('L', 'L')])
def test_bench_receipt_matches_export_identity_and_schedule(tmp_path, lineage, expected):
    from tests.test_native_bench_parity import roots
    root_file = tmp_path/'roots.json'
    root_file.write_text(json.dumps([r.to_native() for r in roots(1, 3)]))
    result = subprocess.run([str(BINARY),'bench-train','--roots',str(root_file),'--seed','1',
                             '--iterations','3','--checkpoints','2,2,0,99','--lineage',lineage,
                             '--out',str(tmp_path/'out')],capture_output=True,text=True,check=True)
    config = json.loads(result.stderr)
    assert config['requested']['lineage'] == lineage
    assert config['requested']['checkpoints'] == [2, 2, 0, 99]
    assert config['options']['lineage'] == expected
    assert config['options']['checkpoints'] == [2, 3]
    assert [int(line.split()[1]) for line in result.stdout.splitlines()] == config['options']['checkpoints']
    assert sorted(p.name for p in (tmp_path/'out').iterdir()) == ['iteration-2', 'iteration-3']
    for checkpoint in config['options']['checkpoints']:
        for path in (tmp_path/'out'/f'iteration-{checkpoint}').glob('*.json'):
            document = json.loads(path.read_text())
            assert document['lineage'] == config['options']['lineage']
            assert document['iteration'] == checkpoint
            assert all(row['lineage'] == document['lineage'] for row in document['groups'])


def test_train_receipt_matches_filtered_milestones_and_resume(tmp_path):
    import gzip
    from src.policies.files import file_hash
    result = subprocess.run([str(BINARY),'train','--nodes','3','--milestones','2,2,0,99',
                             '--recovery','--out',str(tmp_path/'checkpoint-{nodes}.gz')],
                            capture_output=True,text=True,check=True)
    config = json.loads(result.stderr)
    assert config['requested']['milestones'] == [2, 2, 0, 99]
    assert config['options']['milestones'] == [2, 3]
    assert [int(line.split()[1]) for line in result.stdout.splitlines()] == config['options']['milestones']
    checkpoint = tmp_path/'checkpoint-3.gz'
    header = json.loads(gzip.open(checkpoint, 'rt').readline())
    completed = header['native_state']['completed_nodes']
    target = completed + 1
    result = subprocess.run([str(BINARY),'train','--nodes',str(target),
                             '--milestones',f'0,2,{completed},{target},{target+1}',
                             '--resume',str(checkpoint),'--resume-sha256',file_hash(checkpoint),
                             '--out',str(tmp_path/'resumed-{nodes}.gz')],
                            capture_output=True,text=True,check=True)
    config = json.loads(result.stderr)
    assert config['options']['initial_nodes'] == completed
    assert config['options']['initial_iteration'] == header['iteration']
    assert config['options']['milestones'] == [target]
    assert [int(line.split()[1]) for line in result.stdout.splitlines()] == [target]


@pytest.mark.parametrize('dcfr,iterations,message', [
    ('1,0,2', '1', 'alpha > 1'),
    ('1.5,0,9', '1', 'gamma in [0, 8]'),
    ('1.5,0,2', '4294967296', 'before 2^32 iterations'),
])
def test_dcfr_domain_errors_precede_input_loading(tmp_path, dcfr, iterations, message):
    result = subprocess.run([str(BINARY.resolve()),'bench-train','--roots',str(tmp_path/'absent-roots'),
                             '--seed','1','--iterations',iterations,'--lineage','3',
                             '--dcfr',dcfr,'--out',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode == 2
    assert message in result.stderr
    assert 'No such file' not in result.stderr
    assert not result.stderr.startswith('{') and not result.stdout and not list(tmp_path.iterdir())
