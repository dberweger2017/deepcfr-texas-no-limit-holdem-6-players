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
