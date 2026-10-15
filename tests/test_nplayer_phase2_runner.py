"""Campaign controllers refuse partial training and require every export audit."""
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
from scripts import run_nplayer_blueprint_seed as runner


@pytest.mark.parametrize('reason', ['capacity','time','stop-file'])
def test_success_exit_cannot_admit_a_partial_training_endpoint(reason):
    with tempfile.TemporaryDirectory() as root:
        root=Path(root); commands=[]
        def execute(command,stdout,stderr):
            commands.append(command)
            stdout.write(json.dumps({'stop_reason':reason,'frozen_at':None})+'\n')
            stdout.flush()
            return argparse.Namespace(returncode=0)
        with patch.dict(runner.os.environ,{'NPLAYER_STOP_FILE':str(root/'stop')}), \
             patch.object(runner,'sha',return_value='a'*64), \
             patch.object(runner.subprocess,'check_output',side_effect=['','source','native']), \
             patch.object(runner.subprocess,'run',side_effect=execute), \
             patch.object(runner,'audit') as audit:
            with pytest.raises(RuntimeError,match='before scheduled milestone'):
                runner.run(2026101541,root,root,root/'seed')
            audit.assert_not_called()
        assert len(commands)==1
        command=commands[0]
        assert command[command.index('--last-average-street')+1]=='3'
        assert command[command.index('--max-entries')+1]=='150000000'
        assert '--freeze-at-cap' not in command
        assert not (root/'seed/result.json').exists()


def test_fixed_matrix_counts_roots_and_primary_family():
    from scripts.nplayer_phase2_matrix import matrix
    cases=matrix()
    assert len(cases)==42
    assert len({c['name'] for c in cases})==42
    assert [c['root'] for c in cases]==list(range(202610150801,202610150843))
    assert sum(c['blocks']*6 for c in cases)==2_064_384
    primaries=[c for c in cases if c['primary']]
    assert len(primaries)==3
    assert all(c['singleton'].endswith('/1000m') and c['pair'].endswith('/200m') for c in primaries)
