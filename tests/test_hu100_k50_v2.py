"""Production audit accepts real v2 exports and still binds their schema."""
import gzip
import json
from pathlib import Path
import subprocess

import pytest

from scripts.audit_native_hu_checkpoint import inspect
from src.blueprint.abstraction import HU100_MENU_V2_SCHEMA, HU100_SCHEMA

BINARY = Path('native/hu20-trainer/target/release/hu20-trainer')


@pytest.mark.skipif(not BINARY.exists(),reason='build native trainer')
def test_v2_native_export_full_audit_and_wrong_schema_refusal(tmp_path):
    checkpoint, current, average = [tmp_path/(n+'.gz') for n in ('checkpoint','current','average')]
    result = subprocess.run([str(BINARY),'train','--stack-bb','100','--menu','v2','--nodes','10000',
        '--seed','2026100601','--average-rule','opponent-sampled','--out',str(checkpoint)],
        capture_output=True,text=True,check=True)
    assert json.loads(result.stderr)['options']['schema'] == HU100_MENU_V2_SCHEMA
    subprocess.run([str(BINARY),'export',str(checkpoint),'--current',str(current),'--average',str(average),
        '--zero-mass','uniform'],capture_output=True,text=True,check=True)
    with gzip.open(checkpoint,'rt') as stream:
        h = json.loads(next(stream))
    receipt = inspect(checkpoint,current,average,100,h['native_state']['completed_nodes'],expected_schema=HU100_MENU_V2_SCHEMA)
    assert receipt['entries'] > 0
    assert receipt['audit']['all_nodes_verified'] == receipt['entries']
    with pytest.raises(ValueError):
        inspect(checkpoint,current,average,100,expected_schema=HU100_SCHEMA)
