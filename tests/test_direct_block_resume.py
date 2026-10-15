"""Battery pauses retain complete paired blocks without repeating a played hand."""
import gzip
import json
import pytest

from scripts import evaluate_hu100_direct as direct
from tests.test_v2_consumers import synthetic, dev_source
from src.blueprint.abstraction import HU100_SCHEMA, HU100_MENU_V2_SCHEMA


@pytest.mark.parametrize('opponent', [None, 'half_pot_pressure'])
@pytest.mark.parametrize('instrumented', [False,True])
def test_pause_resume_matches_uninterrupted_streams(tmp_path, monkeypatch, opponent, instrumented):
    specs = [synthetic(tmp_path/'left', HU100_SCHEMA, 'left'),
             synthetic(tmp_path/'right', HU100_MENU_V2_SCHEMA, 'right')]
    source = dev_source(monkeypatch)
    options = dict(opponent=opponent)
    if instrumented:
        specs[0]['translation']=dict(max_states=512,max_events=128)
        specs[1]['translation']=None
        options['cache']=tmp_path/'cache'
    _, full = direct.execute(specs, 5, 872361, 'fixture', tmp_path/'full', source, **options)
    for stage, original in [('play', None), ('repeat', tmp_path/'play')]:
        out = tmp_path/stage
        _, paused = direct.execute(specs, 5, 872361, 'fixture', out, source,
            reproduce=original, boundary_stop=lambda next_block: next_block == 2, **options)
        assert paused['status'] == 'paused' and paused['next_block'] == 2
        assert paused['hands'] == (8 if opponent else 4)
        assert not (out/'complete.json').exists()
        _, done = direct.execute(specs, 5, 872361, 'fixture', out, source,
            reproduce=original, resume=True, **options)
        assert done['hands'] == full['hands']
        for name in ('hands.jsonl.gz', 'decisions.jsonl.gz'):
            left=gzip.decompress((out/name).read_bytes());right=gzip.decompress((tmp_path/'full'/name).read_bytes())
            if instrumented and name=='decisions.jsonl.gz':
                assert [direct.stable_decision(json.loads(row)) for row in left.splitlines()]==[
                    direct.stable_decision(json.loads(row)) for row in right.splitlines()]
            else:
                assert left==right
        if stage == 'repeat':
            assert done['reproduced_all_hands_and_decisions']
    assert direct.audit(tmp_path/'play', tmp_path/'audit.json')['status'] == 'verified'


def test_resume_rejects_changed_stream_and_foreign_inputs(tmp_path, monkeypatch):
    specs = [synthetic(tmp_path/'left', HU100_SCHEMA, 'left'),
             synthetic(tmp_path/'right', HU100_SCHEMA, 'right')]
    source = dev_source(monkeypatch); out = tmp_path/'play'
    direct.execute(specs, 3, 872362, 'fixture', out, source, boundary_stop=lambda n: n == 1)
    with pytest.raises(ValueError, match='input identity'):
        direct.execute(specs, 3, 872363, 'fixture', out, source, resume=True)
    stream = out/'hands.jsonl.gz'; stream.write_bytes(stream.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='raw stream changed'):
        direct.execute(specs, 3, 872362, 'fixture', out, source, resume=True)
