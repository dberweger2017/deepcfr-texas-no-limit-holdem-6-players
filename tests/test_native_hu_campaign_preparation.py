"""The new campaign tool writes commands, with no subprocess execution beyond Git reads."""
import json
import subprocess

import pytest
from scripts.prepare_native_hu_campaign import prepare


def test_preparation_records_identity_and_does_not_launch(monkeypatch, tmp_path):
    monkeypatch.setattr('scripts.prepare_native_hu_campaign.platform.platform', lambda: 'fixture-host')
    calls = []
    def git_read(command, **kwargs):
        calls.append(command)
        assert command[:2] == ['git', 'rev-parse'] or command[:2] == ['git', 'status']
        return 'abc123\n' if command[1] == 'rev-parse' else ''
    monkeypatch.setattr('scripts.prepare_native_hu_campaign.subprocess.check_output', git_read)
    binary = tmp_path/'fake-binary'; binary.write_bytes(b'fixture only')
    out = tmp_path/'prepared'
    plan = prepare('pilot', out, binary)
    assert plan['status'] == 'prepared-only'
    assert plan['stack_bb'] == 100 and plan['target_total_nodes'] == 10000000
    assert plan['recipe']['average'] == 'opponent-sampled'
    assert plan['command'][1] == 'train'
    assert plan['limits']['rss_gib'] == 5.5
    assert (out/'commands.txt').read_text().startswith('set -eu\n')
    assert not (out/'training-guard').exists()
    subprocess.run(["bash", "-n", str(out/"commands.txt")], check=True)
    assert len(calls) == 2
    assert json.loads((out/'plan.json').read_text()) == plan
    with pytest.raises(FileExistsError): prepare('pilot',out,binary)


def test_extension_requires_a_measured_quote_and_owner_receipt(monkeypatch,tmp_path):
    monkeypatch.setattr('scripts.prepare_native_hu_campaign.subprocess.check_output', lambda cmd, **kwargs: 'abc\n' if cmd[1] == 'rev-parse' else '')
    binary=tmp_path/'binary'; binary.write_bytes(b'fixture')
    with pytest.raises(ValueError,match='measured quote and owner'):
        prepare('extension',tmp_path/'out',binary)
    assert not (tmp_path/'out').exists()


def test_extension_binds_completed_pilot_and_rejects_incomplete_coverage(monkeypatch, tmp_path):
    import gzip
    from scripts.prepare_native_hu_campaign import SEED, file_hash
    from src.blueprint.artifact import save_training
    from tests.test_native_hu100_preparation import fixture
    monkeypatch.setattr('scripts.prepare_native_hu_campaign.platform.platform', lambda: 'fixture-host')
    monkeypatch.setattr('scripts.prepare_native_hu_campaign.subprocess.check_output', lambda cmd, **kw: 'abc\n' if cmd[1] == 'rev-parse' else '')
    binary = tmp_path/'binary'; binary.write_bytes(b'fixture')
    parent = tmp_path/'parent.gz'
    _, _, trainer = fixture(100)
    save_training(trainer, parent)
    lines = gzip.open(parent,'rt').readlines()
    header = json.loads(lines[0]); header['config']['seed'] = SEED
    header['average_rule'] = 'opponent-sampled'
    header['native_state'] = {'version':1, 'completed_nodes':10000000, 'coverage_start':[0,0,0],
                              'decisions_by_street':[1,1,1,7], 'traverser_visits_by_street':[1,1,1,7]}
    q = {'source':'abc', 'binary_sha256':file_hash(binary), 'seed':SEED, 'target_total_nodes':1000000000,
         'parent_path':str(parent), 'forecast_seconds':2000, 'training_forecast_seconds':1500,
         'audit_forecast_seconds':500, 'forecast_rss_gib':3, 'forecast_disk_free_gib':30,
         'max_entries':1000000, 'pilot_audit_verified':True}
    quote, approval = tmp_path/'quote.json',tmp_path/'approval.json'
    def seal():
        with gzip.open(parent,'wt') as target: target.write(json.dumps(header)+'\n'+''.join(lines[1:]))
        q['parent_sha256'] = file_hash(parent); quote.write_text(json.dumps(q))
        approval.write_text(json.dumps({'approved':True,'quote_sha256':file_hash(quote),
                                       'owner_instruction_url':'fixture://owner', 'approved_at':'fixture'}))
    seal()
    plan = prepare('extension',tmp_path/'accepted',binary,quote=quote,approval=approval)
    assert '--resume' in plan['command']
    assert plan['limits']['total_seconds'] == 7200
    assert plan['limits']['training_seconds'] == 6300
    subprocess.run(['bash','-n',str(tmp_path/'accepted'/'commands.txt')],check=True)
    header['native_state']['completed_nodes'] = 9999999; seal()
    with pytest.raises(ValueError,match='recoverable production'):
        prepare('extension',tmp_path/'short',binary,quote=quote,approval=approval)
    header['native_state']['completed_nodes'] = 10000000
    header['native_state']['traverser_visits_by_street'][3] = 0; seal()
    with pytest.raises(ValueError,match='recoverable production'):
        prepare('extension',tmp_path/'no-river',binary,quote=quote,approval=approval)
    header['native_state']['traverser_visits_by_street'][3] = 7
    header['native_state']['coverage_start'] = [1,100,1]; seal()
    with pytest.raises(ValueError,match='recoverable production'):
        prepare('extension',tmp_path/'legacy',binary,quote=quote,approval=approval)


def test_hu100_audit_rejects_unfinished_target_before_reading_exports(tmp_path):
    from src.blueprint.artifact import save_training
    from tests.test_native_hu100_preparation import fixture
    from scripts.audit_native_hu_checkpoint import inspect
    _, _, trainer = fixture(100)
    parent = tmp_path/'parent.gz'; save_training(trainer,parent)
    with pytest.raises(ValueError, match='Incomplete production HU100'):
        inspect(parent,tmp_path/'missing-current',tmp_path/'missing-average',100,10000000)
