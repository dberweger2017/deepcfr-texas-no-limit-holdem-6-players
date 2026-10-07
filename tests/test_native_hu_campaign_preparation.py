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
