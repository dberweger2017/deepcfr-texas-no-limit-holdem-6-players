"""A terminal partial must never turn completed subsets into a poker readout."""
import json
import pytest
from scripts import closeout_hu100_v053_partial as partial


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_partial_inventory_rejects_any_existing_global_readout(tmp_path, monkeypatch):
    monkeypatch.setattr(partial, 'BASE', tmp_path)
    with pytest.raises(ValueError, match='Terminal scientific failure'):
        partial.inventory()
    write(tmp_path/'controller-failure.json', {'failure':'disk admission'})
    write(tmp_path/'summary.json', {'status':'verified'})
    with pytest.raises(ValueError, match='existing global readout'):
        partial.inventory()


def test_partial_verification_exposes_only_counts_and_costs(tmp_path):
    assert partial.verification(tmp_path)['status']=='not-completed'
    write(tmp_path/'costs.json', dict(blocks=4, fixed_seconds=2, block_work_seconds=3,
                                    forbidden_poker_result='must remain unread'))
    write(tmp_path/'replay.json', dict(status='verified', hands_replayed=8,
                                     coverage={'forbidden':'must remain unread'}))
    write(tmp_path/'reproduction/complete.json', dict(reproduced_all_hands_and_decisions=True))
    write(tmp_path/'reached-key-coverage.json', dict(status='verified', coverage={'forbidden':1}))
    result=partial.verification(tmp_path)
    assert result==dict(status='replayed-and-reproduced', blocks=4, hands=8,
                       selected_coverage_verified=True, fixed_seconds=2, block_work_seconds=3)
    write(tmp_path/'reproduction/complete.json', dict(reproduced_all_hands_and_decisions=False))
    assert partial.verification(tmp_path)['status']=='incomplete'


def test_archive_reuses_member_readback_to_match_audited_model(tmp_path):
    import hashlib
    model=tmp_path/'model';model.write_bytes(b'verified-model')
    pin=dict(bytes=model.stat().st_size,sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    receipt=partial.seal_models([(model,'average')],tmp_path/'good.zip',{}, {'average':pin})
    assert receipt['matches_existing_model_audit_pins']
    assert receipt['all_member_sizes_and_hashes_readback']
    with pytest.raises(ValueError,match='existing verified'):
        partial.seal_models([(model,'average')],tmp_path/'bad.zip',{}, {'average':dict(pin,sha256='changed')})
