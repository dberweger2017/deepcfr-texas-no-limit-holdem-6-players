"""Scheduling failures must preserve the fixed campaign ceiling and evidence."""
import json
from pathlib import Path
import pytest
from scripts import pilot_hu200_capacity as p


def test_clock_cannot_be_reset_after_consumption(tmp_path):
    path=tmp_path/'clock.json'
    original=p.seal_clock(path,100,'source')
    assert original['deadline_epoch']==86500
    before=path.read_bytes()
    with pytest.raises(FileExistsError):p.seal_clock(path,200,'other')
    assert path.read_bytes()==before


def test_prospective_writers_cannot_spend_disk_floor_or_logical_budget():
    assert p.evaluation_fits(10_000_000_000,100_000_000_000,0,24_000_000_000)
    assert not p.evaluation_fits(10_000_000_000,70_000_000_000,0,24_000_000_000)
    assert not p.evaluation_fits(190_000_000_000,100_000_000_000,0,24_000_000_000)


def test_cache_pair_is_serial_before_each_fixed_phase(tmp_path):
    obj=p.Pilot.__new__(p.Pilot)
    obj.base=tmp_path;obj.phase=tmp_path/p.PHASE;obj.config_path=tmp_path/'config.json'
    assert obj.scientific_command({'name':'cache-large'})[-1]=='0'
    assert obj.scientific_command({'name':'cache-twin'})[-1]=='1'
    for phase in ('play','replay','reproduce','verify'):
        cmd=obj.scientific_command({'name':phase})
        assert cmd[-1]=='--pilot'
        assert phase in cmd
    assert 'readout' not in p.RESERVATIONS


def test_hash_mismatch_never_admits_restored_bytes(tmp_path):
    path=tmp_path/'model';path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):
        p.intact(path,{'bytes':7,'sha256':'0'*64})


def test_stop_ends_only_owned_stage_and_preserves_partial(tmp_path,monkeypatch):
    base=tmp_path;folder=base/p.PHASE;folder.mkdir();stop=folder/'cache-large.stop'
    partial=folder/'partial';partial.write_bytes(b'valuable failure evidence')
    class Child:
        returncode=None
        def poll(self):stop.write_text('VRM stop');return None
        def terminate(self):self.returncode=-15
        def wait(self,timeout=None):return self.returncode
    monkeypatch.setattr(p.subprocess,'Popen',lambda command:Child())
    with pytest.raises(RuntimeError,match='partial attempt retained'):
        p.run_stage(base,base/'config.json','cache-large',stop)
    assert partial.read_bytes()==b'valuable failure evidence'
