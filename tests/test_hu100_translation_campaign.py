"""Outcome blindness and small complete telemetry/replay/reproduction integration."""
import json
from pathlib import Path
import subprocess

from scripts.run_hu100_action_translation import quote
from scripts.evaluate_native_hu100_baseline import execute
from scripts.audit_native_hu100_baseline import audit
from src.policies.files import file_hash
from tests.test_native_hu100_baseline import model_fixture


def test_quote_reads_only_cost_fields(tmp_path):
    folder=tmp_path/'pilot';folder.mkdir()
    for name in ('disabled','enabled'):
        for suffix in ('','-reproduction'):
            path=folder/(name+suffix);path.mkdir()
            (path/'complete.json').write_text(json.dumps({'model_load_seconds':2.,'wall_seconds':10.,'payoff':-9999}))
        (folder/(name+'-audit.json')).write_text(json.dumps({'seconds':3.,'panels':{'payoff':-9999}}))
    a=quote(folder,'source')
    for path in folder.rglob('*.json'):
        path.write_text(path.read_text().replace('-9999','9999'))
    assert quote(folder,'source')==a
    assert a['blocks_per_opponent']==2048 and not a['pilot_outcomes_read']
    assert a['execution_budget_seconds']==3*(8+38*128)+120


def test_small_translation_campaign_replays_and_reproduces(tmp_path):
    _,model,path=model_fixture(tmp_path)
    settings=json.loads(Path('configs/arena/hu100-playing-baseline-v1.json').read_text())
    settings['model'].update(name='fixture',path=str(path),sha256=file_hash(path),
        bytes=path.stat().st_size,entries=1,iteration=2,
        source_checkpoint_sha256=model.description['source_checkpoint_sha256'])
    settings['action_translation']={'max_states':512,'max_events':128}
    config=tmp_path/'config.json';config.write_text(json.dumps(settings))
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    result=execute(config,tmp_path/'first',2,129802051,revision)
    assert result['status']=='complete'
    checked=audit(tmp_path/'first',tmp_path/'audit.json')
    assert checked['all_settlements_replayed']
    repeated=execute(config,tmp_path/'second',2,129802051,revision,reproduce=tmp_path/'first')
    assert repeated['reproduced_all_hands_and_decisions']


def test_guard_reads_ac_on_each_iteration_and_stops_transient_breach(monkeypatch,tmp_path):
    import scripts.run_hu100_action_translation as campaign
    from types import SimpleNamespace
    import pytest
    monkeypatch.setattr(campaign,'OUT',tmp_path)
    power=iter(['AC Power','AC Power','Battery Power'])
    def output(command,**kwargs):
        return next(power) if command[0]=='pmset' else 'used = 0.00M'
    monkeypatch.setattr(campaign.subprocess,'check_output',output)
    monkeypatch.setattr(campaign,'memory_snapshot',lambda:{'pressure_level':1,'free_percent':90})
    monkeypatch.setattr(campaign.psutil,'virtual_memory',lambda:SimpleNamespace(used=1024))
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda path:SimpleNamespace(free=30*1024**3))
    fake=SimpleNamespace(children=lambda recursive:[],memory_info=lambda:SimpleNamespace(rss=1024))
    monkeypatch.setattr(campaign.psutil,'Process',lambda pid:fake)
    child=SimpleNamespace(poll=lambda:None,returncode=0)
    monkeypatch.setattr(campaign.subprocess,'Popen',lambda *args,**kwargs:child)
    stopped=[];monkeypatch.setattr(campaign,'terminate_child',lambda process:stopped.append(process))
    monkeypatch.setattr(campaign,'sleep',lambda seconds:None)
    with pytest.raises(RuntimeError,match='AC power guard'):
        campaign.operation('fixture','unused',[],tmp_path)
    receipt=json.loads((tmp_path/'fixture-guard/receipt.json').read_text())
    assert receipt['samples']==2 and receipt['host_sample_seconds']==.2
    assert receipt['status']=='failed' and stopped==[child]
