from dataclasses import asdict
import gzip
import json
from pathlib import Path
import subprocess

import pytest

from scripts import evaluate_hu100_direct as direct
from scripts import run_hu100_k50_2b as campaign
from src.arena.schedule import build_schedule, canonical
from src.blueprint import equity_buckets as buckets
from src.blueprint.abstraction import choices, HU100_SCHEMA, HU100_EQUITY_SCHEMA
from src.game.hand import Hand, Table
from src.game.types import ActionKind
from src.policies.files import file_hash
from tests.native_equity_tables import write_tables
from tests.test_k50_evaluation import write_model


def test_guards_and_cap_keep_owner_limits():
    assert campaign.CAP*campaign.BPE*1.1+campaign.TABLE_BYTES+campaign.RESERVE <= campaign.SOFT
    assert (campaign.CAP+1)*campaign.BPE*1.1+campaign.TABLE_BYTES+campaign.RESERVE > campaign.SOFT
    baseline={'deadline':100,'swap_bytes':1000}
    sample={'pressure_level':1,'free_percent':15,'swap_bytes':1000+512*1024**2,'disk_free_bytes':16*1024**3,'ac':True}
    assert campaign.limits(sample,campaign.SOFT-1,baseline,99) is None
    for key,value in [('pressure_level',2),('free_percent',14),('swap_bytes',sample['swap_bytes']+1),('disk_free_bytes',sample['disk_free_bytes']-1),('ac',False)]:
        assert campaign.limits(dict(sample,**{key:value}),0,baseline,99)
    assert campaign.limits(sample,campaign.SOFT,baseline,99)
    assert campaign.limits(sample,campaign.HARD,baseline,99)
    assert campaign.limits(sample,0,baseline,100)


def test_mixed_hu100_trace_independent_replay_and_reproduction(tmp_path,monkeypatch):
    blocks,root=8,54189
    # Build exact tiny tables for every street/holding in these paired deals.
    dummy=[{'name':name,'path':'fixture','sha256':'a'*64,'format':'holdem-hu100-stored-cfr-average-research-v1'} for name in ('k50','v1')]
    plan=direct.make_plan(dummy,blocks,root,'fixture')
    situations=[];observations=[]
    for block in build_schedule(plan):
        hand=Hand.start(Table(('a','b'),(10000,10000),block.button,50,100,'0.01'),hand_id='fixture',seed=block.deal_seeds[0])
        observations.append(hand.observe(hand.actor))
        while not hand.finished:
            view=hand.observe(hand.actor)
            if view.board:
                situations.extend((hand.observe(seat).hole_cards,view.board) for seat in (0,1))
            menu=choices(view,raise_cap=None,free_fold=False)
            action=next(c.action for c in menu if c.action.kind in (ActionKind.CHECK,ActionKind.CALL))
            hand=hand.apply(action)
    (tmp_path/'tables').mkdir()
    tables=write_tables(tmp_path/'tables',situations)
    cards=buckets.EquityCards(tables,pinned=False)
    monkeypatch.setattr(buckets,'EQUITY_K50_SHA256',cards.sha256)
    specs=[]
    for name,schema,owned in [('k50',HU100_EQUITY_SCHEMA,cards),('v1',HU100_SCHEMA,None)]:
        path=tmp_path/(name+'.gz')
        # One stored key is sufficient; the other reached keys exercise uniform missing-key fallback.
        write_model(path,schema,owned,observations[:1])
        spec={'name':name,'path':str(path),'sha256':file_hash(path),'bytes':path.stat().st_size,
            'format':'holdem-hu100-stored-cfr-average-research-v1','entries':1,'iteration':10,
            'source_checkpoint_sha256':'a'*64,'abstraction':schema}
        if owned:spec['card_buckets']=str(tables)
        specs.append(spec)
    frozen=direct.make_plan(specs,blocks,root,'fixture')
    assert frozen.models[0].abstraction==HU100_EQUITY_SCHEMA and frozen.models[0].card_buckets==str(tables)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    real=direct.subprocess.check_output
    monkeypatch.setattr(direct.subprocess,'check_output',lambda cmd,**kw:'' if cmd[:2]==['git','status'] else real(cmd,**kw))
    registry,_=direct.execute(specs,blocks,root,'fixture',tmp_path/'play',source)
    replay=direct.audit(tmp_path/'play',tmp_path/'audit.json',cost_only=True)
    assert replay['hands_replayed']==2*blocks and 'bb_per_100' not in replay and 'ci95' not in replay
    assert 'paired_sd_bb_per_100' in replay
    _,repeat=direct.execute(specs,blocks,root,'fixture',tmp_path/'repeat',source,registry,tmp_path/'play')
    assert repeat['reproduced_all_hands_and_decisions']
    # A self-consistent raw-file hash cannot hide a foreign policy key from independent replay.
    trace=tmp_path/'play/decisions.jsonl.gz'
    with gzip.open(trace,'rt') as f:rows=list(map(json.loads,f))
    rows[0]['key']='f'*32
    with gzip.open(trace,'wt') as f:
        for row in rows:f.write(canonical(row)+'\n')
    receipt=tmp_path/'play/complete.json';done=json.loads(receipt.read_text())
    done['files'][trace.name]={'bytes':trace.stat().st_size,'sha256':file_hash(trace)}
    receipt.write_text(json.dumps(done))
    with pytest.raises(ValueError,match='key differs'):
        direct.audit(tmp_path/'play',tmp_path/'corrupt-replay.json')


def test_readout_uses_frozen_schedule_for_seats_and_binds_replayed_bytes(tmp_path,monkeypatch):
    # Existing HU100 traces do not repeat the button: it belongs to the frozen schedule.
    monkeypatch.setattr(campaign,'BASE',tmp_path)
    folder=tmp_path/'final/primary/play';folder.mkdir(parents=True)
    trace=folder/'hands.jsonl.gz'
    with gzip.open(trace,'wt') as f:
        for block,rotation,chips in [(0,0,10),(0,1,20),(1,0,30),(1,1,60)]:
            f.write(json.dumps({'block':block,'rotation':rotation,'candidate_chips':chips})+'\n')
    done={'files':{trace.name:{'bytes':trace.stat().st_size,'sha256':file_hash(trace)}}}
    (folder/'complete.json').write_text(json.dumps(done))
    (folder.parent/'replay.json').write_text(json.dumps({'status':'verified','coverage':{}}))
    dummy=[{'name':name,'path':'fixture','sha256':'a'*64,'format':'holdem-hu100-stored-cfr-average-research-v1'} for name in ('k50','v1')]
    (tmp_path/'canonical-plan.json').write_text(json.dumps({'pairs':{'primary':dummy},'counts':{'primary':2},'roots':{'primary':61}}))
    (tmp_path/'all-final-verified.json').write_text(json.dumps({'status':'verified','canonical_sha256':file_hash(tmp_path/'canonical-plan.json')}))
    (tmp_path/'treatment.json').write_text(json.dumps({'primary_reached':True}))
    campaign.readout()
    result=json.loads((tmp_path/'summary.json').read_text())['results']['primary']
    assert result['bb_per_100']==30
    assert result['seats']['button']['bb_per_100']==35
    assert result['seats']['big_blind']['bb_per_100']==25
    assert result['absolute_returns']['v1']=={'overall':-30,'button':-25,'big_blind':-35}
    trace.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed after replay'):campaign.readout()


def test_preservation_permission_cannot_restart_scientific_work(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'BASE',tmp_path)
    (tmp_path/'campaign-failure.json').write_text('{}')
    with pytest.raises(ValueError,match='Terminal failure'):
        campaign.guard('pilot',['.venv/bin/python','-m','scripts.run_hu100_k50_2b','pilot'])
    with pytest.raises(ValueError,match='only admits the archive'):
        campaign.guard('archive',['.venv/bin/python','-m','scripts.run_hu100_k50_2b','pilot'],preservation_only=True)


def test_match_continuation_amends_only_pressure_and_swap():
    from scripts import run_hu100_k50_match_continuation as continuation
    baseline={'deadline':100,'swap_bytes':1000}
    sample={'pressure_level':2,'free_percent':15,'swap_bytes':1000+2_000_000_000,'disk_free_bytes':16*1024**3,'ac':True}
    assert continuation.limits(sample,campaign.SOFT-1,baseline,99) is None
    assert campaign.limits(sample,campaign.SOFT-1,baseline,99)
    for key,value in [('pressure_level',4),('pressure_level',0),('free_percent',14),
                      ('swap_bytes',sample['swap_bytes']+1),('disk_free_bytes',sample['disk_free_bytes']-1),('ac',False)]:
        assert continuation.limits(dict(sample,**{key:value}),0,baseline,99)
    assert continuation.limits(sample,campaign.SOFT,baseline,99)
    assert continuation.limits(sample,campaign.HARD,baseline,99)
    assert continuation.limits(sample,0,baseline,100)
