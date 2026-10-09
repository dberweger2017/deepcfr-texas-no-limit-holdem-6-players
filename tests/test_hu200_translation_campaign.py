"""Pairing, failure admission, complete replay and report tamper regressions."""
import json
from pathlib import Path

import pytest

from scripts import evaluate_hu200_translation as ev
from scripts.run_hu200_translation import GIB, problem, sample_quote, seal
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy
from src.blueprint.abstraction import HU200_SCHEMA
from src.blueprint.solver import HU200_GAME


def uniform():
    m=AveragePolicy.__new__(AveragePolicy);m.players=2;m.raise_cap=None;m.abstraction=HU200_SCHEMA;m.game=HU200_GAME
    m.entries={};m.zero_mass=set();m.visits={};m.description={'entries':0};m.record_translation=True;m.configure_translation(None)
    return m


def plan():
    return dict(model=dict(path='fake',sha256='fake',entries=0),translation={'max_states':512,'max_events':128},
                timing_root=43,final_root=42,blocks=2)


def test_pairing_and_complete_action_telemetry_reproduction():
    m=uniform()
    for seat in (0,1):
        a,_,_=ev.play_hand(m,42,'pot_pressure',0,seat)
        b,_,_=ev.play_hand(m,42,'pot_pressure',0,seat)
        for row in (a,b):
            for d in row['decisions']:d['translation'].pop('lookup_seconds')
            row.pop('sha256')
        assert a==b and sum(a['final_stacks'])==40000
    a,_,_=ev.play_hand(m,42,'random',0,0);b,_,_=ev.play_hand(m,42,'random',0,1)
    assert a['deal_seed']==b['deal_seed'] and a['holes']==b['holes']
    assert a['candidate_seed']!=b['candidate_seed']


def test_independent_selected_witness_is_checked_and_rejects_fabrication():
    from tests.test_hu200_action_translation import initial,witness
    from tests.test_hu100_action_translation import stored
    from src.game.types import Action,ActionKind
    start=initial();v=start.apply(Action(ActionKind.RAISE,550)).observe(1);menu,key=witness(v)
    m=uniform();m.entries[key]=stored(menu);m.configure_translation(TranslationOptions())
    _,_,_,info=m.distribution_with_telemetry(v)
    ev.verify_witness(start,v,menu,info,m)
    with pytest.raises(ValueError):ev.verify_witness(start,v,menu,{**info,'distance':info['distance']+1},m)
    with pytest.raises(ValueError):ev.verify_witness(start,v,menu,{**info,'witness_raise_to':(550,)},m)


def test_loading_fixed_and_timing_only_sample_budget():
    c=dict(model_load_seconds=100,play_replay_seconds=5)
    full=sample_quote(c,1,3600,1024,100*GIB,15.5*GIB)[0]
    assert full['upper_seconds']==2*(100+6*64)+600 and full['admitted']
    assert not sample_quote(c,1,700,1024,100*GIB,15.5*GIB)[-1]['admitted']
    assert not sample_quote(c,1,3600,1024,15.5*GIB,15.5*GIB)[-1]['admitted']


def test_soft_hard_swap_and_disk_guards_latch_before_more_science(tmp_path):
    from scripts.run_hu200_translation import Guard
    s=dict(pressure=1,free_percent=60,swap_bytes=0,disk_free_bytes=20*GIB,ac=True)
    assert problem(s,0,3*GIB,15.5*GIB)=='soft family RSS'
    assert problem(s,0,4*GIB,15.5*GIB)=='hard family RSS'
    assert problem({**s,'swap_bytes':513*1024**2},0,0,15.5*GIB)=='swap limit'
    assert problem({**s,'disk_free_bytes':15*GIB},0,0,15.5*GIB)=='disk floor'
    g=Guard(tmp_path,0,s,15.5*GIB);g.failed=True
    with pytest.raises(RuntimeError,match='Terminal science'):g.run('final',['false'])


def test_worker_report_full_sample_and_tamper(tmp_path,monkeypatch):
    monkeypatch.setattr(ev,'AveragePolicy',lambda *a,**kw:uniform())
    p=plan();root=tmp_path/'run';root.mkdir();ev.worker(p,'final',root/'final');ev.report(p,root)
    summary=json.loads((root/'summary.json').read_text())
    assert summary['counts']['hands']==40
    assert all(summary['behavior_identical'].values())
    assert all(d['bb_per_100']==0 for d in summary['gains'].values())
    # Foreign plans and changed evidence fail before any inference.
    with pytest.raises(ValueError,match='Unverified worker'):ev.report({**p,'blocks':3},root)
    hashes=root/'final/on/hashes.json';data=json.loads(hashes.read_text());data['hands_sha256']='changed';hashes.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Worker bytes changed'):ev.report(p,root)


def test_archive_preserves_failure_partial_and_restoration_history(tmp_path):
    from zipfile import ZipFile
    root=tmp_path/'root';root.mkdir();(root/'failure.json').write_text('{"failure":"fixture"}')
    (root/'partial').write_text('partial');(root/'operations/archive').mkdir(parents=True)
    (root/'operations/archive/live').write_text('mutable')
    archive=tmp_path/'evidence.zip';seal(root,archive)
    receipt=json.loads((root/'archive-receipt.json').read_text());assert receipt['verified_members']==2
    with ZipFile(archive) as z:
        manifest=json.loads(z.read('ARCHIVE-MANIFEST.json'))
        assert all(m['original_path'].startswith(str(root)) and m['mtime_ns'] for m in manifest['members'])
        assert z.read('partial')==b'partial' and 'operations/archive/live' not in z.namelist()
    assert not receipt['cloud_acceptance_claimed']
    with pytest.raises(FileExistsError):seal(root,archive)
