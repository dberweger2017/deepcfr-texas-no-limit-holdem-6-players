"""Meaningful comparison regressions: support, pairing and full replay."""
from types import SimpleNamespace

import pytest

from scripts.evaluate_hu200_diagnosis import exposure, interval, play_hand, seed, verify_rows
from scripts.diagnose_native_hu100 import signature, support_witness
from src.blueprint.abstraction import HU200_SCHEMA, choices, information_key
from src.blueprint.solver import HU200_GAME
from src.arena.policies import NativeHU200Uniform
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


class Uniform:
    visits = {}
    def policy(self, value):
        return NativeHU200Uniform(value)
    def distribution_with_telemetry(self, view):
        menu=choices(view,raise_cap=None,free_fold=False)
        return menu,(1/len(menu),)*len(menu),False,dict(reason='missing-key',mode='uniform')


def test_hu200_off_menu_supported_alias_and_unsupported_open_jam():
    initial=Hand.start(Table(('a','b'),(20000,20000)),hand_id='support',seed=7)
    for amount,expected in [(400,'supported'),(20000,'unsupported-abstract-history-or-menu')]:
        hand=initial.apply(Action(ActionKind.RAISE,amount));view=hand.observe(hand.actor)
        menu=choices(view,raise_cap=None,free_fold=False)
        proof=support_witness(initial,signature(view,menu))
        assert proof['support']==expected
        if expected=='supported':
            witness=initial
            for a in proof['witness_actions']:
                witness=witness.apply(Action(ActionKind(a['kind']),a['raise_to']))
            wv=witness.observe(witness.actor)
            assert information_key(view,menu,schema=HU200_SCHEMA)==information_key(wv,choices(wv,raise_cap=None,free_fold=False),schema=HU200_SCHEMA)


def test_paired_streams_deals_and_full_replay():
    for seat in (0,1):
        a,fixture=play_hand(Uniform(),42,'pot_pressure',0,seat,{},[0.])
        b,_=play_hand(Uniform(),42,'pot_pressure',0,seat,{},[0.])
        assert a==b
        assert sum(a['final_stacks'])==40000
        assert fixture['stack_bb']==200
    left,_=play_hand(Uniform(),42,'random',1,0,{},[0.])
    right,_=play_hand(Uniform(),42,'random',1,1,{},[0.])
    assert left['holes']==right['holes']
    assert left['deal_seed']==right['deal_seed']
    assert left['candidate_seed']!=right['candidate_seed']
    assert seed(42,'deal','random',1)!=seed(43,'deal','random',1)


def test_bonferroni_interval_wider_and_block_units():
    values=[-2,0,3,8]
    ordinary=interval(values);adjusted=interval(values,5)
    assert adjusted['blocks']==4
    assert adjusted['bb_per_100']==ordinary['bb_per_100']
    assert adjusted['interval'][0]<ordinary['interval'][0]
    assert adjusted['interval'][1]>ordinary['interval'][1]
    assert interval([1,1])['reason']=='no-observed-variation'


def test_disjoint_hand_category_priority():
    def d(s='stored-training-witness',l='positive-mass'):
        return dict(support=s,lookup=l)
    assert exposure([])=='no-decision'
    assert exposure([d(),d('supported-observed-menu-path','missing-key')])=='supported-missing'
    assert exposure([d('unsupported-abstract-history-or-menu','missing-key'),d('unresolved','missing-key')])=='ever-unsupported'
    assert exposure([d(),d(l='zero-mass')])=='zero-mass-no-missing'
    assert exposure([d()])=='all-positive'


def test_incomplete_schedule_rejected(tmp_path):
    import gzip
    p=tmp_path/'empty.gz'
    with gzip.open(p,'wt'):pass
    with pytest.raises(ValueError,match='Incomplete'):
        list(verify_rows(p,42,32))


def test_timing_quote_keeps_loading_fixed_and_reserves_closeout():
    from scripts.run_hu200_diagnosis import sample_quote
    costs=[dict(model_load_seconds=100,play_replay_seconds=2),dict(model_load_seconds=30,play_replay_seconds=3)]
    options=sample_quote(costs,1,3600,1024,100*1024**3)
    full=options[0]
    assert full['blocks']==2048
    assert full['upper_seconds']==2*(130+6*64)+600
    assert full['admitted']
    assert not sample_quote(costs,1,100,1024,100*1024**3)[-1]['admitted']
    assert not sample_quote(costs,1,3600,1024,15*1024**3)[-1]['admitted']


def test_archive_readback_preserves_partials_and_excludes_mutable_seal(tmp_path):
    from scripts.run_hu200_diagnosis import seal
    from zipfile import ZipFile
    import json
    root=tmp_path/'root';root.mkdir()
    (root/'failure.json').write_text('{"failure":"fixture"}')
    (root/'operations/archive').mkdir(parents=True)
    (root/'operations/archive/live-log').write_text('mutable')
    archive=tmp_path/'evidence.zip';seal(root,archive)
    receipt=json.loads((root/'archive-receipt.json').read_text())
    assert receipt['verified_members']==1
    assert not receipt['cloud_acceptance_claimed']
    with ZipFile(archive) as z:
        assert z.read('failure.json')==b'{"failure":"fixture"}'
        assert 'operations/archive/live-log' not in z.namelist()
    with pytest.raises(FileExistsError):seal(root,archive)


def test_independent_report_recount_rejects_changed_primary_and_coverage(tmp_path, monkeypatch):
    import json
    from scripts import evaluate_hu200_diagnosis as evaluator
    from scripts.report_hu200_diagnosis import produce
    model=Uniform();model.description={'entries':0};model.translation=None
    monkeypatch.setattr(evaluator,'AveragePolicy',lambda *a,**kw:model)
    plan={'models':[{'target':n,'path':'fake','sha256':'fake','entries':0} for n in (20000000,100000000)],
          'final_root':42,'timing_root':43,'blocks':2}
    root=tmp_path/'run';root.mkdir()
    (root/'frozen-plan.json').write_text(json.dumps(plan))
    for m in plan['models']:
        evaluator.worker(plan,'final',m['target'],root/'final'/str(m['target']))
    evaluator.report(plan,root)
    produce(root,tmp_path/'valid')
    summary=json.loads((root/'summary.json').read_text())
    summary['gains']['random']['bb_per_100']+=1
    (root/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='primary estimate'):
        produce(root,tmp_path/'changed-primary')
    summary['gains']['random']['bb_per_100']-=1
    (root/'summary.json').write_text(json.dumps(summary))
    coverage=json.loads((root/'coverage-behavior.json').read_text())
    coverage[0]['decisions']+=1
    (root/'coverage-behavior.json').write_text(json.dumps(coverage))
    with pytest.raises(ValueError,match='coverage/action'):
        produce(root,tmp_path/'changed-coverage')
