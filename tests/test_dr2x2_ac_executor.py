"""Complete replay/visit evidence and outcome suppression under declared work."""
import gzip
import json
from pathlib import Path
import pytest
from scripts.evaluate_dr2x2_ac import execute, science_fields
from src.diagnostics.cfr_average import extract
from src.diagnostics.history_river import LAW, PROJECTION
from tests.test_dr2x2_evaluation_preflight import fixture
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, HU20_COMPRESSED_SCHEMA


def plan_fixture(tmp_path):
    models=[]
    for schema,cell in ((HU20_UNCAPPED_SCHEMA,'A'),(HU20_COMPRESSED_SCHEMA,'C')):
        spec,t=fixture(tmp_path,schema,cell)
        spec['iteration']=t.iteration
        path=tmp_path/(cell+'-average.gz')
        result=extract(Path(spec['checkpoint_path']),spec,path,expected_schema=schema)
        spec.update(average_path=str(path),average_sha256=result['sha256'])
        models.append(spec)
    return {'law':LAW,'projection':PROJECTION,'models':models,'chance_samples':4,'lbr_seconds':5,
            'admission_blocks':2,'panels':[{'name':'uniform','rule':'hu20_uniform','contract':'menu',
                'blocks':2,'readout_blocks':2,'primary_root':202610120104,'readout_root':202610120504,
                'serialization_root':202610120604}],
            'limits':{'admission_max_seconds':1200,'max_rss_gib':6,'min_free_disk_gib':0,'max_swap_growth_gib':.5}}


def test_serialized_admission_keeps_replay_visits_pairing_and_suppresses_outcomes(tmp_path):
    plan=plan_fixture(tmp_path)
    result=execute(plan,13,tmp_path/'admission',admission=True)
    assert result['status']=='complete' and result['closed_hands']==16
    assert not result['candidate_river_quality_computed']
    grouped=[]
    for task in result['tasks']:
        with gzip.open(tmp_path/'admission'/task['file'],'rt') as f:
            rows=[json.loads(line) for line in f]
        assert all(r['native_replay_verified'] and r['target_chips'] is None and r['tails'] is None
                   and r['net_chips_by_seat'] is None for r in rows)
        assert all(a['observation']['visits'] is not None
                   for r in rows for a in r['actions'] if a['logical_player']==0)
        assert len(task['scientific_fingerprints'])==4
        grouped.append([(r['block'],r['rotation'],r['deal_seed']) for r in rows])
    assert all(g==grouped[0] for g in grouped)
    with pytest.raises(FileExistsError):execute(plan,13,tmp_path/'admission',admission=True)


def test_strength_cannot_run_without_exact_linux_admission(tmp_path):
    plan=plan_fixture(tmp_path)
    with pytest.raises(ValueError,match='lease'):
        execute(plan,13,tmp_path/'strength')
    proof=tmp_path/'proof.json';proof.write_text(json.dumps({'admitted_plan_sha256':'wrong','linux_reference_match':True}))
    with pytest.raises(ValueError,match='Linux admission'):
        execute(plan,13,tmp_path/'strength',control=proof)
    assert not (tmp_path/'strength').exists()


def test_scientific_fingerprint_ignores_latency_but_retains_lbr_completion():
    a={'lbr':{'seconds':1,'preparation_seconds':.3,'completed':True},'actions':[{'kind':'fold','seconds':.1}]}
    b={'lbr':{'seconds':2,'preparation_seconds':.8,'completed':True},'actions':[{'kind':'fold','seconds':.2}]}
    assert science_fields(a)==science_fields(b)
    b['lbr']['completed']=False
    assert science_fields(a)!=science_fields(b)


def test_failed_gameplay_keeps_exact_id_and_partial_before_returning_failure(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path)
    def failed_play(source,spec,rules,contract,block,rotation,root,phase,config,emit,**kwargs):
        emit({'status':'failed','error':'fixture illegal action','actions':[],
              'block':block,'rotation':rotation,'root_seed':root,'deal_seed':42})
        raise RuntimeError('fixture illegal action')
    monkeypatch.setattr('scripts.evaluate_hu20_cards_v2.play',failed_play)
    result=execute(plan,13,tmp_path/'failed',admission=True)
    assert result['status']=='failed-retained' and result['closed_hands']==0
    failure=json.loads((tmp_path/'failed'/'failure.json').read_text())
    evidence=json.loads((tmp_path/'failed'/'failed-hand.json').read_text())
    assert failure['coordinate']==evidence['coordinate']
    assert failure['coordinate']['cell']=='A' and failure['coordinate']['block']==0 and failure['coordinate']['rotation']==0
    assert evidence['row']['error']=='fixture illegal action' and failure['no_retry']
    assert (tmp_path/'failed'/'manifest.json').exists()
