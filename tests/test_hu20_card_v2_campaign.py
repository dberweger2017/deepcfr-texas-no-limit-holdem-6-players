"""Small generated policies exercise the paid campaign gates without rentals."""
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import time
import tarfile
import pytest

from scripts.evaluate_hu20_cards_v2 import hand
from scripts.hu20_cards_v2_rental_guard import validate
from scripts.run_hu20_cards_v2_rentals import verify_archive
from scripts.train_hu20_cards_v2 import fingerprint,run
from src.blueprint.abstraction import HU20_CARD_V2_SCHEMA,HU20_UNCAPPED_SCHEMA
from src.blueprint.artifact import export_policy,save_training,HU20_UNCAPPED_FORMAT
from src.blueprint.solver import BlueprintTrainer,PilotConfig,HU20_UNCAPPED_GAME
from src.diagnostics.saved_hu20 import file_hash,load_saved
from src.game.hand import Table

ROOT=Path(__file__).resolve().parents[1]


def test_explicit_schema_and_actual_replay_for_both_arms(tmp_path):
    evaluation={'chance_samples':4,'lbr_seconds':5}
    for version,schema in [('v1',HU20_UNCAPPED_SCHEMA),('v2',HU20_CARD_V2_SCHEMA)]:
        cfg=PilotConfig(seed=2026093001,abstraction=schema,game=HU20_UNCAPPED_GAME,raise_cap=None,max_nodes=10000)
        t=BlueprintTrainer(Table(('player-0','player-1'),(2000,2000)),cfg)
        for _ in range(3):t.step()
        cp=tmp_path/(version+'-checkpoint.gz');p=tmp_path/(version+'-policy.gz')
        spec={'name':version,'seed':cfg.seed,'players':2,'abstraction':schema,'version':version,
              'path':p.name,'checkpoint_path':cp.name,'format':HU20_UNCAPPED_FORMAT,
              'checkpoint_sha256':save_training(t,cp),'sha256':export_policy(t,p)}
        if version=='v2':
            with pytest.raises(ValueError,match='frozen HU20 lineage'):load_saved(spec,tmp_path)
        source,visits=load_saved(spec,tmp_path,expected_schema=schema)
        for contract in ('menu','native'):
            panel={'name':contract,'rule':'pressure','contract':contract,'root':202610010900}
            rows=[hand(source,visits,spec,panel,0,0,evaluation) for _ in range(2)]
            assert rows[0]['native_replay_verified'] and rows[0]['target_chips']==rows[1]['target_chips']
            assert rows[0]['event_digest']==rows[1]['event_digest']
            assert rows[0]['tails']['counts']['hands']==1
            assert rows[0]['tails']['counts']['target_decisions']==rows[0]['tails']['counts']['trained']+rows[0]['tails']['counts']['fallback']
            assert {a['observation']['key'] for a in rows[0]['actions'] if a['logical_player']==0}=={k['key'] for k in rows[0]['reached_keys']}
        timing=hand(source,visits,spec,panel,0,0,evaluation,True)
        assert timing['target_chips'] is None and timing['net_chips_by_seat'] is None
        assert all('observation' not in a for a in timing['actions'])


def test_small_complete_iteration_train_and_stream_hash(tmp_path):
    plan=json.loads((ROOT/'configs/diagnostics/hu20-card-v2-run.json').read_text())
    plan.update(nodes_per_seed=1000,checkpoint_every_nodes=500)
    out=tmp_path/'train';r=run(plan,plan['seeds'][0],out,time()+30)
    assert r['status']=='complete' and r['completed_nodes']>=1000 and r['overshoot_nodes']>=0
    data=(out/'current.json.gz').read_bytes();f=fingerprint(out/'current.json.gz')
    assert f['sha256']==sha256(data).hexdigest()
    assert f['uncompressed_sha256']==sha256(gzip.decompress(data)).hexdigest()
    rows=[json.loads(line) for line in (out/'iterations.jsonl').read_text().splitlines()]
    assert sum(row['nodes'] for row in rows)==r['completed_nodes']
    assert r['checkpoint_milestones'][-1]['completed_nodes']==r['completed_nodes']
    with pytest.raises(FileExistsError):run(plan,plan['seeds'][0],out,time()+30)
    altered={**plan,'descriptor_sha256':'0'*64}
    with pytest.raises(ValueError,match='descriptor'):run(altered,plan['seeds'][0],tmp_path/'wrong',time()+30)


def test_approved_exact_name_cost_and_cutoff():
    lease={'names':[f'new-guy-hu20-card-v2-{i}-123' for i in range(3)],'started':100,'deadline':18100,'max_hourly_per_pod':.57}
    validate(lease)
    for changed in [{'deadline':18101},{'max_hourly_per_pod':.58},{'names':lease['names'][:2]},
                    {'names':['dr2x2-other',*lease['names'][1:]]},{'names':[lease['names'][0]]*3},{'prior_cost_upper_usd':2},{'prior_cost_upper_usd':-1}]:
        with pytest.raises(ValueError):validate({**lease,**changed})


def test_archive_transport_and_work_hash_before_teardown(tmp_path):
    source=tmp_path/'source/results/work';source.mkdir(parents=True);(source/'x.txt').write_text('retained')
    (source/'manifest.json').write_text(json.dumps({'x.txt':{'bytes':8,'sha256':file_hash(source/'x.txt')}}))
    target=tmp_path/'target';target.mkdir();archive=target/'results.tar'
    with tarfile.open(archive,'w') as tar:tar.add(source.parent,arcname='results')
    (target/'results.tar.sha256').write_text(file_hash(archive)+'  results.tar\n');verify_archive(target)
    assert (target/'verified.json').exists()
    (target/'results.tar.sha256').write_text('0'*64+'  results.tar\n')
    with pytest.raises(ValueError,match='transport'):verify_archive(target)


def test_full_fixture_comparison_metrics_from_actual_raw_hands(tmp_path,monkeypatch):
    import scripts.evaluate_hu20_cards_v2 as evaluation
    model_specs=[]
    for version,schema in [('v1',HU20_UNCAPPED_SCHEMA),('v2',HU20_CARD_V2_SCHEMA)]:
        t=BlueprintTrainer(Table(('player-0','player-1'),(2000,2000)),PilotConfig(seed=7,
             abstraction=schema,game=HU20_UNCAPPED_GAME,raise_cap=None))
        t.step();cp=tmp_path/(version+'-cp.gz');path=tmp_path/(version+'-policy.gz')
        spec={'name':version,'seed':7,'players':2,'version':version,'abstraction':schema,'format':HU20_UNCAPPED_FORMAT,
              'path':path.name,'checkpoint_path':cp.name,'checkpoint_sha256':save_training(t,cp),'sha256':export_policy(t,path)}
        model_specs.append((spec,tmp_path))
    monkeypatch.setattr(evaluation,'specs',lambda *args:model_specs)
    plan=json.loads((ROOT/'configs/diagnostics/hu20-card-v2-run.json').read_text())
    e=plan['evaluation'];e.update(timing_blocks=1,chance_samples=1)
    e['panels']=[{'name':'pressure','rule':'pressure','contract':'native','blocks':2},
                 {'name':'lbr','rule':'lbr','contract':'menu','blocks':2}]
    result=evaluation.evaluate(plan,7,tmp_path,{},tmp_path/'comparison',time()+60)
    assert result['status']=='complete' and result['hands']=={'v1':8,'v2':8}
    with gzip.open(tmp_path/'comparison/hands.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
    from scripts.summarize_hu20_cards_v2 import summarize_records
    independent=summarize_records(tmp_path/'comparison/hands.jsonl.gz')
    assert independent['hands']==16
    for panel in result['panels']:
        recomputed=independent['panels'][panel['panel']]
        assert recomputed['paired_v2_minus_v1_bb_per_100']==panel['paired_v2_minus_v1_bb_per_100']
        selected=[r for r in rows if r['panel']==panel['panel']]
        a=[r['target_chips'] for r in selected if r['version']=='v1'];b=[r['target_chips'] for r in selected if r['version']=='v2']
        assert panel['paired_v2_minus_v1_bb_per_100']['bb_per_100']==(sum(b)-sum(a))/4
        for version in ('v1','v2'):
            assert panel['per_arm'][version]['tails']==recomputed['per_arm'][version]['tails']
            hist=recomputed['per_arm'][version]['visit_bands_by_street']
            assert sum(d['decisions'] for d in hist.values())==panel['per_arm'][version]['tails']['target_decisions']
            assert panel['per_arm'][version]['tails']['hands']==4
            assert panel['per_arm'][version]['bb_per_100']['blocks']==2
        assert len({(r['block'],r['rotation'],r['deal_seed']) for r in selected})==4
