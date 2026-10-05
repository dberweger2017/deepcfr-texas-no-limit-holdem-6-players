"""The published diagnostic aliases must bind exact selected artifact bytes."""
import json
from pathlib import Path

import pytest

from scripts.freeze_hu20_search_references import freeze
from src.blueprint.hu20_turn_solver import file_hash


def corpus(tmp_path, monkeypatch):
    report=tmp_path/'published-report.md';report.write_text('Final diagnostic evidence\n')
    def git(args, **kwargs):
        return report.read_bytes() if args[1]=='show' else 'origin/diagnostic\n'
    monkeypatch.setattr('scripts.freeze_hu20_search_references.subprocess.check_output',git)
    run=tmp_path/'run';run.mkdir()
    (run/'result.json').write_text(json.dumps({'status':'completed','jobs_done':288,'jobs_total':288}))
    planned={'models':[{'name':f'base-{seed}-average','seed':seed,'strategy':'average',
                       'sha256':str(seed),'path':f'{seed}.average.gz'} for seed in (1,2,3)]}
    for seed in (1,2,3):
        for index in range(48):
            directory=run/'spots'/f'{seed}-{index}';directory.mkdir(parents=True)
            original=dict(planned['models'][seed-1],name=f'old-{seed}-stored-average',strategy='stored-average')
            (directory/'job.json').write_text(json.dumps({'policy':original,'job':{'root':{'spot':str(index)}}}))
            prepared=directory/'attempt-01'/'prepared';prepared.mkdir(parents=True)
            request=prepared/'request.json';request.write_text(json.dumps({'ranges':[['reference'],['law']]}))
            (directory/'result.json').write_text(json.dumps({'event':'spot_complete','gates':[{'gate':'V1','passed':True}],
                'attempt_path':'/original/attempt-01','request_sha256':file_hash(request),'e_bp_pct_pot':4,
                'reach_weight':.125}))
    return run,planned,report


def test_stored_average_alias_uses_hash_lineage_and_published_blob_path(tmp_path,monkeypatch):
    run,planned,report=corpus(tmp_path,monkeypatch)
    result=freeze(run,planned,{'status':'complete','base_decision':{'base':'average'}},report,'published',
                  'docs/reports/hu20-exact-turn-check.md')
    assert len(result['items'])==144 and result['exclusions']==[]
    assert all(item['policy']['strategy']=='average' and item['root']['reach_weight']==.125 for item in result['items'])
    assert result['items'][0]['provenance']['original_policy']['strategy']=='stored-average'


def test_alias_does_not_accept_a_different_artifact(tmp_path,monkeypatch):
    run,planned,report=corpus(tmp_path,monkeypatch)
    path=run/'spots'/'1-0'/'job.json';job=json.loads(path.read_text());job['policy']['sha256']='wrong'
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError,match='artifact differs'):
        freeze(run,planned,{'status':'complete','base_decision':{'base':'average'}},report,'published',
               'docs/reports/hu20-exact-turn-check.md')
