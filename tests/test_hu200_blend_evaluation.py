"""The blend campaign keeps costs blind and all four panel arms coupled."""
from dataclasses import asdict
import json

import pytest

from scripts import evaluate_hu200_blend as campaign
from src.blueprint.abstraction import HU200_SCHEMA
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy
from tests.test_hu200_evaluation import fixture_spec
from tests.test_v2_consumers import dev_source


def test_fixed_roots_and_counts_separate_pilots_and_pair_all_models():
    final, pilots = campaign.comparisons(), campaign.comparisons(True)
    assert len(final) == len(pilots) == 24
    assert {c['root'] for c in final}.isdisjoint(c['root'] for c in pilots)
    for opponent in campaign.direct.PANEL_OPPONENTS:
        rows = [c for c in final if c['opponent'] == opponent]
        assert len(rows) == 3 and len({c['root'] for c in rows}) == 1
        assert {c['blocks'] for c in rows} == {16384}
    direct = [c for c in final if c['opponent'] is None]
    assert len({c['root'] for c in direct}) == 3
    assert {c['blocks'] for c in direct} == {131072}


def test_pilot_pairs_replay_repeat_and_share_blend_arms(tmp_path, monkeypatch):
    dev_source(monkeypatch)
    specs = [dict(fixture_spec(tmp_path/name, name=name), translation=asdict(TranslationOptions()))
             for name in ('blend', 'seed-1', 'seed-2', 'seed-3')]
    cache = tmp_path/'cache'
    for spec in specs:
        AveragePolicy(spec['path'], spec['sha256'], expected_schema=HU200_SCHEMA,
                      cache=cache, sorted_rows=True)
    fingerprints = []
    for i, seed in enumerate(specs[1:]):
        out = tmp_path/('pair-'+str(i))
        campaign.pair(out, [specs[0], seed], 3, 987123, 'half_pot_pressure', cache, pilot=True)
        replay = json.loads((out/'replay.json').read_text())
        assert replay['hands_replayed'] == 12 and replay['status'] == 'verified'
        assert 'bb_per_100' not in replay
        assert json.loads((out/'reproduction/complete.json').read_text())['reproduced_all_hands_and_decisions']
        fingerprints.append(campaign.candidate_fingerprints(out))
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_no_readout_without_global_verification_or_with_changed_evidence(tmp_path):
    plan = dict(sha256='abc')
    with pytest.raises(ValueError, match='All final hands'):
        campaign.readout(tmp_path, plan)
    (tmp_path/'all-final-verified.json').write_text(json.dumps(dict(status='incomplete', plan_sha256='abc')))
    with pytest.raises(ValueError, match='All final hands'):
        campaign.readout(tmp_path, plan)
    (tmp_path/'changed').write_bytes(b'changed')
    (tmp_path/'all-final-verified.json').write_text(json.dumps(dict(status='verified', plan_sha256='abc',
        files={'changed': {'bytes': 7, 'sha256': '0'*64}})))
    with pytest.raises(ValueError, match='evidence changed'):
        campaign.readout(tmp_path, plan)
    assert not (tmp_path/'summary.json').exists()
