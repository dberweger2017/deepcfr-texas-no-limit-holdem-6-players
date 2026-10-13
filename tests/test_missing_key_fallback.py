"""Missing-only point masses, retained rows, sampler streams and trace identity."""
from dataclasses import asdict
import gzip
import hashlib
import json

import pytest

from scripts import evaluate_hu100_direct as direct
from scripts.pot_pressure_diagnostic import diagnostic_action
from src.blueprint.average import AveragePolicy
from src.blueprint.abstraction import HU200_SCHEMA
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from tests.test_hu200_evaluation import fixture_spec
from tests.test_v2_consumers import dev_source


def model(tmp_path, fallback='uniform'):
    spec = fixture_spec(tmp_path / 'model')
    return AveragePolicy(spec['path'], spec['sha256'], expected_schema=HU200_SCHEMA,
                         missing_key_fallback=fallback)


def views():
    hand = Hand.start(Table(('a', 'b'), (20000, 20000)), hand_id='fallback', seed=19)
    facing = hand.observe(hand.actor)
    hand = hand.apply(Action(ActionKind.CALL))
    return facing, hand.observe(hand.actor)


def test_missing_only_matches_pr251_and_consumes_one_draw(tmp_path):
    m = model(tmp_path, 'check-call'); m.entries = {}; m.zero_mass = set()
    for view in views():
        menu, probabilities, known, info = m.distribution_with_telemetry(view)
        assert not known and info['reason'] == 'missing-key' and info['fallback_applied']
        expected = diagnostic_action(view, 'call')
        assert probabilities == tuple(float(c.action == expected) for c in menu)
        for seed in range(30):
            a = m.policy(seed)
            assert a.choose_action(view) == expected
            state = a.random.getstate()
            m.configure_missing_key_fallback('uniform')
            b = m.policy(seed); b.choose_action(view)
            assert b.random.getstate() == state
            m.configure_missing_key_fallback('check-call')
        if ActionKind.CHECK in view.legal_actions.kinds:
            assert expected.kind == ActionKind.CHECK
        else: assert expected.kind == ActionKind.CALL


@pytest.mark.parametrize('zero_mass', [False, True])
def test_stored_rows_unchanged_even_without_average_mass(tmp_path, zero_mass):
    m = model(tmp_path)
    for view in views():
        menu, _, _ = m.distribution(view); key = m.information_key(view, menu)
        p = tuple((i + 1) / sum(range(1, len(menu) + 1)) for i in range(len(menu)))
        m.entries = {key: (tuple(c.name for c in menu), p)}
        m.zero_mass = {key} if zero_mass else set()
        expected = [m.policy(seed).choose_action(view) for seed in range(20)]
        m.configure_missing_key_fallback('check-call')
        assert m.distribution(view) == (menu, p, True)
        assert [m.policy(seed).choose_action(view) for seed in range(20)] == expected
        assert not m.distribution_with_telemetry(view)[3]['fallback_applied']
        m.configure_missing_key_fallback('uniform')


def test_option_validation_and_metadata(tmp_path):
    m = model(tmp_path)
    assert 'missing_key_fallback' not in m.description
    assert 'missing_key_fallback' not in m.distribution_with_telemetry(views()[0])[3]
    m.configure_missing_key_fallback('check-call')
    assert m.description['missing_key_fallback'] == 'check-call'
    with pytest.raises(ValueError, match='fallback'): m.configure_missing_key_fallback('fold')
    m.configure_missing_key_fallback('uniform')
    assert 'missing_key_fallback' not in m.description
    with pytest.raises(ValueError, match='Read-only'):
        AveragePolicy(maybe_path := tmp_path/'model/average.gz', direct.file_hash(maybe_path),
                      expected_schema=HU200_SCHEMA, cache=tmp_path/'absent', cache_read_only=True)


def test_same_model_shares_storage_but_has_private_fallback_and_cold_reproduction(tmp_path, monkeypatch):
    source = dev_source(monkeypatch); spec = fixture_spec(tmp_path / 'model')
    models = [{**spec, 'name': 'cc', 'missing_key_fallback': 'check-call'},
              {**spec, 'name': 'plain', 'missing_key_fallback': 'uniform'}]
    registry, _ = direct.execute(models, 16, 202610130801, 'fallback', tmp_path/'play', source,
                                 stack_bb=200, cache=tmp_path/'cache')
    a, b = registry.models.values()
    assert a.entries is b.entries and a.visits is b.visits and a.zero_mass is b.zero_mass
    assert a is not b and a.description is not b.description
    assert a.missing_key_fallback == 'check-call' and b.missing_key_fallback == 'uniform'
    checked = direct.audit(tmp_path/'play', tmp_path/'audit.json', cost_only=True)
    assert sum(c.get('fallback_decisions', 0) for c in checked['coverage'].values()) > 0
    _, repeat = direct.execute(models, 16, 202610130801, 'fallback', tmp_path/'repeat', source,
                               reproduce=tmp_path/'play', stack_bb=200, cache=tmp_path/'cache', cache_read_only=True)
    assert repeat['reproduced_all_hands_and_decisions']
    # Tampering with the identity cannot reinterpret retained point masses as uniform.
    inputs = tmp_path/'play/inputs.json'; data = json.loads(inputs.read_text())
    data['models'][0]['missing_key_fallback'] = 'uniform'; inputs.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='fallback'):
        direct.audit(tmp_path/'play', tmp_path/'bad.json')


def test_guard_thresholds_and_sustained_memory():
    from scripts.run_missing_key_evaluation import breached
    s = {'available_bytes': 1_999_999_999, 'swap_used_bytes': 100, 'disk_free_bytes': 20_000_000_000}
    reason, low = breached(s, 100, None, 0); assert reason is None and low == 0
    assert breached(s, 100, low, 59.99)[0] is None
    assert breached(s, 100, low, 60)[0] == 'memory-below-2GB-for-60s'
    assert breached({**s, 'available_bytes': 2_000_000_000}, 100, low, 60) == (None, None)
    assert breached({**s, 'swap_used_bytes': 1_000_000_101}, 100, None, 0)[0] == 'swap-growth-above-1GB'
    assert breached({**s, 'disk_free_bytes': 19_999_999_999}, 100, None, 0)[0] == 'disk-below-20GB'


def test_slumbot_check_call_telemetry_is_hash_chained_and_checked(tmp_path):
    from src.arena.external.journal import PrefixJournal, verify_journal
    from src.arena.external.slumbot import SlumbotAdapter, replay_record, verify_record
    from tests.test_external_slumbot_session import LIVE
    m = model(tmp_path, 'check-call'); m.entries = {}; m.zero_mass = set()
    player = m.policy(47)
    response = LIVE[0][0]; view = SlumbotAdapter().observation(response, 'h')
    row = replay_record(response, 'h', player.choose_action(view))
    row['policy_telemetry'] = {k: player.last_decision[3][k] for k in ('reason', 'missing_key_fallback', 'fallback_applied')}
    row['policy_telemetry']['street'] = view.street.value
    journal = PrefixJournal(tmp_path/'journal.jsonl', {'missing_key_fallback': 'check-call'})
    journal.append(row); journal.close()
    assert verify_journal(tmp_path/'journal.jsonl')['decisions'] == 1
    assert player.fallback_decisions_by_street == {'preflop': 1}
    bad = json.loads(json.dumps(row)); bad['policy_telemetry']['fallback_applied'] = False
    with pytest.raises(ValueError, match='telemetry'): verify_record(bad)


def test_reanalysis_separates_selected_hands_and_ignores_unfinished_decisions(tmp_path):
    from types import SimpleNamespace
    from scripts.reanalyze_slumbot_missing import describe
    from src.arena.external.journal import PrefixJournal
    from src.arena.external.session import play_hand
    from src.arena.external.slumbot import SlumbotConnection, replay_record
    from tests.test_external_slumbot_session import LIVE, Replay, HU200
    replay = Replay(LIVE[:2]); journal = PrefixJournal(tmp_path/'journal.jsonl', {'model_sha256': 'a'*64})
    for i in range(2): play_hand(SlumbotConnection(replay, HU200), replay, f'h{i}', journal)
    journal.append(replay_record(LIVE[0][0], 'unfinished')); journal.close()
    m = SimpleNamespace(description={'weights_sha256': 'a'*64}, missing_key_fallback='uniform', zero_mass={'h1'},
        distribution=lambda view: ((), (), view.hand_id != 'h0'), information_key=lambda view, menu: view.hand_id)
    result, values = describe(tmp_path/'journal.jsonl', m)
    assert result['groups']['missing']['hands'] == result['groups']['no_missing']['hands'] == 1
    assert values['missing'] == [LIVE[0][-1]['winnings']]
    assert values['no_missing'] == [LIVE[1][-1]['winnings']]
    assert result['decisions_without_terminal'] == 1
    assert sum(c['zero_mass'] for c in result['coverage_completed_hands'].values()) > 0
    assert sum(result['first_missing_street'].values()) == 1


def test_hu200_default_main_trace_sha256(tmp_path, monkeypatch):
    source = dev_source(monkeypatch)
    models = [fixture_spec(tmp_path/'left', name='left'), fixture_spec(tmp_path/'right', name='right')]
    direct.execute(models, 32, 202610130701, 'identity', tmp_path/'play', source, stack_bb=200)
    expected = {'hands.jsonl.gz': '3362aaf7038fd138b6315998c790f5cbd61cd52ba0892deac33eec092becd70e',
                'decisions.jsonl.gz': '8a74f8bb5fa2bf4ce61665b8d47d41ab5370883a0bb1a44f084dbefd869dd584'}
    assert {name: hashlib.sha256(gzip.decompress((tmp_path/'play'/name).read_bytes())).hexdigest()
            for name in expected} == expected
