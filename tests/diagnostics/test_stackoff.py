"""Frozen opponent and saved-policy diagnostics, using generated tiny exports."""
import copy
import gzip
import json
from dataclasses import replace
from random import Random

import pytest

from scripts.evaluate_hu20_stackoff import Contexts, recorded_hand
from scripts.inspect_hu20_stackoff import holdings
from src.arena.schedule import digest
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, choices, information_key
from src.blueprint.artifact import HU20_UNCAPPED_FORMAT, export_policy, save_training
from src.blueprint.search import DECK, _sample_world
from src.blueprint.solver import BlueprintTrainer, HU20_UNCAPPED_GAME, Node, PilotConfig
from src.diagnostics.saved_hu20 import file_hash, load_saved
from src.diagnostics.selective_stackoff import SelectiveStackoff, is_jam, raise_call_amount, strength
from src.diagnostics.stackoff_report import summarize
from src.diagnostics.stackoff_tails import hand_tails, public_context, snapshot
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


def hand_view(cards=('Ac', 'Ad'), board=('2c', '5d', '8h', 'Ts', 'Jc'), river=False):
    # Seat 1 gets first card in each round. Own seat is 0 before the flop.
    prefix = ('Kc', cards[0], 'Kd', cards[1]) + board
    hand = Hand.from_deck(Table(('hero', 'rival'), (2000, 2000)), hand_id='fixture',
                          deck=prefix + tuple(c for c in DECK if c not in prefix))
    if river:
        while len(hand.observe(hand.actor).board) < 5:
            view = hand.observe(hand.actor)
            hand = hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL))
    return hand, hand.observe(hand.actor)


@pytest.fixture
def saved(tmp_path):
    config = PilotConfig(seed=7, abstraction=HU20_UNCAPPED_SCHEMA, game=HU20_UNCAPPED_GAME,
                         raise_cap=None, max_entries=1000, max_nodes=1000, max_seconds=30)
    trainer = BlueprintTrainer(Table(('hero', 'rival'), (2000, 2000)), config)
    trainer.iteration = 7
    _, view = hand_view()
    menu = choices(view, raise_cap=None, free_fold=False)
    key = information_key(view, menu, schema=HU20_UNCAPPED_SCHEMA)
    trainer.nodes[key] = Node(tuple(c.name for c in menu), [float(i+1) for i in range(len(menu))],
                              [0.] * len(menu), 333)
    spec = {'name': 'tiny', 'path': 'policy.gz', 'checkpoint_path': 'checkpoint.gz',
            'format': HU20_UNCAPPED_FORMAT, 'players': 2, 'seed': 7, 'milestone': 100,
            'abstraction': HU20_UNCAPPED_SCHEMA, 'dual_menu_telemetry': True}
    spec['sha256'] = export_policy(trainer, tmp_path / spec['path'])
    spec['checkpoint_sha256'] = save_training(trainer, tmp_path / spec['checkpoint_path'])
    return spec, tmp_path


@pytest.mark.parametrize('cards,tier', [(('Ac','Ad'),'strong'),(('As','Kh'),'strong'),
    (('Tc','Td'),'strong'),(('9c','9d'),'medium'),(('Ac','3d'),'medium'),(('Tc','Jh'),'medium'),
    (('7c','3d'),'weak')])
def test_preflop_frozen_tiers(cards, tier):
    _, view = hand_view(cards)
    assert strength(view) == tier


def test_private_worlds_cannot_change_opponent_choice_or_public_context():
    _, view = hand_view()
    cards = [c for c in DECK if c not in view.hole_cards]
    a = _sample_world(view, {1-view.seat: ((tuple(cards[:2]), 1),)}, Random(12))
    b = _sample_world(view, {1-view.seat: ((tuple(cards[-2:]), 1),)}, Random(33))
    assert a.observe(view.seat) == b.observe(view.seat)
    assert SelectiveStackoff(8).choose_action(a.observe(view.seat)) == SelectiveStackoff(8).choose_action(b.observe(view.seat))
    assert public_context(view) == public_context(replace(view, hand_id='different', hole_cards=('Qs','Qh')))


def test_one_rng_draw_and_restart_next_action():
    _, view = hand_view()
    player = SelectiveStackoff(21)
    oracle = Random(21)
    player.choose_action(view); oracle.random()
    assert player.random.getstate() == oracle.getstate()
    restored = SelectiveStackoff(0)
    restored.random.setstate(player.random.getstate())
    assert restored.choose_action(view) == player.choose_action(view)
    assert restored.random.getstate() == player.random.getstate()


def test_large_call_is_amount_owed_not_raise_to_and_allin_call_is_not_jam():
    hand, view = hand_view()
    action = Action(ActionKind.RAISE, 900)
    assert raise_call_amount(view, action) == 800
    assert not is_jam(view, action)
    assert is_jam(view, Action(ActionKind.RAISE, 2000))
    assert not is_jam(view, Action(ActionKind.CALL))
    hand = hand.apply(action)
    rival = hand.observe(hand.actor)
    assert rival.legal_actions.call_amount == 800
    weak = replace(rival, hole_cards=('7c','3d'))
    assert SelectiveStackoff(1).choose_action(weak).kind == ActionKind.FOLD
    assert SelectiveStackoff(1).choose_action(replace(rival,hole_cards=('As','Ah'))).kind == ActionKind.CALL


def test_board_only_is_not_strong():
    _, view = hand_view(('3h','4d'), ('Ac','Ks','Qh','Js','Tc'), river=True)
    assert strength(view) == 'board_only'


def test_readonly_current_audit_and_rejection(saved):
    spec, path = saved
    source, visits = load_saved(spec, path)
    assert list(visits.values()) == [333]
    assert source.description['iteration'] == 7
    assert spec['milestone'] == 100  # Work milestone is not traversal iteration.
    for field, value in [('sha256','0'*64), ('seed',8), ('checkpoint_sha256','0'*64)]:
        with pytest.raises(ValueError):
            load_saved({**spec,field:value}, path)
    document = json.loads(gzip.open(path / spec['path'], 'rt').read())
    document['strategy'] = 'average'
    (path / spec['path']).write_bytes(gzip.compress(json.dumps(document).encode()))
    with pytest.raises(ValueError, match='lineage'):
        load_saved({**spec,'sha256':file_hash(path / spec['path'])}, path)


def test_generated_native_hands_are_legal_replayable_and_deterministic(saved):
    spec, path = saved
    source, visits = load_saved(spec, path)
    plan = {'chance_samples':4,'lbr_seconds':5}
    panel = {'name':'Selective-stackoff-v1','rule':'selective_stackoff','contract':'restricted','root':871}
    for block in range(16):
        for rotation in (0,1):
            first = recorded_hand(source, visits, spec, panel, block, rotation, plan)
            second = recorded_hand(source, visits, spec, panel, block, rotation, plan)
            assert first['status'] == 'complete' and first['native_replay_verified']
            assert sum(first['net_chips_by_seat']) == 0
            assert first['event_digest'] == second['event_digest']
            assert first['tails'] == second['tails']
            for action in first['actions']:
                menu = action['observation']['menu']
                assert any(c['kind']==action['kind'] and c['raise_to']==action['raise_to'] for c in menu)


def test_tail_arithmetic_partitions_first_raise_and_full_stack_signs():
    hand, view = hand_view()
    hand = hand.apply(Action(ActionKind.RAISE, 600))
    view = hand.observe(hand.actor)
    menu = choices(view, raise_cap=None, free_fold=False)
    own = snapshot(view, menu, [1/len(menu)]*len(menu),True,400)
    amount = next(c for c in own['menu'] if c['rival_call_amount'] >= 800)
    raise_action = {'logical_player':0,'street':'preflop','kind':'raise','raise_to':amount['raise_to'],'observation':own}
    rival = {'logical_player':1,'street':'preflop','kind':'call','raise_to':None,
             'observation':{'call_amount':amount['rival_call_amount']}}
    row = {'status':'complete','target_chips':-2000,'actions':[raise_action,rival,raise_action,{**rival,'kind':'fold'}]}
    result = hand_tails(row)
    assert result['counts']['large_actions'] == 2
    assert result['counts']['rival_continuations'] == result['counts']['rival_folds'] == 1
    assert result['first_large_raise_response'] == 'continued'
    assert result['counts']['full_stack_losses'] == 1
    assert result['counts']['full_stack_wins'] == 0
    assert hand_tails({**row,'target_chips':1900})['counts']['full_stack_wins'] == 0
    assert hand_tails({**row,'actions':[raise_action]})['counts']['no_response'] == 1


def test_compatible_inspection_holdings_do_not_condition_on_hidden_cards():
    _, view = hand_view(river=True)
    pairs = list(holdings(view))
    assert len(pairs) == 47*46//2
    assert any(view.hole_cards[0] in pair for pair in pairs)


def test_shared_lineage_blocks_and_positions_not_pseudoreplicates():
    models = [{'name':f'{seed}-{m}', 'seed':seed,'milestone':m} for seed in (1,2,3) for m in (20,40)]
    plan = {'models':models,'panels':[{'name':'stress','blocks':32}], 'interval':'paired'}
    rows = []
    for model in models:
        for b in range(32):
            for r in (0,1):
                profit = b + model['milestone'] + (100 if r == b%2 else -100)
                net = [0,0];net[r]=profit;net[1-r]=-profit
                rows.append({'policy':model['name'],'panel':'stress','block':b,'rotation':r,'button':b%2,
                    'status':'complete','native_replay_verified':True,'net_chips_by_seat':net,'target_chips':profit,
                    'actions':[], 'tails':{'counts':{'hands':1},'first_large_raise_response':'no_large_raise'}})
    result = summarize(plan, reversed(rows))
    assert result['status'] == 'complete'
    assert result['three_lineage_aggregate'][0]['overall']['blocks'] == 32
    assert result['three_lineage_aggregate'][0]['positions']['button']['bb_per_100'] == 135.5
    assert result['checkpoint_changes'][0]['paired_difference']['bb_per_100'] == 20
    with pytest.raises(ValueError,match='duplicate'):
        summarize(plan, rows + [rows[0]])
    assert summarize(plan, rows[:-1])['status'] == 'incomplete'


@pytest.mark.parametrize('cards,board,tier', [
    (('Ac','Ad'),('2c','5d','8h','Ts','Jc'),'medium'),
    (('2h','5h'),('2c','5d','8h','Ts','Jc'),'strong'),
    (('3h','4d'),('2c','5d','8h','Ts','Jc'),'weak'),
])
def test_postflop_rule_uses_made_hand_not_future_equity(cards, board, tier):
    _, view = hand_view(cards, board, river=True)
    # The first river actor is seat 1; replace only the observer's own cards.
    assert strength(replace(view,hole_cards=cards)) == tier


def test_context_replay_queries_actual_menu_and_includes_hidden_opponent_cards(saved):
    from scripts.inspect_hu20_stackoff import context_view, queries
    spec, path = saved
    source, visits = load_saved(spec,path)
    hand = Hand.start(Table(('player-0','player-1'),(2000,2000),button=0),
                      hand_id='robustness-stackoff-v1-2-0',seed=31)
    prefix = []
    while len(hand.observe(hand.actor).board) < 5:
        view = hand.observe(hand.actor)
        action = Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL)
        prefix.append({'seat':hand.actor,'kind':action.kind.value,'raise_to':action.raise_to})
        hand = hand.apply(action)
    view = hand.observe(hand.actor)
    context = {'public_context':public_context(view),'id':digest(public_context(view)),
               'origin':{'rotation':0,'button':0,'phase':'stackoff-v1','block':0,'deal_seed':31},'prefix':prefix}
    context = json.loads(json.dumps(context))
    assert context_view(context) == view
    rows = list(queries(source,visits,context))
    assert len(rows) == 1081
    assert all(r['probabilities'] and len(r['menu']) == len(r['probabilities']) for r in rows)
    assert all(0 <= r['large_raise_probability'] <= 1 for r in rows)
    altered = copy.deepcopy(context); altered['public_context']['pot'] += 1
    with pytest.raises(ValueError,match='public replay'):
        context_view(altered)


def test_candidate_selection_ignores_terminal_payoff_and_keeps_lowest_hashes():
    plan = {'inspection_origin_milestone':100, 'minimum_visits':300,'small_call_chips':200,
            'contexts_per_seed_position_kind':1,'models':[{'seed':7}]}
    observed = {'trained':True,'visits':300,'call_amount':100,'pot':500,'street_bet':0,
                'position':'button','public_context_id':'b','public_context':{'board':['Ac']}}
    row = {'players':2,'rotation':0,'button':0,'phase':'fixture','block':0,'deal_seed':13,'target_chips':-2000,
           'actions':[{'logical_player':1,'seat':1,'street':'river','kind':'raise','raise_to':100,'observation':{}},
                      {'logical_player':0,'seat':0,'street':'river','kind':'call','raise_to':None,'observation':observed}]}
    spec = {'milestone':100,'seed':7}
    first, second = Contexts(plan), Contexts(plan)
    first.consider(row,spec);second.consider({**row,'target_chips':2000},spec)
    assert first.document() == second.document()
    changed = copy.deepcopy(row);changed['actions'][1]['observation']['public_context_id']='a'
    first.consider(changed,spec)
    assert first.document()['contexts'][0]['id'] == 'a'
    third=Contexts(plan);changed['actions'][1]['observation']['visits']=299;third.consider(changed,spec)
    assert third.document()['contexts'] == []


def test_checkpoint_wrong_regrets_rejected_even_with_new_byte_hash(saved):
    spec,path=saved
    lines=gzip.open(path/spec['checkpoint_path'],'rt').read().splitlines()
    node=json.loads(lines[1]);node[2][0]+=100
    lines[1]=json.dumps(node)
    (path/spec['checkpoint_path']).write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode()))
    with pytest.raises(ValueError,match='current extraction'):
        load_saved({**spec,'checkpoint_sha256':file_hash(path/spec['checkpoint_path'])},path)


def test_lbr_wrapper_keeps_actual_actions_separate_from_hypothetical_queries(saved):
    spec,path=saved
    source,visits=load_saved(spec,path)
    plan={'chance_samples':1,'lbr_seconds':.01}
    panel={'name':'fixture-lbr','rule':'lbr','contract':'menu','root':31}
    row=recorded_hand(source,visits,spec,panel,0,0,plan)
    assert row['status']=='complete'
    assert all('lbr' in a for a in row['actions'] if a['logical_player']==1)
    assert all(a['observation']['logical_player']==a['logical_player'] for a in row['actions'])


def test_checkpoint_traversal_iteration_must_match_export(saved):
    spec,path=saved
    lines=gzip.open(path/spec['checkpoint_path'],'rt').read().splitlines()
    header=json.loads(lines[0]);header['iteration']+=1;lines[0]=json.dumps(header)
    (path/spec['checkpoint_path']).write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode()))
    with pytest.raises(ValueError,match='lineage'):
        load_saved({**spec,'checkpoint_sha256':file_hash(path/spec['checkpoint_path'])},path)


def test_cli_vertical_slice_fixture_and_explicit_guard_abort(saved,tmp_path):
    from scripts.evaluate_hu20_stackoff import run
    from scripts.report_hu20_stackoff import records
    spec,inputs=saved
    plan={'models':[spec], 'panels':[{'name':'fixture','rule':'selective_stackoff',
          'contract':'restricted','root':134,'blocks':2}], 'chance_samples':4,'lbr_seconds':5,
          'inspection_origin_milestone':100,'minimum_visits':300,'small_call_chips':200,
          'contexts_per_seed_position_kind':2,'interval':'paired',
          'limits':{'max_seconds':30,'max_rss_gib':6,'max_swap_growth_gib':.5,'min_free_gib':0}}
    out=tmp_path/'run'
    result=run(plan,inputs,out)
    assert result['status']=='complete' and result['hands']==4
    assert summarize(plan,records(out))['status']=='complete'
    assert len(json.loads((out/'manifest.json').read_text())['outputs']) >= 4
    impossible={**plan,'limits':{**plan['limits'],'max_seconds':-1}}
    failed=run(impossible,inputs,tmp_path/'aborted')
    assert failed['status']=='incomplete' and failed['hands']==0
    with pytest.raises(FileExistsError):
        run(plan,inputs,out)


def test_diagnostic_audit_failure_is_retained_as_failed_hand(saved,monkeypatch):
    import scripts.evaluate_hu20_stackoff as evaluator
    spec,path=saved
    source,visits=load_saved(spec,path)
    def broken(row):
        raise ValueError('injected audit defect')
    monkeypatch.setattr(evaluator,'hand_tails',broken)
    row=evaluator.recorded_hand(source,visits,spec,
        {'name':'fixture','rule':'selective_stackoff','contract':'restricted','root':91},0,0,
        {'chance_samples':4,'lbr_seconds':5})
    assert row['status']=='failed'
    assert row['actions'] and 'injected audit defect' in row['error']


def test_compact_export_native_replays_and_links_raw_decisions(saved,tmp_path):
    import csv
    from scripts.evaluate_hu20_stackoff import run
    from scripts.export_hu20_stackoff import export
    from scripts.play_robustness import replay_row
    spec,inputs=saved
    plan={'models':[spec], 'panels':[{'name':'fixture','rule':'selective_stackoff',
          'contract':'restricted','root':134,'blocks':2}], 'chance_samples':4,'lbr_seconds':5,
          'inspection_origin_milestone':100,'minimum_visits':300,'small_call_chips':200,
          'contexts_per_seed_position_kind':2,'interval':'paired',
          'limits':{'max_seconds':30,'max_rss_gib':6,'max_swap_growth_gib':.5,'min_free_gib':0}}
    folder=tmp_path/'run';run(plan,inputs,folder)
    destination=tmp_path/'evidence';manifest=export(folder,destination)
    rows=[json.loads(line) for line in gzip.open(destination/'generated-hands.jsonl.gz','rt')]
    for row in rows:
        replay_row(row)
        assert all('observation' not in action for action in row['actions'])
    decisions=list(csv.DictReader(gzip.open(destination/'decisions.csv.gz','rt')))
    assert len(decisions)==sum(len(row['actions']) for row in rows)
    assert all('target_chips' not in d for d in decisions)
    assert manifest['files']['decisions.csv.gz']['sha256']==file_hash(destination/'decisions.csv.gz')


def test_uniform_comparison_reuses_existing_cap2_and_sampling():
    from scripts.evaluate_hu20_reopening import UniformPlayer
    from scripts.evaluate_hu20_stackoff import opponent
    from src.arena.schedule import stream_seed
    panel={'root':53,'rule':'hu20_uniform','contract':'secondary'}
    actual=opponent(panel,None,2,{})
    assert isinstance(actual,UniformPlayer)
    reference=UniformPlayer(stream_seed(53,'test','action',2,2,1))
    _,view=hand_view()
    assert actual.choose_action(view)==reference.choose_action(view)


def test_lbr_telemetry_counts_budget_batches_and_cumulative_events_once():
    plan={'models':[{'name':'tiny','seed':1,'milestone':20}],
          'panels':[{'name':'lbr','blocks':1}],'interval':'paired'}
    telemetry={'requested_samples':4,'samples':2,'completed':False,
               'over_soft_budget':True,'zero_likelihood_events':2}
    rows=[{'policy':'tiny','panel':'lbr','block':0,'rotation':r,'button':0,'status':'complete',
           'native_replay_verified':True,'net_chips_by_seat':[0,0],'target_chips':0,
           'tails':{'counts':{'hands':1},'first_large_raise_response':'no_large_raise'},
           'actions':[{'lbr':telemetry},{'lbr':{**telemetry,'zero_likelihood_events':3}}]} for r in (0,1)]
    counts=summarize(plan,rows)['per_seed'][0]['counts']
    assert counts['lbr_decisions']==4 and counts['lbr_over_soft_budget']==4
    assert counts['lbr_requested_batches']==16 and counts['lbr_completed_batches']==8
    assert counts['lbr_zero_likelihood_events']==6


def test_dashboard_labels_incomplete_and_exploratory_intervals():
    from scripts.report_hu20_stackoff import markdown
    result={'status':'incomplete','attempted_hands':0,'requested_hands':100,'failed_hands':0,
            'unattempted_hands':100,'three_lineage_aggregate':[], 'checkpoint_changes':[], 'per_seed':[]}
    rendered=markdown(result)
    assert '**incomplete**' in rendered
    assert 'not individual-bet EV' in rendered
    assert 'not independent samples' in rendered
    assert '800 additional chips' in rendered


def test_inspection_deduplicates_matched_contexts_without_erasing_selection_origins():
    from scripts.inspect_hu20_stackoff import unique_contexts
    public={'board':['Ac'],'pot':600}
    first={'id':digest(public),'public_context':public,'origin_seed':1}
    second={**first,'origin_seed':2}
    document={'contexts':[first,second]}
    assert unique_contexts(document)==[first]
    assert len(document['contexts'])==2
    with pytest.raises(ValueError,match='digest'):
        unique_contexts({'contexts':[{**first,'id':'wrong'}]})


def test_public_evidence_checker_recomputes_csv_tails_and_paired_summary(saved,tmp_path):
    from scripts.evaluate_hu20_stackoff import run
    from scripts.export_hu20_stackoff import export
    from scripts.report_hu20_stackoff import records
    from scripts.evaluate_hu20 import write_json
    from scripts.check_hu20_stackoff import check
    spec,inputs=saved
    plan={'models':[spec], 'panels':[{'name':'fixture','rule':'selective_stackoff',
          'contract':'restricted','root':134,'blocks':2}], 'chance_samples':4,'lbr_seconds':5,
          'inspection_origin_milestone':100,'minimum_visits':300,'small_call_chips':200,
          'contexts_per_seed_position_kind':2,'interval':'paired',
          'limits':{'max_seconds':30,'max_rss_gib':6,'max_swap_growth_gib':.5,'min_free_gib':0}}
    folder=tmp_path/'run';run(plan,inputs,folder)
    write_json(folder/'summary.json',summarize(plan,records(folder)))
    out=tmp_path/'evidence';export(folder,out)
    result=check(out)
    assert result['status']=='complete' and result['native_replays']==4
    assert result['raw_decisions']>=4  # A valid hand can end with its first fold.
    from scripts.report_hu20_stackoff_made_hands import report
    raw_hashes={name:file_hash(out/name) for name in ('decisions.csv.gz','generated-hands.jsonl.gz','summary.json')}
    report(out)
    assert check(out)['first_large_raise_hands']==0  # This fixture panel is outside the stress panel.
    assert raw_hashes=={name:file_hash(out/name) for name in raw_hashes}
    derived=out/'large-raise-made-hands.json'
    original=derived.read_text()
    altered=json.loads(original);altered['first_large_raise_hands']=1;write_json(derived,altered)
    manifest=json.loads((out/'evidence-manifest.json').read_text())
    manifest['files'][derived.name]={'bytes':derived.stat().st_size,'sha256':file_hash(derived)}
    write_json(out/'evidence-manifest.json',manifest)
    with pytest.raises(ValueError,match='made-hand comparisons'):
        check(out)
    derived.write_text(original)
    manifest['files'][derived.name]={'bytes':derived.stat().st_size,'sha256':file_hash(derived)}
    write_json(out/'evidence-manifest.json',manifest)
    (out/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='hash or size'):
        check(out)


def test_summary_check_allows_quantile_roundoff_but_not_count_or_effect_changes():
    from scripts.check_hu20_stackoff import same_summary
    original={'blocks':1024,'interval':[1.0,2.0]}
    assert same_summary(original,{'blocks':1024,'interval':[1.0+1e-12,2.0]})
    assert not same_summary(original,{'blocks':1025,'interval':[1.0,2.0]})
    assert not same_summary(original,{'blocks':1024,'interval':[1.0,2.01]})
    assert not same_summary(original,{'blocks':1024,'interval':[1.0]})


def made_hand_row(board=('Ac','Kd','7h'), target=('As','2c'), rival=('Ks','Kh'), response='call'):
    def action(index, actor, street, kind, amount=None, cards=None, shown=None):
        return {'index':index,'logical_player':actor,'street':street,'kind':kind,'raise_to':amount,
                'observation':{'hole_cards':list(cards or (target if actor==0 else rival)),
                    'board':list(board if shown is None else shown),
                    'menu':[{'kind':'raise','raise_to':1000,'rival_call_amount':800}]}}
    return {'status':'complete','policy':'tiny','block':0,'rotation':0,'button':0,'target_chips':-2000,
            'actions':[action(0,1,'preflop','raise',200,shown=()),
                       action(1,1,'flop' if board else 'preflop','raise',200),
                       action(2,0,'flop' if board else 'preflop','raise',1000),
                       action(3,1,'flop' if board else 'preflop',response)]}


def test_first_large_raise_uses_current_board_and_counts_each_hand_once():
    from src.diagnostics.stackoff_made_hands import first_large_raise, summarize_events
    row=made_hand_row()
    # Two future aces would reverse the made-hand order; it must not enter this comparison.
    future=copy.deepcopy(row['actions'][2]);future.update(index=4,street='river')
    future['observation']['board']=['Ac','Kd','7h','Ah','Ad']
    from src.game.showdown import hand_value
    board=tuple(future['observation']['board'])
    assert hand_value(('As','2c')+board)>hand_value(('Ks','Kh')+board)
    row['actions'].append(future)
    event=first_large_raise(row)
    assert event['index']==2 and event['comparison']=='behind'
    assert event['target_category']=='pair' and event['rival_category']=='trips'
    assert event['situation']=='after_rival_raise' and event['board']==['Ac','Kd','7h']
    result=summarize_events([event],{'models':[{'name':'tiny','milestone':100}]})
    counts=next(r['counts'] for r in result['groups'] if r['model']=='aggregate')
    assert counts['hands']==1 and counts['target_chips']==-2000
    assert counts['continued_behind']==1 and counts['category_pair']==1
    assert counts['street_flop_continued']==1


@pytest.mark.parametrize('target,rival,expected',[(('As','Ah'),('Ks','2c'),'ahead'),
    (('As','2c'),('Ks','Kh'),'behind'),(('As','2c'),('Ah','2d'),'tied')])
def test_made_hand_order_compares_complete_hand_values(target,rival,expected):
    from src.diagnostics.stackoff_made_hands import first_large_raise
    assert first_large_raise(made_hand_row(target=target,rival=rival))['comparison']==expected


def test_made_hand_preflop_fold_and_same_street_denominators():
    from src.diagnostics.stackoff_made_hands import first_large_raise,summarize_events
    folded=made_hand_row(response='fold')
    # A raise on a previous street is not a same-street rival raise.
    folded['actions'].pop(1)
    event=first_large_raise(folded)
    assert event['situation']=='no_rival_raise' and event['response']=='folded'
    preflop=first_large_raise(made_hand_row(board=()))
    assert preflop['comparison']=='preflop' and preflop['target_category']=='preflop'
    result=summarize_events([event,preflop],{'models':[{'name':'tiny','milestone':100}]})
    c=next(r['counts'] for r in result['groups'] if r['model']=='aggregate' and r['situation']=='after_rival_raise')
    assert c['continued_preflop']==1 and 'continued_ahead' not in c
    no_raise=copy.deepcopy(folded);no_raise['actions'][1]['kind']='call'
    assert first_large_raise(no_raise) is None
