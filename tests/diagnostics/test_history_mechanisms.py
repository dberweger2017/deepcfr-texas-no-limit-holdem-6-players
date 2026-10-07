"""Explicit schema isolation, unchanged average math, common restricted law."""
from dataclasses import replace
from pathlib import Path
import pytest
import numpy as np

from src.blueprint.abstraction import HU20_COMPRESSED_SCHEMA, HU20_UNCAPPED_SCHEMA, choices, information_key
from src.blueprint.artifact import save_training, export_policy
from src.blueprint.solver import BlueprintTrainer, PilotConfig, Node, HU20_UNCAPPED_GAME
from src.diagnostics.cfr_average import extract, audit
from src.blueprint.average import AveragePolicy
from src.diagnostics.history_river import CommonRiverGame, common_ranges, restricted_distribution, policy_profile, LAW
from src.blueprint.river_cfr import profile_quality
from src.blueprint.river_game import river_root_history
from scripts.evaluate_hu20_river import root_hand


def test_compressed_average_requires_explicit_schema_and_preserves_math(tmp_path):
    hand = root_hand(202610120301)
    cfg = PilotConfig(seed=7, raise_cap=None, game=HU20_UNCAPPED_GAME, abstraction=HU20_COMPRESSED_SCHEMA)
    trainer = BlueprintTrainer(hand.table, cfg)
    trainer.iteration = 10
    view = hand.observe(hand.actor)
    menu = choices(view, raise_cap=None, free_fold=False)
    key = information_key(view, menu, schema=cfg.abstraction)
    n = len(menu)
    trainer.nodes[key] = Node(tuple(c.name for c in menu), [1]+[0]*(n-1), [0]+[2]*(n-1), 3)
    trainer.nodes['f'*32] = Node(('check',), [0], [0], 0)
    cp, current, avg = (tmp_path/x for x in ('cp.gz', 'current.gz', 'average.gz'))
    spec = {'name':'C', 'seed':7, 'iteration':10, 'checkpoint_sha256':save_training(trainer,cp), 'sha256':export_policy(trainer,current)}
    before = cp.read_bytes()
    with pytest.raises(ValueError, match='identity'):
        extract(cp,spec,avg)
    result = extract(cp,spec,avg,expected_schema=cfg.abstraction)
    assert audit(cp,current,avg,spec,result['sha256'],expected_schema=cfg.abstraction)['all_nodes_verified']==2
    with pytest.raises(ValueError, match='identity'):
        AveragePolicy(avg,result['sha256'])
    source = AveragePolicy(avg,result['sha256'],expected_schema=cfg.abstraction)
    actual_menu,p,hit = source.distribution(view)
    assert hit and p[0]==0 and sum(p)==pytest.approx(1)
    assert source.visits[key]==3 and 'f'*32 in source.zero_mass
    assert source.distribution(replace(view,hand_id='same-public-history'))==(actual_menu,p,hit)
    assert cp.read_bytes()==before
    with pytest.raises(ValueError, match='Unknown'):
        AveragePolicy(avg,result['sha256'],expected_schema='unregistered')


class Uniform:
    def distribution(self,view):
        menu = choices(view,raise_cap=None,free_fold=False)
        return menu,(1/len(menu),)*len(menu),False


def test_common_ranges_do_not_query_a_candidate_and_match_all_compatible_holdings():
    root=river_root_history(root_hand(202610120301).events)
    ranges=common_ranges(root)
    assert len(ranges[0])==len(ranges[1])==1081
    assert ranges[0]==ranges[1] and all(mass==1 for _,mass in ranges[0])
    assert all(not set(pair)&set(root_hand(202610120301).observe(0).board) for pair,_ in ranges[0])


def test_restricted_projection_reports_removed_and_zero_mass():
    view=root_hand(202610120301).observe(0)
    menu=choices(view,raise_cap=None,free_fold=False)
    class LastOnly:
        def distribution(self,view):return menu,tuple(float(i==len(menu)-1) for i in range(len(menu))),True
    p,t=restricted_distribution(LastOnly(),view,menu[:-1])
    assert p==(1/len(p),)*len(p) and t['zero_retained_mass'] and t['removed_mass']==1
    assert restricted_distribution(Uniform(),view,menu)[1]['removed_mass']==pytest.approx(0)
    class Negative:
        def distribution(self,view):return menu,(-1,)+(2/(len(menu)-1),)*(len(menu)-1),True
    with pytest.raises(ValueError,match='probability'):
        restricted_distribution(Negative(),view,menu)
    with pytest.raises(ValueError,match='menu'):
        restricted_distribution(Uniform(),view,())


def test_small_common_game_projection_uses_exact_existing_quality_machinery():
    root=river_root_history(root_hand(202610120301).events)
    # A small stipulated law keeps the regression independent of full-range cost.
    rows=common_ranges(root)[0]
    selected=[rows[i] for i in (0,100,300,700)]
    game=CommonRiverGame(root,{0:tuple(selected),1:tuple(selected)})
    profile,counts=policy_profile(game,Uniform())
    assert game.law_label==LAW and game.joint.sum()==pytest.approx(1)
    assert counts['queries']>0 and counts['queries']==counts['missing']
    for node in game.nodes:
        if node.actor is not None:
            assert np.allclose(profile[node.id],1/len(node.menu))
    quality=profile_quality(game,profile)
    assert quality['zero_sum_error_bb']<1e-10
    assert all(x>=0 for x in quality['best_response_gains_bb'])
