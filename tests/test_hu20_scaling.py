"""Continued work, atomic recovery and correlated block-shard accounting."""

from dataclasses import asdict
import json
from pathlib import Path
from time import time

import pytest

from scripts.evaluate_hu20_scaling import shard_owner, tasks
from scripts.hu20_reopening_common import write_cases
from scripts.hu20_scaling_common import parent_trainer
from scripts.report_hu20_scaling import contrast, merge_panels
from scripts.train_hu20_scaling import run
from src.blueprint.artifact import load_training, save_training
from src.blueprint.solver import BlueprintTrainer, HU20_UNCAPPED_GAME, PilotConfig
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.game.hand import Table


def test_continuation_keeps_the_complete_parent_and_cumulative_iteration_weights(tmp_path):
    trainer = BlueprintTrainer(Table(("player-0","player-1"),(2000,2000)),
        PilotConfig(seed=91,raise_cap=None,max_entries=10000,max_nodes=250000,
                    abstraction=HU20_UNCAPPED_SCHEMA,game=HU20_UNCAPPED_GAME))
    initial_nodes = sum(trainer.step().nodes for _ in range(3))
    checkpoint = tmp_path/"parent.json.gz"; h = save_training(trainer,checkpoint)
    parent = {"seed":91,"entries":len(trainer.nodes),"iteration":trainer.iteration,
              "completed_nodes":initial_nodes,"checkpoint_path":str(checkpoint),"checkpoint_sha256":h}
    fixture=tmp_path/"independent.jsonl.gz"
    write_cases({"independent_blocks":1,"independent_root":102},fixture)
    plan={"limits":{"max_entries":10000,"max_rss_gib":10.5,"min_free_gib":8},
          "training_total_nodes":initial_nodes+3000,"milestones":[initial_nodes+3000],
          "recovery_nodes":1000,"recovery_seconds":900,"independent_path":str(fixture)}
    result=run(plan,parent,tmp_path/"continued",time()+60)
    assert result["status"]=="complete"
    assert result["initial_iteration"]==3 and result["initial_nodes"]==initial_nodes
    assert result["additional_nodes"]==result["completed_nodes"]-initial_nodes
    assert result["additional_iterations"]==result["completed_iterations"]-3
    while trainer.iteration<result["completed_iterations"]:trainer.step()
    expected=save_training(trainer,tmp_path/"expected.json.gz")
    assert expected==result["milestones"][-1]["checkpoint_sha256"]
    assert load_training(checkpoint).iteration==3
    assert result["milestones"][0]["parent_checkpoint_sha256"]==h
    assert (tmp_path/"continued/last-recovery.json").exists()


def test_changed_parent_hash_is_rejected_before_continuation(tmp_path):
    path=tmp_path/"bad";path.write_text("not a checkpoint")
    with pytest.raises(ValueError,match="parent hash"):
        parent_trainer({"checkpoint_path":str(path),"checkpoint_sha256":"0"*64})


def test_each_independent_block_keeps_all_targets_on_the_same_host():
    plan={"block_host_cycle":["m4","m4","m4","m1","m1"]}
    assignments={host:[b for b in range(4096) if shard_owner(b,plan)==host] for host in ("m1","m4")}
    assert not set(assignments["m1"]) & set(assignments["m4"])
    assert sorted(assignments["m1"]+assignments["m4"])==list(range(4096))


def test_primary_averages_lineage_contrasts_inside_blocks():
    panels={}
    for seed in (1,2,3):
        panels[(f"B-{seed}-20000000","LBR-original-cap2")]={"blocks":{"0":seed*10,"1":seed*20,"2":seed*30}}
        panels[(f"B-{seed}-100000000","LBR-original-cap2")]={"blocks":{"0":seed*10+10,"1":seed*20+20,"2":seed*30+30}}
    r=contrast(panels,[1,2,3],"LBR-original-cap2",20000000,100000000)
    assert r["long_minus_20M"]["blocks"]==3
    assert r["long_minus_20M"]["bb100"]==20
    assert r["long_minus_20M"]["bb_hand"]==.2
    del panels[("B-3-100000000","LBR-original-cap2")]["blocks"]["0"]
    assert contrast(panels,[1,2,3],"LBR-original-cap2",20000000,100000000)["status"]=="unavailable"


def test_duplicate_host_blocks_are_a_failure():
    p={"policy":"test","attacker":"test","blocks":{"0":1},"roles":{}}
    with pytest.raises(ValueError,match="Duplicated"):
        merge_panels([{"panels":[p]},{"panels":[p]}])


def test_reference_contracts_do_not_follow_the_target_menu():
    plan={"training_total_nodes":100000000,"cheap_blocks":4,"lbr_blocks":2,"secondary_blocks":2,
          "stress_root":10,"lbr_root":20,"secondary_root":30}
    specs=[{"name":"B-final","arm":"B","milestone":100000000},
           {"name":"A-reference","arm":"A","milestone":20000000}]
    rows=list(tasks(plan,specs))
    assert next(r for r in rows if r[1]=="LBR-original-cap2")[3]=="menu"
    assert {r[1] for r in rows if r[0]["arm"]=="A"}=={"Pressure-native","Minraise-original-cap2","Passive"}


def test_production_two_host_shards_native_replay_and_aggregate(tmp_path):
    from scripts.evaluate_hu20_scaling import run as evaluate
    from scripts.hu20_scaling_common import specification
    from scripts.report_hu20_scaling import audit, combine
    from src.blueprint.artifact import export_policy
    specs=[]
    for seed in (101,102,103):
        trainer=BlueprintTrainer(Table(("player-0","player-1"),(2000,2000)),
            PilotConfig(seed=seed,raise_cap=None,max_entries=10000,max_nodes=250000,
                        abstraction=HU20_UNCAPPED_SCHEMA,game=HU20_UNCAPPED_GAME))
        for label in (20000000,100000000):
            trainer.step()
            cp=tmp_path/f"cp-{seed}-{label}.gz";policy=tmp_path/f"policy-{seed}-{label}.gz"
            specs.append(specification(seed,label,trainer.iteration,cp,policy,
                         save_training(trainer,cp),export_policy(trainer,policy)))
    models=tmp_path/"models.json";models.write_text(json.dumps(specs))
    plan={"training_total_nodes":100000000,"cheap_blocks":2,"lbr_blocks":2,"secondary_blocks":2,
          "stress_root":301,"lbr_root":302,"secondary_root":303,"training_seeds":[101,102,103],
          "block_host_cycle":["m1","m4"],"milestones":[100000000],"coordinator_models":str(models),
          "limits":{"max_rss_gib":10.5,"min_free_gib":8}}
    paths=[]
    for host in ("m1","m4"):
        r=evaluate(plan,specs,host,tmp_path/f"eval-{host}",time()+120)
        assert r["status"]=="complete" and r["hands"]==144
        a=audit(plan,tmp_path/f"eval-{host}",tmp_path/f"audit-{host}")
        assert a["status"]=="complete" and a["native_replayed_hands"]==144
        paths.append(tmp_path/f"audit-{host}/results.json")
    r=combine(plan,paths,tmp_path/"report.json")
    assert r["status"]=="complete" and r["native_replayed_hands"]==288
    assert r["primary"]["LBR-original-cap2"]["long_minus_20M"]["blocks"]==2
