"""Playable bots are actual pinned TP20 artifacts; all three positions replay."""

from scripts.play_tp20 import play, replay_history
from src.blueprint.artifact import export_policy
from src.blueprint.solver import BlueprintTrainer
from tests.test_blueprint_tp20 import config, table


def test_complete_three_position_human_play_replay(tmp_path):
    paths, hashes = [], []
    for seed in (71,72):
        trainer = BlueprintTrainer(table(),config(seed=seed))
        trainer.step()
        path = tmp_path/f"{seed}.gz"
        hashes.append(export_policy(trainer,path)); paths.append(path)
    outputs = []
    path = tmp_path/"human.jsonl"
    result = play(paths,hashes,session_seed=71,history=path,max_hands=6,
                  input_fn=lambda prompt:"1",output=outputs.append)
    assert result["hands"] == replay_history(path) == 6
    assert sum(result["net_chips"]) == 0
    assert any("Pinned bot models" in line for line in outputs)
