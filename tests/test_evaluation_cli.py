import csv
import json
from pathlib import Path

import torch

import scripts.evaluate_models as eval_mod
from src.core.deep_cfr import DeepCFRAgent


def write_checkpoint(path: Path, iteration: int) -> None:
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    agent.iteration_count = iteration
    torch.save(
        {
            "iteration": agent.iteration_count,
            "advantage_net": agent.advantage_net.state_dict(),
            "strategy_net": agent.strategy_net.state_dict(),
            "min_bet_size": agent.min_bet_size,
            "max_bet_size": agent.max_bet_size,
        },
        path,
    )


def test_evaluate_checkpoint_paths_and_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    checkpoint_a = tmp_path / "checkpoint_a.pt"
    checkpoint_b = tmp_path / "checkpoint_b.pt"
    write_checkpoint(checkpoint_a, iteration=11)
    write_checkpoint(checkpoint_b, iteration=22)

    results = eval_mod.evaluate_checkpoint_paths(
        [checkpoint_a, checkpoint_b],
        games_random=1,
        games_pool=1,
        seed=0,
        device="cpu",
        strict=False,
        stake=200.0,
        sb=1.0,
        bb=2.0,
    )

    assert len(results) == 2
    assert {result["name"] for result in results} == {"checkpoint_a.pt", "checkpoint_b.pt"}
    assert {result["iteration"] for result in results} == {11, 22}

    for result in results:
        assert "vs_random" in result
        assert "vs_checkpoint_pool" in result
        assert result["vs_random"]["requested_games"] == 1
        assert result["vs_checkpoint_pool"]["requested_games"] == 1

    json_path = tmp_path / "evaluation.json"
    csv_path = tmp_path / "evaluation.csv"
    eval_mod.write_json_results(results, str(json_path))
    eval_mod.write_csv_results(results, str(csv_path))

    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(json_data) == 2

    csv_rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(csv_rows) == 2
    assert {row["name"] for row in csv_rows} == {"checkpoint_a.pt", "checkpoint_b.pt"}


def test_evaluation_cli_main_with_checkpoint_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    checkpoint_dir = tmp_path / "models"
    (checkpoint_dir / "phase1").mkdir(parents=True)
    (checkpoint_dir / "selfplay").mkdir(parents=True)
    write_checkpoint(checkpoint_dir / "phase1" / "checkpoint_iter_5.pt", iteration=5)
    write_checkpoint(checkpoint_dir / "selfplay" / "selfplay_checkpoint_iter_6.pt", iteration=6)

    json_path = tmp_path / "results" / "evaluation.json"
    csv_path = tmp_path / "results" / "evaluation.csv"

    exit_code = eval_mod.main(
        [
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--pattern",
            "*checkpoint_iter_",
            "--games-random",
            "1",
            "--games-pool",
            "1",
            "--device",
            "cpu",
            "--json-out",
            str(json_path),
            "--csv-out",
            str(csv_path),
        ]
    )

    assert exit_code == 0
    assert json_path.exists()
    assert csv_path.exists()

    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert {entry["name"] for entry in json_data} == {
        "checkpoint_iter_5.pt",
        "selfplay_checkpoint_iter_6.pt",
    }
