"""The paid-check runner uses the same saved state and paired arena contract."""

from json import dumps, loads

from scripts.check_blueprint_scale import main
from src.blueprint.artifact import save_training
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Table


def test_worker_measurements_and_learning_check_from_one_checkpoint(tmp_path):
    table = Table(
        ("player-0", "player-1"),
        (200, 200),
        small_blind=1,
        big_blind=2,
        chip_unit="1",
    )
    trainer = BlueprintTrainer(
        table,
        PilotConfig(seed=11, raise_cap=0, max_nodes=5000, max_seconds=30),
    )
    trainer.step()
    source = tmp_path / "source.json.gz"
    source_hash = save_training(trainer, source)
    for workers in (1, 2):
        assert main(
            [
                "worker",
                "--checkpoint",
                str(source),
                "--expected-sha256",
                source_hash,
                "--out",
                str(tmp_path / f"worker-{workers}"),
                "--workers",
                str(workers),
                "--steps",
                "2",
            ]
        ) == 0
    one = loads((tmp_path / "worker-1" / "result.json").read_text())
    two = loads((tmp_path / "worker-2" / "result.json").read_text())
    assert one["source_sha256"] == two["source_sha256"] == source_hash
    assert one["checkpoint_sha256"] == two["checkpoint_sha256"]
    assert one["nodes"] == two["nodes"]

    plan = tmp_path / "plan.json"
    plan.write_text(
        dumps(
            {
                "scenarios": [
                    {
                        "name": "heads-up",
                        "stacks": [200, 200],
                        "small_blind": 1,
                        "big_blind": 2,
                        "chip_unit": "1",
                    }
                ],
                "candidate": "blueprint",
                "baseline": "blueprint_uniform",
                "opponents": ["check_call"],
                "blocks": 2,
                "root_seed": 17,
                "split": "validation",
            }
        )
    )
    assert main(
        [
            "learning",
            "--checkpoint",
            str(source),
            "--expected-sha256",
            source_hash,
            "--plan",
            str(plan),
            "--out",
            str(tmp_path / "learning"),
        ]
    ) == 0
    learning = loads((tmp_path / "learning" / "result.json").read_text())
    assert learning["source_sha256"] == source_hash
    for strategy in ("current", "average"):
        result = learning["strategies"][strategy]
        assert result["status"] == "valid"
        assert result["completed_hands"] == 8
        assert result["invalid_actions"] == 0
        assert result["preflop_probe"]["hand_classes"] == 169
