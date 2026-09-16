import json
from dataclasses import replace
from hashlib import sha256

import pytest

from scripts.profile_holdem_collection import profile_root, traversal_digest
from src.game.hand import Hand
from src.holdem.checkpoint import save_training
from src.holdem.collection import collect_traversal, collection_seed
from src.holdem.training import HoldemTrainer
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import table
from tests.test_holdem_training import config


def test_checkpoint_profile_matches_direct_collection_and_retains_failures(tmp_path):
    with deterministic_cpu():
        trainer = HoldemTrainer(table(4, (6,) * 4), config())
        trainer.step()
    job = tmp_path / "saved"
    job.mkdir()
    checkpoint = job / "training-1.pt"
    digest = save_training(trainer, checkpoint, manifest={"fixture": "profiling"})
    (tmp_path / "manifest.json").write_text(json.dumps({"fixture": "profiling"}))
    (job / "training-1.json").write_text(json.dumps({"iteration": 1, "sha256": digest}))
    hand = Hand.start(
        replace(trainer.table, button=1),
        hand_id="collection-2-0-0",
        seed=collection_seed(config().seed, 2, 0, 0, "deal"),
    )
    expected = collect_traversal(
        hand,
        trainer.current_profile(),
        0,
        iteration=2,
        action_seed=collection_seed(config().seed, 2, 0, 0, "opponents"),
    )
    reports = [
        profile_root(
            tmp_path,
            "saved",
            1,
            0,
            0,
            10000,
            30,
            tmp_path / name,
            instrument=instrument,
        )
        for name, instrument in (("plain", False), ("profiled", True))
    ]
    for report in reports:
        assert report["status"] == "completed"
        assert report["traversal_sha256"] == traversal_digest(expected)
        assert report["nodes"] == expected.nodes
        assert report["table"]["button"] == 1
        assert report["checkpoint_sha256"] == digest
    assert (tmp_path / "profiled/collection.prof").is_file()
    failure = profile_root(tmp_path, "saved", 1, 0, 0, 1, 30, tmp_path / "limited")
    assert failure["status"] == "collection_limit"
    assert "traversal_sha256" not in failure
    assert "no batch returned" in failure["error"]
    assert json.loads((tmp_path / "limited/report.json").read_text()) == json.loads(json.dumps(failure))
    assert sha256(checkpoint.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        profile_root(tmp_path, "saved", 1, 0, 0, 1, 30, tmp_path / "limited")
