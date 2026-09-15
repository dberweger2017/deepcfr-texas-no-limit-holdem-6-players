import json
import subprocess
import sys


def test_two_training_iterations_reproduce_in_a_fresh_process():
    command = [sys.executable, "-m", "scripts.check_holdem_training"]
    first, second = [
        json.loads(subprocess.check_output(command, text=True, timeout=90))
        for _ in range(2)
    ]
    for key in (
        "reports",
        "profile_sha256",
        "replay_sha256",
        "table",
        "config",
        "source_sha256",
    ):
        assert first[key] == second[key]
    assert first["reproduced"] and second["reproduced"]
    assert (
        first["reports"][1]["collection_profile"]
        == first["reports"][0]["fitted_profile"]
    )
    assert first["reports"][1]["fitted_profile"] == first["profile_sha256"]
    assert len(first["replay_sha256"]) == 6
    assert all(
        r["fit"] is not None for report in first["reports"] for r in report["roles"]
    )
    assert first["environment"]["engine"]["commit"]
