import json
import subprocess
import sys


def test_neural_collection_reproduces_in_a_fresh_process():
    command = [
        sys.executable,
        "-m",
        "scripts.check_holdem_collection",
        "--policy",
        "neural",
    ]
    reports = [
        json.loads(subprocess.check_output(command, text=True, timeout=45))
        for _ in range(2)
    ]
    first, second = reports
    for key in (
        "collection_sha256",
        "profile_sha256",
        "source_sha256",
        "traversals",
        "config",
    ):
        assert first[key] == second[key]
    for report in reports:
        assert report["reproduced"] and report["stack_bb"] == 100
        assert report["collection_schema"] == "holdem-external-sampling-v1"
        assert {r["seat"] for r in report["traversals"]} == set(range(6))
        assert all(
            r["targets"] > 0 and r["terminals"] > 0 for r in report["traversals"]
        )
        assert report["environment"]["engine"]["commit"]
