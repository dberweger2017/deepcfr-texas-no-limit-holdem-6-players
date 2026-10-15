"""Pinned-engine parity and failure detection for the future N-player trainer's game."""

from copy import deepcopy
import json
import subprocess

import pytest

from scripts.check_nplayer_parity import BINARY, engine_identity, record, run

pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="build the native nplayer_parity example")


def check(records):
    return subprocess.run([str(BINARY)], input="".join(json.dumps(r) + "\n" for r in records),
                          text=True, capture_output=True, timeout=60)


@pytest.mark.parametrize("players", range(2, 7))
def test_pinned_engine_parity_for_all_supported_player_counts(tmp_path, players):
    result = run(BINARY, 160, 202610150000, players, tmp_path / "gate")
    assert result["status"] == "passed"
    assert result["hands"] == 160
    assert result["forced_all_in_hands"] == 20
    assert result["decisions"] > 100
    assert result["refund_hands"] > 0
    assert result["river_decisions"] > 0
    assert not (tmp_path / "gate/failure-hand.json").exists()


def rigged(holes, board, stacks, button, actions=(), force_all_in=True):
    def card(c):
        return "23456789TJQKA".index(c[0]) * 4 + "cdhs".index(c[1])
    n = len(stacks)
    deck = [card(holes[(button + offset) % n][r]) for r in range(2) for offset in range(1, n + 1)]
    deck += [card(c) for c in board]
    deck += [c for c in range(52) if c not in deck]
    return record(0, n, preset={"deck": deck, "stacks": stacks, "button": button,
                               "sb": 1, "bb": 2, "actions": actions, "force_all_in": force_all_in})


def test_fixed_side_pots_and_odd_chips_match_engine_and_expected_payouts():
    engine_identity()
    board = ["2c", "3d", "7h", "8s", "9c"]
    side = rigged([["Ac", "Ad"], ["Kc", "Kd"], ["Qc", "Qd"]], board, [50, 100, 200], 2)
    odd = rigged([["Ac", "Ad"], ["Ah", "As"], ["Kc", "Kd"]], board, [5, 5, 5], 2)
    folded = rigged([["2c", "3c"], ["2d", "3d"], ["2h", "3h"], ["2s", "3s"], ["4c", "5c"]],
                    ["Tc", "Jc", "Qc", "Kc", "Ac"], [5] * 5, 0,
                    [("raise", 5), ("call", 0), ("call", 0), ("fold", 0), ("fold", 0)])
    for hand, expected in [(side, [150, 100, 100]), (odd, [8, 7, 0]), (folded, [6, 4, 3, 6, 6])]:
        assert [x[0] for x in hand["states"][-1]["players"]] == expected
    result = check([side, odd, folded])
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(result.stdout.splitlines()) == 3


def test_two_spare_chips_go_to_distinct_winners_in_each_actual_pot():
    hand = rigged([["Ac", "3c"], ["6c", "7d"], ["Ad", "4c"], ["8c", "9d"], ["Ah", "5c"]],
                  ["Kc", "Qd", "Jh", "Ts", "2c"], [60, 97, 20, 169, 169], 1,
                  [("raise", 169), ("call", 0), ("fold", 0), ("call", 0), ("call", 0)])
    assert [x[0] for x in hand["states"][-1]["players"]] == [86, 97, 27, 0, 305]
    assert [pot[0] for pot in hand["settlement"]["pots"]] == [80, 120, 218]
    result = check([hand])
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("field", ["actor", "board", "street", "pot", "wager", "min_raise", "players", "holes", "legal", "rewards"])
def test_checker_rejects_corrupted_state_fields(field):
    hand = record(202610150000)
    mutated = deepcopy(hand)
    mutated["states"][0][field] = "corrupted"
    result = check([mutated])
    assert result.returncode != 0
    assert json.loads(result.stdout)["ok"] is False
    assert "state 0" in result.stdout


@pytest.mark.parametrize("field", ["contributions", "refunds", "pots"])
def test_checker_rejects_corrupted_settlement(field):
    hand = record(202610150000)
    hand["settlement"][field] = []
    result = check([hand])
    assert result.returncode != 0
    assert "pot/refund mismatch" in result.stdout


def test_checker_rejects_missing_transition_and_unsettled_trace():
    hand = record(202610150000)
    broken = deepcopy(hand)
    broken["states"].pop()
    assert check([broken]).returncode != 0
    hand["actions"].pop()
    hand["states"].pop()
    result = check([hand])
    assert result.returncode != 0
    assert "did not settle" in result.stdout


def test_checker_rejects_corrupted_public_history():
    hand = record(202610150000)
    hand["history"][0][1] = 5
    result = check([hand])
    assert result.returncode != 0
    assert "public history mismatch" in result.stdout


def test_driver_preserves_first_failure_and_does_not_claim_the_full_gate(tmp_path):
    # A fake checker rejects the first real engine fixture. This tests orchestration,
    # not correctness of the game, and must never be counted as a parity pass.
    binary = tmp_path / "reject.py"
    binary.write_text("#!/usr/bin/env python3\nimport sys\nsys.stdin.readline()\nprint('{\"ok\":false,\"error\":\"injected mismatch\"}', flush=True)\n")
    binary.chmod(0o755)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected mismatch"):
        run(binary, 10, 202610150000, 3, out)
    receipt = json.loads((out / "result.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt.get("hands", 0) == 0
    assert (out / "failure-hand.json").exists()
