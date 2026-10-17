"""Lossless native transport and strict corruption checks."""

import json

import numpy as np
import pytest

from src.blueprint.hu20_turn_solver import PROFILE_FORMAT, SolveFailure, parse_profiles


def fixture(tmp_path):
    hands = [["2c", "3c"], ["4c", "5c"]]
    actions = [{"kind": "Check"}, {"kind": "Bet", "amount": 100}]
    node = dict(line=[], terminal=False, street="river", player=0, actions=actions,
                names=["check", "bet"], native_actions=[
                    {"kind": "check", "raise_to": None}, {"kind": "raise", "raise_to": 100}])
    request = dict(initial_street="river", board=["Ac", "Kd", "Qh", "Js", "Tc"],
                   nodes=[node], ranges=[[dict(hand=h, weight=.5) for h in hands]] * 2)
    # The old Value export promotes f32 to f64 before JSON serialization.
    strategy = np.asarray([.2, .3, .8, .7], dtype=np.float32).astype(np.float64)
    row = dict(line=[], board=request["board"], player=0, actions=actions, holdings=hands,
               strategy=strategy.tolist(), action_values_chips=[-0., None, 17., -3.5])
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps(row) + "\n")
    arrays = dict(strategy=strategy, action_values_chips=np.asarray([-0., np.nan, 17., -3.5]),
                  own_weights=np.asarray([.5, .5]), opponent_weights=np.asarray([.5, .5]))
    metadata = {k: json.loads(json.dumps(row[k])) for k in ("line", "board", "player", "actions")}
    metadata["arrays"] = {}
    offset = 0
    with (tmp_path / "profile.f64").open("wb") as out:
        for field, values in arrays.items():
            metadata["arrays"][field] = dict(offset=offset, count=len(values))
            out.write(values.astype("<f8").tobytes())
            offset += values.size * 8
    header = dict(format=PROFILE_FORMAT, binary_file="profile.f64", holdings=[hands, hands])
    path = tmp_path / "profile.jsonl"
    path.write_text(json.dumps(header) + "\n" + json.dumps(metadata) + "\n")
    return request, old, path, header, metadata


def test_promoted_f64_transport_preserves_normalized_strategy_ev_and_null_mask(tmp_path):
    request, old, path, _, _ = fixture(tmp_path)
    baseline = parse_profiles(request, old)[()]
    compact = parse_profiles(request, path, compact=True)[()]
    assert compact.menu == baseline.menu and compact.holdings == baseline.holdings
    assert np.array_equal(compact.probabilities.view(np.uint64), baseline.probabilities.view(np.uint64))
    assert np.array_equal(compact.action_values_chips, baseline.action_values_chips, equal_nan=True)
    assert np.array_equal(np.signbit(compact.action_values_chips), np.signbit(baseline.action_values_chips))
    assert not compact.probabilities.flags.writeable


@pytest.mark.parametrize("damage", ("truncated", "trailing", "offset", "count", "actions", "strategy", "lock"))
def test_compact_transport_keeps_identity_bounds_normalization_and_lock_checks(tmp_path, damage):
    request, _, path, header, metadata = fixture(tmp_path)
    binary = tmp_path / "profile.f64"
    if damage == "truncated":
        binary.write_bytes(binary.read_bytes()[:-1])
    elif damage == "trailing":
        with binary.open("ab") as stream:
            stream.write(b"x")
    elif damage == "offset":
        metadata["arrays"]["strategy"]["offset"] = 8
    elif damage == "count":
        metadata["arrays"]["strategy"]["count"] += 1
    elif damage == "actions":
        metadata["actions"][1]["amount"] = 101
    elif damage == "strategy":
        with binary.open("r+b") as stream:
            stream.write(np.asarray([np.nan], dtype="<f8").tobytes())
    elif damage == "lock":
        request["locks"] = [dict(line=[], player=0, actions=request["nodes"][0]["actions"],
            holdings=header["holdings"][0], strategy=[.5] * 4)]
    path.write_text(json.dumps(header) + "\n" + json.dumps(metadata) + "\n")
    with pytest.raises(SolveFailure):
        parse_profiles(request, path, compact=True)


def test_owned_matrices_survive_removal_of_successful_native_scratch(tmp_path, monkeypatch):
    from src.blueprint import river_evidence as evidence
    request, old, path, _, _ = fixture(tmp_path)
    expected = parse_profiles(request, old)[()]
    actual = parse_profiles(request, path, compact=True)[()]
    (tmp_path / "request.json").write_text(json.dumps(request))
    pins = {p.name: evidence.file_pin(p) for p in tmp_path.iterdir() if p.is_file()}
    monkeypatch.setattr(evidence, "selected_body", lambda _: False)
    evidence.seal_evidence(tmp_path, request, {"status": "completed"}, pins)
    assert not path.exists() and not (tmp_path / "profile.f64").exists()
    assert np.array_equal(actual.probabilities.view(np.uint64), expected.probabilities.view(np.uint64))
    assert np.array_equal(actual.action_values_chips, expected.action_values_chips, equal_nan=True)
