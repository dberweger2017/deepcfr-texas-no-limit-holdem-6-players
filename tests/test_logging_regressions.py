from pathlib import Path
import importlib.machinery
import sys
import types

import pokers as pkrs
import pytest

def make_stub(name):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


matplotlib_stub = make_stub("matplotlib")
matplotlib_pyplot_stub = make_stub("matplotlib.pyplot")
matplotlib_stub.pyplot = matplotlib_pyplot_stub
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", matplotlib_pyplot_stub)
sys.modules.setdefault("pandas", make_stub("pandas"))
sys.modules.setdefault("seaborn", make_stub("seaborn"))

import scripts.visualize_tournament as tournament_mod
from src.utils.logging import apply_action_with_logging


class DummyInvalidAgent:
    def __init__(self, player_id=0):
        self.player_id = player_id

    def choose_action(self, state):
        return pkrs.Action(pkrs.ActionEnum.Raise, 1000.0)


def test_apply_action_with_logging_writes_utf8_log_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = pkrs.State.from_seed(
        n_players=6,
        button=0,
        sb=1,
        bb=2,
        stake=200.0,
        seed=0,
    )

    new_state, log_file, status = apply_action_with_logging(
        state,
        pkrs.Action(pkrs.ActionEnum.Raise, 1000.0),
        strict=False,
    )

    assert new_state is None
    assert status == pkrs.StateStatus.HighBet
    assert log_file is not None

    log_text = Path(log_file).read_text(encoding="utf-8")
    assert "State status not OK (StateStatus.HighBet)" in log_text
    assert any(suit in log_text for suit in "♣♦♥♠")


def test_tournament_logs_invalid_state_before_raising(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        tournament_mod,
        "load_agent_from_checkpoint",
        lambda checkpoint_path, player_id=0, device="cpu": DummyInvalidAgent(player_id),
    )

    with pytest.raises(ValueError, match=r"State status not OK during tournament game 1"):
        tournament_mod.run_tournament(["dummy.pt"], num_games=1, device="cpu")

    log_files = sorted((tmp_path / "logs").glob("poker_error_*.txt"))
    assert log_files

    log_text = log_files[-1].read_text(encoding="utf-8")
    assert "State status not OK during tournament game 1" in log_text
    assert "Action: ActionEnum.Raise" in log_text


def test_all_in_side_pot_check_cycle_resolves_to_showdown():
    state = pkrs.State.from_seed(
        n_players=6,
        button=1,
        sb=1.0,
        bb=2.0,
        stake=200.0,
        seed=61,
    )
    actions = [
        pkrs.Action(pkrs.ActionEnum.Raise, 4.679007411003113),
        pkrs.Action(pkrs.ActionEnum.Fold),
        pkrs.Action(pkrs.ActionEnum.Raise, 15.42989415118555),
        pkrs.Action(pkrs.ActionEnum.Raise, 49.631354041135765),
        pkrs.Action(pkrs.ActionEnum.Fold),
        pkrs.Action(pkrs.ActionEnum.Call),
        pkrs.Action(pkrs.ActionEnum.Call),
        pkrs.Action(pkrs.ActionEnum.Raise, 128.25974439667556),
        pkrs.Action(pkrs.ActionEnum.Call),
        pkrs.Action(pkrs.ActionEnum.Fold),
        pkrs.Action(pkrs.ActionEnum.Fold),
    ]

    for action in actions:
        state, _, status = apply_action_with_logging(state, action, strict=True)
        assert status == pkrs.StateStatus.Ok

    assert state.final_state
    assert state.stage == pkrs.Stage.Showdown
    assert state.legal_actions == []
    assert abs(sum(player.reward for player in state.players_state)) < 1e-9
