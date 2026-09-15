import numpy as np
import pokers as pkrs
from src.game.legacy import TrackedState

from src.agents.random_agent import RandomAgent
from src.opponent_modeling.deep_cfr_with_opponent_modeling import (
    DeepCFRAgentWithOpponentModeling,
)


def test_table_opponent_features_are_zero_without_history():
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=3, device="cpu")

    assert np.array_equal(
        agent.get_table_opponent_features(
            TrackedState.from_seed(3, 0, 1, 2, 20, 0).observe(0)
        ),
        np.zeros(20, dtype=np.float32),
    )


def test_table_opponent_features_average_known_opponent_histories(monkeypatch):
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=3, device="cpu")
    agent.opponent_modeling.opponent_histories = {
        "seat-1": [object()],
        "seat-2": [object()],
    }

    def fake_features(opponent_id):
        return np.full(20, int(opponent_id[-1]), dtype=np.float32)

    monkeypatch.setattr(agent.opponent_modeling, "get_opponent_features", fake_features)

    assert np.array_equal(
        agent.get_table_opponent_features(
            TrackedState.from_seed(3, 0, 1, 2, 20, 0).observe(0)
        ),
        np.full(20, 1.5, dtype=np.float32),
    )


def test_record_opponent_action_stores_action_and_context():
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=3, device="cpu")
    state = TrackedState.from_seed(
        n_players=3,
        button=0,
        sb=1,
        bb=2,
        stake=20.0,
        seed=0,
    )

    agent.record_opponent_action(
        state.observe(agent.player_id), action_id=2, opponent_id=1
    )

    history = agent.current_game_history["seat-1"]
    assert len(history["actions"]) == 1
    assert len(history["contexts"]) == 1
    assert np.array_equal(history["actions"][0], np.array([0.0, 0.0, 1.0, 0.0]))
    assert history["contexts"][0].shape == (25,)


def test_om_traversal_stores_table_features_in_replay(monkeypatch):
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=3, device="cpu")
    feature_vector = np.full(20, 0.25, dtype=np.float32)
    monkeypatch.setattr(
        agent, "get_table_opponent_features", lambda state: feature_vector
    )

    state = TrackedState.from_seed(
        n_players=3,
        button=0,
        sb=1,
        bb=2,
        stake=20.0,
        seed=0,
    )
    opponents = [None, RandomAgent(1), RandomAgent(2)]

    agent.cfr_traverse(state, iteration=1, opponents=opponents)

    assert len(agent.advantage_memory) > 0
    _, opponent_features, _, _, _ = agent.advantage_memory.buffer[0]
    assert np.array_equal(opponent_features, feature_vector)


def test_features_follow_identity_when_seats_change_and_ignore_replacements(
    monkeypatch,
):
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=3, device="cpu")
    agent.opponent_modeling.opponent_histories = {"alice": [object()]}
    monkeypatch.setattr(
        agent.opponent_modeling,
        "get_opponent_features",
        lambda identity: np.full(20, 7, dtype=np.float32),
    )
    for ids, expected in (
        (["hero", "alice", "bob"], 7),
        (["hero", "bob", "alice"], 7),
        (["hero", "new", "bob"], 0),
    ):
        state = TrackedState.from_seed(3, 0, 1, 2, 20, 0, player_ids=ids)
        assert np.array_equal(
            agent.get_table_opponent_features(state.observe(0)),
            np.full(20, expected, dtype=np.float32),
        )
        assert agent._opponent_identity(state.observe(0), 1) == ids[1]


def test_recorded_outcome_uses_identity_and_rejects_another_hand():
    import pytest

    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=3, device="cpu")
    state = TrackedState.from_seed(
        3, 0, 1, 2, 20, 0, player_ids=("hero", "alice", "bob"), hand_id="one"
    )
    agent.record_opponent_action(state.observe(0), 1, 1)
    other = TrackedState.from_seed(3, 0, 1, 2, 20, 1, hand_id="two")
    with pytest.raises(ValueError, match="previous hand"):
        agent.record_opponent_action(other.observe(0), 1, 1)
    with pytest.raises(ValueError, match="completed hand"):
        agent.end_game_recording(state.observe(0))
    while not state.final_state:
        action = (
            pkrs.ActionEnum.Check
            if pkrs.ActionEnum.Check in state.legal_actions
            else pkrs.ActionEnum.Fold
        )
        state = state.apply_action(pkrs.Action(action))
    agent.end_game_recording(state.observe(0))
    histories = agent.opponent_modeling.opponent_histories
    assert set(histories) == {"alice"}
    assert histories["alice"][0][2] == state.players_state[1].reward
    assert np.isfinite(agent.opponent_modeling.train(batch_size=1))
    assert agent.current_game_history == {}
