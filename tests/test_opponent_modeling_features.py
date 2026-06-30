import numpy as np
import pokers as pkrs

from src.agents.random_agent import RandomAgent
from src.opponent_modeling.deep_cfr_with_opponent_modeling import (
    DeepCFRAgentWithOpponentModeling,
)


def test_table_opponent_features_are_zero_without_history():
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=3, device="cpu")

    assert np.array_equal(agent.get_table_opponent_features(), np.zeros(20, dtype=np.float32))


def test_table_opponent_features_average_known_opponent_histories(monkeypatch):
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=3, device="cpu")
    agent.opponent_modeling.opponent_histories = {
        1: [object()],
        2: [object()],
    }

    def fake_features(opponent_id):
        return np.full(20, opponent_id, dtype=np.float32)

    monkeypatch.setattr(agent.opponent_modeling, "get_opponent_features", fake_features)

    assert np.array_equal(
        agent.get_table_opponent_features(),
        np.full(20, 1.5, dtype=np.float32),
    )


def test_record_opponent_action_stores_action_and_context():
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=3, device="cpu")
    state = pkrs.State.from_seed(
        n_players=3,
        button=0,
        sb=1,
        bb=2,
        stake=20.0,
        seed=0,
    )

    agent.record_opponent_action(state, action_id=2, opponent_id=1)

    history = agent.current_game_history[1]
    assert len(history["actions"]) == 1
    assert len(history["contexts"]) == 1
    assert np.array_equal(history["actions"][0], np.array([0.0, 0.0, 1.0, 0.0]))
    assert history["contexts"][0].shape == (25,)


def test_om_traversal_stores_table_features_in_replay(monkeypatch):
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=3, device="cpu")
    feature_vector = np.full(20, 0.25, dtype=np.float32)
    monkeypatch.setattr(agent, "get_table_opponent_features", lambda: feature_vector)

    state = pkrs.State.from_seed(
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
