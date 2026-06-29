"""Shared helpers for constructing agents from checkpoint files."""

import os

from src.utils.actions import sanitize_action
from src.utils.checkpoints import (
    AGENT_TYPE_OPPONENT_MODELING,
    AGENT_TYPE_STANDARD,
    checkpoint_uses_opponent_modeling_state,
    load_checkpoint,
)


class CheckpointAgent:
    """Agent wrapper that loads the right implementation from checkpoint metadata."""

    def __init__(
        self,
        player_id,
        model_path,
        device="cpu",
        sanitize_actions=True,
        with_opponent_modeling=None,
    ):
        self.player_id = player_id
        self.model_path = str(model_path)
        self.device = device
        self.sanitize_actions = sanitize_actions
        self.name = f"Model Agent {player_id} ({os.path.basename(self.model_path)})"

        checkpoint = load_checkpoint(self.model_path, map_location=device)
        self.with_opponent_modeling = checkpoint_uses_opponent_modeling_state(checkpoint)
        self.agent = create_agent_for_checkpoint(
            self.model_path,
            player_id=player_id,
            device=device,
            checkpoint=checkpoint,
        )

    def choose_action(self, state):
        action = self.agent.choose_action(state)
        if self.sanitize_actions:
            return sanitize_action(state, action)
        return action


def checkpoint_uses_opponent_modeling(model_path):
    """Detect whether a checkpoint contains opponent-modeling weights."""
    checkpoint = load_checkpoint(model_path, map_location="cpu")
    return checkpoint_uses_opponent_modeling_state(checkpoint)


def create_agent_for_checkpoint(model_path, player_id, device="cpu", checkpoint=None):
    """Create and load the correct agent implementation for a checkpoint."""
    if checkpoint is None:
        checkpoint = load_checkpoint(model_path, map_location=device)

    if checkpoint_uses_opponent_modeling_state(checkpoint):
        from src.opponent_modeling.deep_cfr_with_opponent_modeling import (
            DeepCFRAgentWithOpponentModeling,
        )

        agent = DeepCFRAgentWithOpponentModeling(player_id=player_id, num_players=6, device=device)
    else:
        from src.core.deep_cfr import DeepCFRAgent

        agent = DeepCFRAgent(player_id=player_id, num_players=6, device=device)

    agent.load_model(model_path)
    return agent


def agent_type_for_checkpoint(model_path):
    """Return the explicit agent type label for a checkpoint path."""
    if checkpoint_uses_opponent_modeling(model_path):
        return AGENT_TYPE_OPPONENT_MODELING
    return AGENT_TYPE_STANDARD
