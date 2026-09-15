import random

import pokers as pkrs

from src.game.legacy import require_policy_view
from src.utils import settings
from src.utils.actions import ActionMappingFailure, preset_raise_action


class RandomAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.name = f"RandomAgent_{player_id}"

    def choose_action(self, state):
        require_policy_view(state)
        if not state.legal_actions:
            raise ActionMappingFailure("No legal action for this player")
        action = random.choice(state.legal_actions)
        if action != pkrs.ActionEnum.Raise:
            return pkrs.Action(action)
        return preset_raise_action(
            state,
            random.choice(["min", "half_pot", "pot", "all_in"]),
            strict=settings.is_strict_checking(),
        )
