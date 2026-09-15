import pickle
import random

import numpy as np
import pokers
import pytest
import torch

from src.agents.random_agent import RandomAgent
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import encode_state
from src.game import Action, ActionKind
from src.game.hand import Hand
from src.game.legacy import TrackedState
from src.game.observation import HandFinished
from src.game.play import PlayerHistory, RandomPolicy, play_hand
from src.game.types import Street
from src.opponent_modeling.deep_cfr_with_opponent_modeling import (
    DeepCFRAgentWithOpponentModeling,
)
from src.utils.evaluation import choose_agent_action, evaluate_agent_matchup
from tests.test_hand_observations import DECK, public_values_only, table


@pytest.mark.parametrize("n", [4, 5, 6])
@pytest.mark.parametrize("street", [Street.PREFLOP, Street.FLOP, Street.TURN])
def test_hidden_cards_do_not_change_model_inputs_or_sampled_actions(n, street):
    observer = 3 if street == Street.PREFLOP else 1
    visible_board = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4}[street]
    protected = {(observer - 1) % n, (observer - 1) % n + n}
    protected.update(range(2 * n, 2 * n + visible_board))
    hidden = [i for i in range(52) if i not in protected]
    changed = list(DECK)
    shuffled = [changed[i] for i in hidden]
    random.Random(19).shuffle(shuffled)
    for i, card in zip(hidden, shuffled):
        changed[i] = card
    hands = [
        Hand.from_deck(table(n), hand_id="paired", deck=deck)
        for deck in (DECK, tuple(changed))
    ]
    for index, hand in enumerate(hands):
        while hand.observe(observer).street != street:
            legal = hand.observe(hand.actor).legal_actions
            hand = hand.apply(
                Action(
                    ActionKind.CHECK
                    if ActionKind.CHECK in legal.kinds
                    else ActionKind.CALL
                )
            )
        hands[index] = hand
    first, second = [h.observe(observer) for h in hands]
    assert first == second
    assert RandomPolicy(72).distribution(first) == RandomPolicy(72).distribution(second)
    assert RandomPolicy(72).choose_action(first) == RandomPolicy(72).choose_action(
        second
    )
    states = [TrackedState(h) for h in hands]
    for agent_class in (DeepCFRAgent, DeepCFRAgentWithOpponentModeling):
        torch.manual_seed(6)
        agent = agent_class(player_id=observer, num_players=n, device="cpu")
        views = [s.observe(observer) for s in states]
        inputs = [encode_state(view, observer) for view in views]
        assert np.array_equal(*inputs)
        with torch.no_grad():
            outputs = [
                agent.strategy_net(
                    torch.as_tensor(encoded, dtype=torch.float32).unsqueeze(0)
                )
                for encoded in inputs
            ]
        for first_output, second_output in zip(*outputs):
            torch.testing.assert_close(first_output, second_output, rtol=0, atol=0)
        decisions = []
        for state in states:
            np.random.seed(72)
            random.seed(72)
            action = choose_agent_action(agent, state)
            decisions.append((int(action.action), action.amount))
        assert decisions[0] == decisions[1]
        for view in views:
            assert not hasattr(view, "apply_action")
            assert not hasattr(view, "deck")
            assert not hasattr(view, "_hand")
            assert all(not p.hand for p in view.players_state if p.player != observer)


@pytest.mark.parametrize(
    "agent_class", [DeepCFRAgent, DeepCFRAgentWithOpponentModeling, RandomAgent]
)
def test_policies_reject_privileged_state(agent_class):
    agent = agent_class(player_id=0)
    raw = pokers.State.from_seed(6, 3, 1, 2, 200, 0)
    with pytest.raises(TypeError, match="observation"):
        agent.choose_action(raw)
    with pytest.raises(TypeError, match="tracked"):
        choose_agent_action(agent, raw)


def test_counterfactual_traversal_does_not_record_opponent_history():
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=3, device="cpu")
    agent.current_game_history = {
        1: {"actions": [np.zeros(4)], "contexts": [np.zeros(25)]}
    }
    before = pickle.dumps(
        (agent.current_game_history, agent.opponent_modeling.opponent_histories)
    )
    state = TrackedState.from_seed(3, 0, 1, 2, 4, 71, hand_id="counterfactual")
    original = state.observe(0)
    agent.cfr_traverse(
        state, iteration=1, opponents=[None, RandomAgent(1), RandomAgent(2)]
    )
    assert (
        pickle.dumps(
            (agent.current_game_history, agent.opponent_modeling.opponent_histories)
        )
        == before
    )
    assert state.observe(0) == original
    assert len(agent.advantage_memory) > 0


class CallObserver:
    def __init__(self, player_id):
        self.player_id = player_id
        self.views = []

    def choose_action(self, view):
        assert view.observation.seat == self.player_id
        assert not hasattr(view, "deck")
        public_values_only(view.observation)
        self.views.append(view.observation)
        return pokers.Action(
            pokers.ActionEnum.Check
            if pokers.ActionEnum.Check in view.legal_actions
            else pokers.ActionEnum.Call
        )


def test_evaluation_records_only_real_hands_in_each_players_history():
    agents = [CallObserver(i) for i in range(4)]
    result = evaluate_agent_matchup(
        agents[0], agents, num_games=3, num_players=4, strict=True
    )
    assert result["completed_games"] == 3
    for agent in agents:
        starts = {view.hand_id: view for view in agent.views}
        assert [len(v.previous_hands) for v in starts.values()] == [0, 1, 2]
        for view in agent.views:
            assert all(h.player_id == view.player_id for h in view.previous_hands)
            assert all(
                isinstance(h.events[-1], HandFinished) for h in view.previous_hands
            )


@pytest.mark.parametrize("n", [4, 5, 6])
def test_headless_runner_keeps_policy_instances_and_histories_separate(n):
    policies = {f"player-{i}": RandomPolicy(100 + i) for i in range(n)}
    histories = {}
    for deal in range(3):
        hand = play_hand(
            Hand.start(table(n), hand_id=f"hand-{deal}", seed=deal), policies, histories
        )
        next_histories = {
            identity: histories.get(identity, PlayerHistory(identity)).append(
                hand.observe(seat)
            )
            for seat, identity in enumerate(hand.table.player_ids)
        }
        assert all(len(h.hands) == deal + 1 for h in next_histories.values())
        assert all(len(h.hands) == deal for h in histories.values())
        histories = next_histories
    shared = RandomPolicy(1)
    with pytest.raises(ValueError, match="own policy"):
        play_hand(
            Hand.start(table(n), hand_id="shared", seed=1),
            {identity: shared for identity in policies},
        )


def test_own_cards_and_distinct_betting_histories_remain_visible():
    hand = Hand.start(table(3), hand_id="history", seed=13)
    raised_first = (
        hand.apply(Action(ActionKind.RAISE, 4))
        .apply(Action(ActionKind.CALL))
        .apply(Action(ActionKind.CALL))
    )
    raised_second = (
        hand.apply(Action(ActionKind.CALL))
        .apply(Action(ActionKind.RAISE, 4))
        .apply(Action(ActionKind.CALL))
        .apply(Action(ActionKind.CALL))
    )
    first, second = raised_first.observe(1), raised_second.observe(1)
    assert first.board == second.board
    assert first.players == second.players
    assert first.pots == second.pots
    assert first.history != second.history
    assert hand.observe(0).hole_cards != hand.observe(1).hole_cards
    assert hand.observe(0).player_id != hand.observe(1).player_id


def test_dispatch_rejects_an_agent_from_another_seat():
    state = TrackedState.from_seed(4, 0, 1, 2, 200, 0)
    with pytest.raises(ValueError, match="different seat"):
        choose_agent_action(RandomAgent(0), state)
    with pytest.raises(ValueError, match="current decision"):
        RandomAgent(0).choose_action(state.observe(0))
