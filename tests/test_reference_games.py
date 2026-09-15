from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from src.solver.games import Action as A
from src.solver.games import new_game
from src.solver.tree import GameTree


def dealt(game, cards):
    state = new_game(game)
    for card in cards:
        state = state.deal(card)
    return state


def play(state, *actions):
    for action in actions:
        state = state.play(action)
    return state


def test_kuhn_betting_and_net_payoffs():
    state = dealt("kuhn", (2, 0))
    assert state.actor == 0 and state.actions() == (A.CHECK, A.RAISE)
    assert play(state, A.CHECK, A.CHECK).returns() == (1, -1)
    assert play(state, A.RAISE, A.CALL).returns() == (2, -2)
    assert play(state, A.CHECK, A.RAISE, A.FOLD).returns() == (-1, 1)
    assert play(state, A.CHECK, A.RAISE, A.CALL).returns() == (2, -2)
    with pytest.raises(ValueError, match="Illegal"):
        play(state, A.RAISE, A.RAISE)
    with pytest.raises(ValueError, match="terminal"):
        state.returns()


def test_leduc_raise_cap_round_sizes_and_folds():
    state = dealt("leduc", (0, 2))
    raised = play(state, A.RAISE, A.RAISE)
    assert raised.committed == (3, 5)
    assert raised.actions() == (A.FOLD, A.CALL)
    assert raised.play(A.FOLD).returns() == (-3, 3)
    flop = raised.play(A.CALL)
    assert flop.actor == -1 and flop.committed == (5, 5)
    assert dict(flop.chance_outcomes()) == {1: 0.25, 3: 0.25, 4: 0.25, 5: 0.25}
    flop = flop.deal(1)
    assert flop.actor == 0 and flop.street_bets == (0, 0)
    showdown = play(flop, A.RAISE, A.RAISE, A.CALL)
    assert showdown.committed == (13, 13)
    assert showdown.returns() == (13, -13)  # Jack pair beats unpaired queen.
    assert play(flop, A.CHECK, A.RAISE, A.FOLD).returns() == (-5, 5)


def test_leduc_ties_high_cards_and_card_removal():
    same_rank = dealt("leduc", (2, 3))
    flop = play(same_rank, A.CHECK, A.CHECK).deal(0)
    assert play(flop, A.RAISE, A.CALL).returns() == (0, 0)
    high = play(dealt("leduc", (4, 2)), A.CHECK, A.CHECK).deal(0)
    assert play(high, A.CHECK, A.CHECK).returns() == (1, -1)
    with pytest.raises(ValueError, match="available"):
        play(same_rank, A.CHECK, A.CHECK).deal(2)
    with pytest.raises(ValueError, match="chance"):
        flop.deal(1)


@pytest.mark.parametrize(
    "game, nodes, infosets", [("kuhn", 58, 12), ("leduc", 9457, 288)]
)
def test_complete_tree_preserves_information_and_probability(game, nodes, infosets):
    tree = GameTree(game)
    assert len(tree.states) == nodes and len(tree.information_sets) == infosets
    masses = [Fraction(0)] * nodes
    masses[0] = Fraction(1)
    for node, state in enumerate(tree.states):
        if state.terminal:
            assert sum(state.returns()) == 0
            if state.folded is None:
                assert state.committed[0] == state.committed[1]
            continue
        children = tree.children[node]
        for child in children:
            masses[child] = masses[node] / len(children)
        if state.actor == -1:
            assert sum(p for _, p in state.chance_outcomes()) == pytest.approx(1)
        else:
            key = state.information_set()
            assert key.player == state.actor
            assert key.card == state.rank(state.private[state.actor])
            assert key.actions == state.actions()
    assert sum(m for m, s in zip(masses, tree.states) if s.terminal) == 1
    policy = tree.mask / tree.mask.sum(axis=1, keepdims=True)
    edges = tree.edge_probabilities(policy)
    reach = tree.reaches(edges).prod(axis=1)
    expected = sum(
        float(m) * s.returns()[0] for m, s in zip(masses, tree.states) if s.terminal
    )
    assert tree.values(edges)[0] == pytest.approx(expected)
    assert reach[tree.actor == -2].sum() == pytest.approx(1)


def test_information_sets_hide_opponent_cards_and_retain_public_history():
    first = dealt("leduc", (0, 2))
    assert first.information_set() == dealt("leduc", (0, 4)).information_set()
    assert first.information_set() == dealt("leduc", (1, 4)).information_set()
    assert first.information_set() != dealt("leduc", (2, 4)).information_set()
    flop = play(first, A.CHECK, A.CHECK).deal(4)
    other_history = play(first, A.RAISE, A.CALL).deal(4)
    assert flop.information_set() != other_history.information_set()
    assert replace(flop, board=3).information_set() != flop.information_set()


def test_policy_validation_rejects_bad_distributions():
    tree = GameTree("kuhn")
    uniform = tree.mask / tree.mask.sum(axis=1, keepdims=True)
    tree.validate_policy(uniform)
    for bad in (uniform[:, :2], uniform * 2, np.full_like(uniform, np.nan)):
        with pytest.raises(ValueError, match="probabilities"):
            tree.validate_policy(bad)
