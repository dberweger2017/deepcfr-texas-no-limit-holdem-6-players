from dataclasses import replace
from random import Random

import pytest
import torch

from src.game.hand import Hand
from src.game.types import Action, ActionKind, Street
from src.holdem.actions import bet_candidates, record_execution
from src.holdem.betting import ActionScores, BettingNetwork, betting_loss
from src.holdem.targets import action_targets
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import DECK, call_down, table
from tests.test_holdem_encoding import advance, change_suits, rotate


@pytest.fixture(autouse=True)
def cpu():
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(74)
        yield


def opening(n=6):
    hand = Hand.start(table(n), hand_id="targets", seed=17)
    return bet_candidates(hand.observe(hand.actor))


def supervision(candidates):
    size = len(candidates.actions)
    return action_targets(
        candidates,
        (1 / size,) * size,
        {a: 2 * i for i, a in enumerate(candidates.actions)},
    )


def test_targets_use_separate_branch_values_and_a_policy_weighted_baseline():
    candidates = opening()
    # Fold -2 chips, call +4, and successively different values for each raise.
    values = [-2, 4] + list(range(6, 6 + 2 * (len(candidates.actions) - 2), 2))
    policy = [0.0] * len(values)
    policy[0], policy[1] = 0.25, 0.75
    target = action_targets(
        candidates, policy, dict(reversed(list(zip(candidates.actions, values))))
    )
    assert target.values_bb == tuple(v / 2 for v in values)
    assert target.regrets_bb == tuple((v - 2.5) / 2 for v in values)
    assert sum(p * r for p, r in zip(policy, target.regrets_bb)) == 0
    assert len(set(target.regrets_bb[2:])) == len(values) - 2
    assert target.regrets_bb[2] > 0  # An unchosen raise still receives supervision.


@pytest.mark.parametrize(
    "bad", ["missing", "extra", "nan", "policy-length", "negative", "sum", "infinite"]
)
def test_targets_reject_incomplete_or_invalid_branch_evaluations(bad):
    candidates = opening()
    n = len(candidates.actions)
    policy = [1 / n] * n
    values = {a: 0 for a in candidates.actions}
    if bad == "missing":
        del values[candidates.actions[-1]]
    elif bad == "extra":
        values[Action(ActionKind.RAISE, 199)] = 2
    elif bad == "nan":
        values[candidates.actions[0]] = float("nan")
    elif bad == "policy-length":
        policy.pop()
    elif bad == "negative":
        policy[0] = -0.1
    elif bad == "sum":
        policy = [0] * n
    else:
        policy[0] = float("inf")
    with pytest.raises(ValueError):
        action_targets(candidates, policy, values)


def test_action_conditioned_heads_are_equivariant_to_candidate_order():
    candidates = opening()
    order = tuple(reversed(range(len(candidates.actions))))
    permuted = replace(
        candidates,
        actions=tuple(candidates.actions[i] for i in order),
        features=tuple(candidates.features[i] for i in order),
    )
    model = BettingNetwork(32).eval()
    original, changed = model([candidates, permuted])
    torch.testing.assert_close(changed.regrets, original.regrets[list(order)])
    torch.testing.assert_close(changed.values, original.values[list(order)])
    assert original.regrets[2:].unique().numel() > 1
    assert original.values[2:].unique().numel() > 1
    target = supervision(candidates)
    permuted_target = action_targets(
        permuted,
        tuple(target.policy[i] for i in order),
        {a: target.values_bb[i] * 2 for i, a in enumerate(candidates.actions)},
    )
    torch.testing.assert_close(
        betting_loss([changed], [permuted_target]), betting_loss([original], [target])
    )


def test_mixed_candidate_counts_match_single_inference_without_padding():
    batch = [opening(4), opening(5)]
    hand = advance(Hand.start(table(6), hand_id="long", seed=17), Street.RIVER)
    batch.append(bet_candidates(hand.observe(hand.actor)))
    short = Hand.start(table(4, (5, 200, 200, 200)), hand_id="short", seed=20).apply(
        Action(ActionKind.RAISE, 10)
    )
    batch.append(bet_candidates(short.observe(short.actor)))
    assert len({len(c.actions) for c in batch}) > 1
    model = BettingNetwork(32).eval()
    combined = model(batch)
    for c, score in zip(batch, combined):
        single = model([c])[0]
        assert score.regrets.shape == score.values.shape == (len(c.actions),)
        torch.testing.assert_close(score.regrets, single.regrets, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(score.values, single.values, rtol=1e-5, atol=1e-6)
        assert score.probabilities().sum().item() == pytest.approx(1)
        rng1, rng2 = Random(77), Random(77)
        first = [score.choose(rng1) for _ in range(20)]
        assert first == [score.choose(rng2) for _ in range(20)]
        assert set(first) <= set(c.actions)
    loss = betting_loss(combined, [supervision(c) for c in batch])
    loss.backward()
    for part in (model.encoder, model.action, model.regret, model.value):
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in part.parameters()
        )
        assert sum(p.grad.abs().sum() for p in part.parameters()) > 0


@pytest.mark.parametrize("n", [4, 5, 6])
def test_policy_respects_seat_and_suit_symmetry(n):
    hand = Hand.from_deck(table(n), hand_id="symmetry", deck=DECK)
    view = hand.observe(hand.actor)
    positions = [i for i, card in enumerate(DECK) if card not in view.hole_cards]
    deck = list(DECK)
    for i, j in zip(positions, reversed(positions)):
        deck[i] = DECK[j]
    alternate = Hand.from_deck(table(n), hand_id="symmetry", deck=tuple(deck))
    alternatives = [
        rotate(view, 1),
        change_suits(view, dict(zip("cdhs", "hsdc"))),
        alternate.observe(alternate.actor),
    ]
    model = BettingNetwork(32).eval()
    # Compare the same inference shape; mixed-batch numerics are checked separately.
    scores = [model([bet_candidates(v)])[0] for v in [view, *alternatives]]
    for score in scores[1:]:
        torch.testing.assert_close(scores[0].regrets, score.regrets, rtol=0, atol=0)
        torch.testing.assert_close(scores[0].values, score.values, rtol=0, atol=0)
        torch.testing.assert_close(
            scores[0].probabilities(), score.probabilities(), rtol=0, atol=0
        )


def test_regret_matching_uses_positive_regrets_not_values_or_softmax():
    candidates = opening()
    regrets = torch.full((len(candidates.actions),), -1.0)
    regrets[1], regrets[-1] = 2, 6
    values = torch.full_like(regrets, 1000)
    score = ActionScores(candidates, regrets, values)
    expected = torch.zeros_like(regrets).double()
    expected[1], expected[-1] = 0.25, 0.75
    torch.testing.assert_close(score.probabilities(), expected)
    regrets.fill_(-3)
    regrets[-2] = -1
    assert score.probabilities().argmax() == len(regrets) - 2
    assert score.probabilities().max() == 1
    regrets.fill_(0)
    assert score.probabilities()[0] == 1  # Stable first-candidate tie break.
    regrets[0] = float("nan")
    with pytest.raises(ValueError):
        score.probabilities()


def test_loss_preserves_every_candidate_gradient_and_separate_heads():
    candidates = opening()
    target = supervision(candidates)
    n = len(candidates.actions)
    regrets = torch.zeros(n, requires_grad=True)
    values = torch.zeros(n, requires_grad=True)
    loss = betting_loss([ActionScores(candidates, regrets, values)], [target])
    loss.backward()
    torch.testing.assert_close(regrets.grad, -2 * torch.tensor(target.regrets_bb))
    torch.testing.assert_close(values.grad, -2 * torch.tensor(target.values_bb))
    assert loss.item() == pytest.approx(
        sum(r * r + v * v for r, v in zip(target.regrets_bb, target.values_bb))
    )


def test_targets_cannot_be_attached_to_an_equal_encoding_with_different_source():
    candidates = opening()
    view = candidates.decision.source
    changed_source = replace(view, hand_id="another-hand")
    changed = replace(
        candidates, decision=replace(candidates.decision, source=changed_source)
    )
    assert candidates == changed  # Numerical equality deliberately excludes the source.
    score = BettingNetwork(16)([candidates])[0]
    with pytest.raises(ValueError, match="different"):
        betting_loss([score], [supervision(changed)])


def test_model_and_loss_reject_invalid_inputs():
    candidates = opening()
    model = BettingNetwork(16)
    for batch in ([], [candidates.decision.source]):
        with pytest.raises(TypeError):
            model(batch)
    for c in (
        replace(candidates, schema="future"),
        replace(candidates, actions=()),
        replace(candidates, features=()),
    ):
        with pytest.raises(ValueError):
            model([c])
    score = model([candidates])[0]
    target = supervision(candidates)
    with pytest.raises(ValueError):
        betting_loss([score], [])
    with pytest.raises(ValueError):
        betting_loss([replace(score, values=torch.zeros(1))], [target])
    with pytest.raises(FloatingPointError):
        betting_loss(
            [score],
            [replace(target, values_bb=(float("nan"),) * len(candidates.actions))],
        )


def test_branch_payoffs_come_from_settlement_and_remain_outside_policy_inputs():
    hand = advance(Hand.start(table(4), hand_id="settled-values", seed=7), Street.RIVER)
    view = hand.observe(hand.actor)
    candidates = bet_candidates(view)
    values = {}
    for index, action in enumerate(candidates.actions):
        child = hand.apply(action)
        event = child.events[len(hand.events)]
        assert record_execution(candidates, index, event).event.action == action
        terminal = call_down(child).observe(view.seat)
        values[action] = (
            terminal.players[view.seat].stack - view.players[view.seat].starting_stack
        )
    probabilities = [1 / len(values)] * len(values)
    targets = action_targets(candidates, probabilities, values)
    assert targets.values_bb == tuple(
        values[a] / view.big_blind for a in candidates.actions
    )
    assert sum(
        p * r for p, r in zip(targets.policy, targets.regrets_bb)
    ) == pytest.approx(0, abs=1e-12)
    assert hand.observe(hand.actor) == view
    assert bet_candidates(view) == candidates


def test_regret_matching_does_not_overflow_on_large_finite_predictions():
    candidates = opening()
    regrets = torch.full((len(candidates.actions),), 1e308, dtype=torch.float64)
    score = ActionScores(candidates, regrets, torch.zeros_like(regrets))
    torch.testing.assert_close(
        score.probabilities(), torch.full_like(regrets, 1 / len(regrets))
    )
