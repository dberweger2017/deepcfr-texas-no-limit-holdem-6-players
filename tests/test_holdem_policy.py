from dataclasses import replace

import pytest
import torch

from src.game.hand import Hand
from src.game.types import TableSeat
from src.holdem.actions import bet_candidates
from src.holdem.betting import BettingNetwork
from src.holdem.policy import FrozenProfile
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import DECK, table
from tests.test_holdem_encoding import change_suits


@pytest.fixture(autouse=True)
def cpu():
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(281)
        yield


def folding_model():
    model = BettingNetwork(8)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.regret.bias.fill_(-1)
    return model


def test_profile_copies_weights_without_changing_the_training_model():
    model = BettingNetwork(16).train()
    random_state = torch.get_rng_state().clone()
    profile = FrozenProfile([model] * 6)
    assert torch.equal(torch.get_rng_state(), random_state)
    assert model.training and all(p.requires_grad for p in model.parameters())
    assert len({next(m.parameters()).data_ptr() for m in profile._models}) == 6
    assert all(
        not m.training and all(not p.requires_grad for p in m.parameters())
        for m in profile._models
    )
    hand = Hand.start(table(), hand_id="frozen", seed=31)
    candidates = bet_candidates(hand.observe(hand.actor))
    before = profile.distribution(candidates)
    original_hash = profile.fingerprint
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1)
    profile.assert_unchanged()
    assert profile.distribution(candidates) == before
    assert FrozenProfile([model] * 6).fingerprint != original_hash


def test_fingerprints_cover_weights_widths_uniform_slots_and_physical_ownership():
    model = folding_model()
    profile = FrozenProfile([model, None, None, None])
    assert profile.fingerprint == FrozenProfile([model, None, None, None]).fingerprint
    assert profile.fingerprint != FrozenProfile([None, model, None, None]).fingerprint
    assert profile.fingerprint != FrozenProfile([None] * 4).fingerprint
    assert (
        profile.fingerprint
        != FrozenProfile([BettingNetwork(16), None, None, None]).fingerprint
    )


def test_sparse_physical_seats_select_the_right_policy():
    config = replace(
        table(4),
        seat_numbers=(0, 2, 3, 5),
        capacity=6,
        table_seats=(
            TableSeat(0, "player-0", 200, "playing"),
            TableSeat(1, "away", 200, "sitting_out"),
            TableSeat(2, "player-1", 200, "playing"),
            TableSeat(3, "player-2", 200, "playing"),
            TableSeat(5, "player-3", 200, "playing"),
        ),
    )
    hand = Hand.start(config, hand_id="sparse", seed=3)
    assert hand.actor == 3
    candidates = bet_candidates(hand.observe(hand.actor))
    models = [None] * 6
    models[3] = folding_model()
    assert FrozenProfile(models).distribution(candidates) == (
        1 / len(candidates.actions),
    ) * len(candidates.actions)
    models[5] = folding_model()
    assert FrozenProfile(models).distribution(candidates) == (1.0,) + (0.0,) * (
        len(candidates.actions) - 1
    )


@pytest.mark.parametrize("n", [4, 5, 6])
def test_profile_queries_use_only_the_owners_public_candidate_input(n):
    hand = Hand.from_deck(table(n), hand_id="private", deck=DECK)
    view = hand.observe(hand.actor)
    positions = [i for i, card in enumerate(DECK) if card not in view.hole_cards]
    deck = list(DECK)
    for i, j in zip(positions, reversed(positions)):
        deck[i] = DECK[j]
    alternate = Hand.from_deck(table(n), hand_id="private", deck=tuple(deck))
    profile = FrozenProfile([BettingNetwork(8) for _ in range(n)])
    expected = profile.distribution(bet_candidates(view))
    assert (
        profile.distribution(bet_candidates(alternate.observe(alternate.actor)))
        == expected
    )
    assert (
        profile.distribution(
            bet_candidates(change_suits(view, dict(zip("cdhs", "hsdc"))))
        )
        == expected
    )
    for invalid in (hand, hand._state, view):
        with pytest.raises(TypeError):
            profile.distribution(invalid)


def test_internal_weight_or_mode_changes_are_detected():
    profile = FrozenProfile([folding_model(), None])
    profile._models[0].train()
    with pytest.raises(RuntimeError, match="changed"):
        profile.assert_unchanged()
    profile._models[0].eval()
    with torch.no_grad():
        profile._models[0].regret.bias.add_(1)
    with pytest.raises(RuntimeError, match="changed"):
        profile.assert_unchanged()


def test_profile_rejects_missing_roles_and_nonfinite_models():
    for models in ([None], [None] * 7, [object(), None]):
        with pytest.raises(ValueError):
            FrozenProfile(models)
    model = folding_model()
    with torch.no_grad():
        model.regret.bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        FrozenProfile([model, None])
    hand = Hand.start(table(4), hand_id="wrong-capacity", seed=9)
    with pytest.raises(ValueError, match="table"):
        FrozenProfile([None] * 6).distribution(bet_candidates(hand.observe(hand.actor)))
