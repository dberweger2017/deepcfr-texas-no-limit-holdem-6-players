import json
import random
from dataclasses import asdict, replace
from hashlib import sha256

import numpy as np
import pytest
import torch

from src.arena.catalog import Checkpoint
from src.arena.frozen import FrozenNetwork, inference_runtime
from src.arena.registry import PolicyRegistry
from src.arena.run import reproduce, run
from src.arena.schedule import Plan, Scenario
from src.core.model import PokerNetwork
from src.game.hand import Hand, Table
from src.game.legacy import legacy_view
from src.game.types import Action, ActionKind, Street
from src.utils.actions import action_type_to_pokers_action
from tests.test_hand_observations import DECK, table


def checkpoint(tmp_path, players=4, seed=7, **extra):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = PokerNetwork(120 + 6 * players, hidden_size=16)
    path = tmp_path / f"fixture-{players}-{seed}.pt"
    torch.save(
        {
            "strategy_net": net.state_dict(),
            "agent_type": "standard",
            "num_players": players,
            "training_seed": seed,
            **extra,
        },
        path,
    )
    return Checkpoint(
        f"fixture-{players}-{seed}", str(path), sha256(path.read_bytes()).hexdigest()
    )


def view(players=4, seed=1):
    return Hand.start(
        Table(
            tuple(f"p{i}" for i in range(players)),
            (2000,) * players,
            button=(players - 3) % players,
        ),
        hand_id="hand",
        seed=seed,
    ).observe(0)


def test_loading_and_sampling_leave_global_random_streams_and_weights_unchanged(
    tmp_path,
):
    spec = checkpoint(tmp_path)
    torch_before = torch.random.get_rng_state().clone()
    python_before, numpy_before = random.getstate(), np.random.get_state()
    model = FrozenNetwork(spec, tmp_path / "fixture-4-7.pt")
    first, second = model.policy(123), model.policy(123)
    weights = {k: v.clone() for k, v in model.network.state_dict().items()}
    with inference_runtime():
        assert [first.choose_action(view()) for _ in range(20)] == [
            second.choose_action(view()) for _ in range(20)
        ]
    assert torch.equal(torch_before, torch.random.get_rng_state())
    assert random.getstate() == python_before
    after = np.random.get_state()
    assert np.array_equal(numpy_before[1], after[1]) and numpy_before[2:] == after[2:]
    assert not model.network.training
    assert all(not p.requires_grad for p in model.network.parameters())
    assert all(
        torch.equal(weights[k], value)
        for k, value in model.network.state_dict().items()
    )
    assert model.description["training_seed"] == 7


def test_frozen_distribution_preserves_legacy_sizing_and_masks_actions(tmp_path):
    spec = checkpoint(tmp_path)
    model = FrozenNetwork(spec, tmp_path / "fixture-4-7.pt")
    policy = model.policy(1)
    current = view()
    actions = policy.distribution(current)
    assert sum(probability for _, probability in actions) == pytest.approx(1)
    legacy = legacy_view(current)
    from src.core.model import encode_state

    with torch.inference_mode():
        logits, sizing = model.network(
            torch.tensor(encode_state(legacy, 0), dtype=torch.float32).unsqueeze(0)
        )
    expected = torch.softmax(logits[0].double(), dim=0).tolist()
    assert [p for _, p in actions] == pytest.approx(expected)
    raise_action = next(a for a, _ in actions if a.kind == ActionKind.RAISE)
    old = action_type_to_pokers_action(
        2, legacy, bet_size_multiplier=sizing.item(), strict=True
    )
    wager = current.players[current.seat].street_bet + current.legal_actions.call_amount
    assert raise_action.raise_to == wager + round(old.amount / legacy.chip_unit)
    for action, _ in actions:
        current.legal_actions.validate(action)
    with pytest.raises(TypeError):
        policy.choose_action(object())
    with pytest.raises(ValueError, match="player count"):
        policy.choose_action(view(5))


@pytest.mark.parametrize("players", [4, 5, 6])
@pytest.mark.parametrize("street", ["preflop", "flop", "turn"])
def test_frozen_models_cannot_distinguish_hidden_worlds(tmp_path, players, street):
    spec = checkpoint(tmp_path, players)
    model = FrozenNetwork(spec, tmp_path / f"fixture-{players}-7.pt")
    observer = 3 if street == "preflop" else 1
    board_count = {"preflop": 0, "flop": 3, "turn": 4}[street]
    protected = {(observer - 1) % players, (observer - 1) % players + players}
    protected.update(range(2 * players, 2 * players + board_count))
    hidden = [i for i in range(52) if i not in protected]
    changed = list(DECK)
    shuffled = [changed[i] for i in hidden]
    random.Random(91).shuffle(shuffled)
    for index, card in zip(hidden, shuffled):
        changed[index] = card
    observations = []
    for deck in (DECK, tuple(changed)):
        hand = Hand.from_deck(table(players), hand_id="paired", deck=deck)
        while hand.observe(observer).street != Street(street):
            legal = hand.observe(hand.actor).legal_actions
            hand = hand.apply(
                Action(
                    ActionKind.CHECK
                    if ActionKind.CHECK in legal.kinds
                    else ActionKind.CALL
                )
            )
        observations.append(hand.observe(observer))
    assert observations[0] == observations[1]
    assert model.policy(9).distribution(observations[0]) == model.policy(
        9
    ).distribution(observations[1])
    assert model.policy(9).choose_action(observations[0]) == model.policy(
        9
    ).choose_action(observations[1])


def test_checkpoint_hash_metadata_architecture_and_finite_values_are_checked(tmp_path):
    spec = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="hash"):
        FrozenNetwork(replace(spec, sha256="0" * 64), tmp_path / "fixture-4-7.pt")
    for extra, error in (
        ({"num_players": 6}, "Player count"),
        ({"agent_type": "opponent_modeling"}, "standard networks"),
        ({"min_bet_size": float("nan")}, "sizing bounds"),
    ):
        bad = checkpoint(tmp_path, **extra)
        with pytest.raises(ValueError, match=error):
            FrozenNetwork(bad, tmp_path / "fixture-4-7.pt")
    bad = checkpoint(tmp_path)
    path = tmp_path / "fixture-4-7.pt"
    data = torch.load(path, weights_only=True)
    data["strategy_net"]["action_head.bias"][0] = float("inf")
    torch.save(data, path)
    with pytest.raises(ValueError, match="finite"):
        FrozenNetwork(replace(bad, sha256=sha256(path.read_bytes()).hexdigest()), path)


def test_registry_rejects_incompatible_lineups_before_creating_output(tmp_path):
    spec = checkpoint(tmp_path)
    for scenario in (
        Scenario("five", (2000,) * 5),
        Scenario("session", (2000,) * 4, mode="session", hands_per_rotation=2),
    ):
        plan = Plan((scenario,), candidate=spec.name, models=(spec,), blocks=1)
        with pytest.raises(ValueError, match="fixed 4-player"):
            run(plan, tmp_path / "invalid")
        assert not (tmp_path / "invalid").exists()
    with pytest.raises(ValueError, match="shadow"):
        PolicyRegistry(
            Plan(
                (Scenario("four", (2000,) * 4),),
                models=(replace(spec, name="random"),),
                opponents=("random",),
            )
        )


def test_model_bundles_reproduce_after_original_files_are_removed(tmp_path):
    spec = checkpoint(tmp_path)
    plan = Plan(
        (Scenario("four", (2000,) * 4),),
        candidate=spec.name,
        baseline=spec.name,
        opponents=("pot_pressure",),
        models=(spec,),
        blocks=2,
    )
    assert Plan.from_dict(asdict(plan)) == plan
    result = run(plan, tmp_path / "one")
    assert result["status"] == "valid"
    (tmp_path / "fixture-4-7.pt").unlink()
    assert (tmp_path / f"one/models/{spec.sha256}.pt").exists()
    assert (
        reproduce(tmp_path / "one", tmp_path / "two")["outcomes_sha256"]
        == result["outcomes_sha256"]
    )
    manifest = json.loads((tmp_path / "one/manifest.json").read_text())
    assert manifest["policies"][spec.name]["weights_sha256"] == spec.sha256
    assert manifest["protocol"]["inference"]["threads"] == 1
    (tmp_path / f"one/models/{spec.sha256}.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        reproduce(tmp_path / "one", tmp_path / "three")
