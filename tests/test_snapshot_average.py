import json
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from random import getstate
from time import perf_counter

import numpy as np
import pytest
import torch

from src.solver.games import Action, new_game
from src.solver.neural.average import (
    OwnDecision,
    StrategyArchive,
    load_archive,
    record_iteration,
    save_archive,
)
from src.solver.neural.network import deterministic_cpu, new_network
from src.solver.neural.solver import Config, DeepCFR
from src.solver.tree import GameTree


@pytest.fixture(autouse=True)
def cpu():
    with deterministic_cpu():
        yield


def constant(values):
    model = new_network(4, 0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.layers[-1].bias.copy_(torch.tensor(values, dtype=torch.float32))
    return model


def archive_for(game):
    archive = StrategyArchive(game, 4)
    archive.append(1, (constant([1, 9, 1]), constant([7, 2, 3])))
    archive.append(2, (constant([9, 1, 9]), constant([1, 5, 2])))
    archive.append(3, (constant([-1, -2, 0]), constant([0, 0, 0])))
    return archive


def history_for(tree, index):
    return tuple(
        OwnDecision(
            tree.information_sets[prior], tree.information_sets[prior].actions[action]
        )
        for prior, action in tree.own_sequences[index]
    )


def table_for(tree, archive):
    table = np.zeros(tree.mask.shape)
    policies = [archive.policy(player) for player in (0, 1)]
    for index, info in enumerate(tree.information_sets):
        distribution = policies[info.player].distribution(
            info, history_for(tree, index)
        )
        table[index, : len(info.actions)] = [
            probability for _, probability in distribution
        ]
    tree.validate_policy(table)
    return table


def test_own_reach_changes_the_average_at_a_later_decision():
    archive = StrategyArchive("kuhn", 4)
    archive.append(1, (constant([1, 9, 1]), constant([1, 1, 1])))
    archive.append(2, (constant([9, 1, 9]), constant([1, 1, 1])))
    state = new_game("kuhn").deal(0).deal(2)
    earlier = OwnDecision(state.information_set(), Action.CHECK)
    later = state.play(Action.CHECK).play(Action.RAISE).information_set()
    result = dict(archive.policy(0).distribution(later, (earlier,)))
    expected = (0.9 * 0.1 + 2 * 0.1 * 0.9) / (0.9 + 2 * 0.1)
    assert result[Action.FOLD] == pytest.approx(expected)
    assert result[Action.FOLD] != pytest.approx((0.1 + 2 * 0.9) / 3)
    with pytest.raises(ValueError, match="complete own-decision"):
        archive.policy(0).distribution(later)


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_every_realization_probability_matches_the_mixture_of_whole_policies(game):
    tree = GameTree(game)
    archive = archive_for(game)
    average = table_for(tree, archive)
    components = []
    for pair in archive._snapshots:
        single = StrategyArchive(game, 4)
        single.append(1, pair)
        components.append(table_for(tree, single))
    for index, info in enumerate(tree.information_sets):
        for action in range(len(info.actions)):
            sequence = tree.own_sequences[index] + ((index, action),)
            expected = (
                sum(
                    weight
                    * np.prod([table[prior, choice] for prior, choice in sequence])
                    for weight, table in enumerate(components, 1)
                )
                / 6
            )
            realized = np.prod([average[prior, choice] for prior, choice in sequence])
            assert realized == pytest.approx(expected, abs=1e-12)


def test_zero_own_reach_has_a_legal_uniform_fallback():
    archive = StrategyArchive("kuhn", 4)
    archive.append(1, (constant([0, 0, 1]), constant([0, 0, 1])))
    first = new_game("kuhn").deal(0).deal(2)
    later = first.play(Action.CHECK).play(Action.RAISE).information_set()
    history = (OwnDecision(first.information_set(), Action.CHECK),)
    assert dict(archive.policy(0).distribution(later, history)) == {
        Action.FOLD: 0.5,
        Action.CALL: 0.5,
    }


def test_history_preserves_board_reveal_time_and_rejects_forged_prefixes():
    tree = GameTree("leduc")
    index = next(
        i
        for i, info in enumerate(tree.information_sets)
        if len(info.history) == 2 and info.player == 0 and tree.own_sequences[i]
    )
    info = tree.information_sets[index]
    history = history_for(tree, index)
    policy = archive_for("leduc").policy(0)
    policy.distribution(info, history)
    assert history[0].information.board is None and info.board is not None
    invalid = [
        replace(
            history[0], information=replace(history[0].information, board=info.board)
        ),
        replace(
            history[0],
            information=replace(history[0].information, card=(info.card + 1) % 3),
        ),
        replace(history[0], information=replace(history[0].information, player=1)),
        replace(history[0], action=Action.FOLD),
    ]
    for first in invalid:
        with pytest.raises(ValueError):
            policy.distribution(info, (first,) + history[1:])
    with pytest.raises(ValueError):
        policy.distribution(info, history + history)


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_recording_matches_played_average_without_changing_training(game):
    tree = GameTree(game)
    config = Config(
        hidden=4,
        traversals=8,
        advantage_steps=3,
        strategy_steps=2,
        batch_size=8,
        capacity=32,
        seed=101,
    )
    baseline, recorded = DeepCFR(tree, config), DeepCFR(tree, config)
    archive = StrategyArchive(game, 4)
    python_random, torch_random = getstate(), torch.get_rng_state().clone()
    for iteration in range(1, 5):
        previous_player1 = recorded.advantages[1]
        baseline.step()
        record_iteration(recorded, archive)
        assert archive.iterations == iteration
        assert table_for(tree, archive) == pytest.approx(
            recorded.played_average(), abs=1e-6
        )
        for saved, previous in zip(
            archive._snapshots[-1][1].parameters(), previous_player1.parameters()
        ):
            assert torch.equal(saved, previous)
        assert np.array_equal(baseline.current_policy(), recorded.current_policy())
        assert (
            baseline.traversal_random.getstate() == recorded.traversal_random.getstate()
        )
        assert baseline.fits == recorded.fits
        for a, b in zip(
            baseline.advantage_memories + [baseline.strategy_memory],
            recorded.advantage_memories + [recorded.strategy_memory],
        ):
            assert a.seen == b.seen and a.random.getstate() == b.random.getstate()
            for field in ["infos", "iterations", "targets"]:
                assert np.array_equal(
                    getattr(a, field)[: a.size], getattr(b, field)[: b.size]
                )
    assert getstate() == python_random and torch.equal(
        torch.get_rng_state(), torch_random
    )
    baseline.fit_strategy()
    recorded.fit_strategy()
    assert np.array_equal(baseline.average_policy(), recorded.average_policy())


def test_snapshot_policies_do_not_change_when_models_or_archive_advance():
    archive = StrategyArchive("kuhn", 4)
    original = constant([1, 9, 1])
    archive.append(1, (original, original))
    published = archive.policy(0)
    info = new_game("kuhn").deal(0).deal(2).information_set()
    before = published.distribution(info)
    with torch.no_grad():
        original.layers[-1].bias.copy_(torch.tensor([9.0, 1.0, 9.0]))
    archive.append(2, (original, original))
    assert published.distribution(info) == before
    assert archive.policy(0).distribution(info) != before


def test_failed_and_missing_iterations_cannot_publish_complete_history():
    tree = GameTree("kuhn")
    config = Config(
        hidden=4, traversals=1, advantage_steps=1, strategy_steps=1, batch_size=1
    )
    solver, archive = DeepCFR(tree, config), StrategyArchive("kuhn", 4)
    with pytest.raises(TimeoutError):
        record_iteration(solver, archive, deadline=perf_counter() - 1)
    assert archive.iterations == 0 and solver.failed
    with pytest.raises(RuntimeError):
        record_iteration(solver, archive)
    with pytest.raises(ValueError):
        archive.policy(0)
    with pytest.raises(ValueError):
        archive.append(2, tuple(solver.advantages))
    solver = DeepCFR(tree, config)
    solver.step()
    with pytest.raises(ValueError, match="complete iteration history"):
        record_iteration(solver, archive)


def test_hand_sampling_uses_iteration_weights_and_keeps_the_selected_component():
    archive = archive_for("kuhn")
    policy = archive.policy(0)
    first = new_game("kuhn").deal(0).deal(2)
    later = first.play(Action.CHECK).play(Action.RAISE)
    counts = Counter()
    for seed in range(600):
        hand = policy.sample_hand(seed)
        counts[hand.iteration] += 1
        index = hand.iteration
        hand.choose_action(first.information_set())
        hand.choose_action(later.information_set())
        assert hand.iteration == index
        replay = policy.sample_hand(seed)
        assert hand.distribution(later.information_set()) == replay.distribution(
            later.information_set()
        )
    for iteration, probability in [(1, 1 / 6), (2, 2 / 6), (3, 3 / 6)]:
        assert counts[iteration] / 600 == pytest.approx(probability, abs=0.05)


def test_policy_accepts_only_its_owners_information_and_ignores_hidden_worlds():
    archive = archive_for("leduc")
    policy = archive.policy(0)
    first, second = new_game("leduc").deal(0).deal(2), new_game("leduc").deal(0).deal(4)
    assert policy.distribution(first.information_set()) == policy.distribution(
        second.information_set()
    )
    for candidate in [policy, policy.sample_hand(13)]:
        with pytest.raises(TypeError):
            candidate.distribution(first)
        with pytest.raises(ValueError):
            candidate.distribution(first.play(Action.CHECK).information_set())
    with pytest.raises(ValueError):
        archive.policy(True)


def test_archive_round_trips_in_a_fresh_process(tmp_path):
    archive = archive_for("leduc")
    path = tmp_path / "average.pt"
    digest = save_archive(archive, path)
    restored = load_archive(path, digest)
    tree = GameTree("leduc")
    assert np.array_equal(table_for(tree, archive), table_for(tree, restored))
    assert restored.parameter_bytes == 6 * (4 * 4 + 53 * 4 + 3) * 4
    source = """
import json, sys
from src.solver.games import new_game
from src.solver.neural.average import load_archive
from src.solver.neural.network import deterministic_cpu
with deterministic_cpu():
    archive = load_archive(__import__('pathlib').Path(sys.argv[1]), sys.argv[2])
    info = new_game('leduc').deal(0).deal(2).information_set()
    print(json.dumps(archive.policy(0).distribution(info)))
"""
    output = subprocess.check_output(
        [sys.executable, "-c", source, str(path), digest], text=True
    )
    info = new_game("leduc").deal(0).deal(2).information_set()
    assert json.loads(output) == json.loads(
        json.dumps(archive.policy(0).distribution(info))
    )
    with pytest.raises(FileExistsError):
        save_archive(archive, path)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_archive(path, "0" * 64)


@pytest.mark.parametrize(
    "damage", ["count", "pair", "nan", "shape", "dtype", "weighting"]
)
def test_archive_rejects_malformed_payloads(tmp_path, damage):
    path = tmp_path / "average.pt"
    save_archive(archive_for("kuhn"), path)
    payload = torch.load(path, weights_only=True)
    weights = payload["snapshots"][0][0]
    key = next(iter(weights))
    if damage == "count":
        payload["iterations"] += 1
    elif damage == "pair":
        payload["snapshots"][0].pop()
    elif damage == "nan":
        weights[key].flatten()[0] = float("nan")
    elif damage == "shape":
        weights[key] = weights[key][:-1]
    elif damage == "dtype":
        weights[key] = weights[key].double()
    else:
        payload["weighting"] = "uniform"
    stream = BytesIO()
    torch.save(payload, stream)
    path.write_bytes(stream.getvalue())
    with pytest.raises(ValueError):
        load_archive(path, sha256(stream.getvalue()).hexdigest())


def test_recorder_rejects_a_different_solver_or_an_inference_only_archive(tmp_path):
    config = Config(
        hidden=4, traversals=2, advantage_steps=1, strategy_steps=1, batch_size=2
    )
    first, second = (DeepCFR(GameTree("kuhn"), config) for _ in range(2))
    archive = StrategyArchive("kuhn", 4)
    record_iteration(first, archive)
    second.step()
    with pytest.raises(ValueError, match="original live solver"):
        record_iteration(second, archive)
    path = tmp_path / "average.pt"
    restored = load_archive(path, save_archive(archive, path))
    with pytest.raises(ValueError, match="cannot resume training"):
        record_iteration(first, restored)
    assert first.iterations == second.iterations == archive.iterations == 1


def test_failure_after_the_first_player_update_leaves_no_partial_pair(monkeypatch):
    config = Config(
        hidden=4, traversals=2, advantage_steps=1, strategy_steps=1, batch_size=2
    )
    solver = DeepCFR(GameTree("kuhn"), config)
    archive = StrategyArchive("kuhn", 4)
    fit = solver._fit
    original = solver.advantages[0]

    def fail_second_player(memory, iteration, player, strategy, deadline):
        if player == 1:
            raise TimeoutError("second player exceeded deadline")
        return fit(memory, iteration, player, strategy, deadline)

    monkeypatch.setattr(solver, "_fit", fail_second_player)
    with pytest.raises(TimeoutError):
        record_iteration(solver, archive)
    assert solver.advantages[0] is not original
    assert solver.failed and archive.iterations == 0
