"""Average retained advantage policies using only the player's own decision history."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from random import Random
from weakref import ReferenceType, ref

import numpy as np
import torch

from src.solver.games import Action, InformationSet
from src.solver.neural.checkpoint import atomic_write
from src.solver.neural.encoding import ACTION_SLOT, encode, legal_mask
from src.solver.neural.network import (
    Network,
    new_network,
    predict,
    regret_matching,
    stream_seed,
)
from src.solver.neural.solver import DeepCFR

FORMAT = "small-game-snapshot-average-v1"


@dataclass(frozen=True, slots=True)
class OwnDecision:
    information: InformationSet
    action: Action


def _check_information(info: InformationSet, game: str, player: int) -> None:
    if not isinstance(info, InformationSet):
        raise TypeError("A snapshot policy accepts player information sets only")
    if info.game != game or info.player != player:
        raise ValueError("Information set does not belong to this policy")
    if (
        not isinstance(info.history, tuple)
        or not 1 <= len(info.history) <= (1 if game == "kuhn" else 2)
        or any(
            not isinstance(street, tuple) or len(street) > 4 for street in info.history
        )
        or len(info.history[-1]) % 2 != player
        or type(info.card) is not int
        or not 0 <= info.card <= 2
        or (len(info.history) == 1 and info.board is not None)
        or (
            len(info.history) == 2
            and (type(info.board) is not int or not 0 <= info.board <= 2)
        )
        or not info.actions
        or len(set(info.actions)) != len(info.actions)
        or any(not isinstance(action, Action) for action in info.actions)
        or any(
            not isinstance(action, Action)
            for street in info.history
            for action in street
        )
    ):
        raise ValueError("Invalid small-game decision observation")


def _decision_path(
    info: InformationSet, history: tuple[OwnDecision, ...]
) -> list[InformationSet]:
    expected = [
        (street, position, action)
        for street, actions in enumerate(info.history)
        for position, action in enumerate(actions)
        if position % 2 == info.player
    ]
    if len(history) != len(expected):
        raise ValueError("The complete own-decision history is required")
    path = []
    for decision, (street, position, action) in zip(history, expected):
        if not isinstance(decision, OwnDecision):
            raise TypeError("History must contain own decision records")
        previous = decision.information
        _check_information(previous, info.game, info.player)
        prefix = info.history[:street] + (info.history[street][:position],)
        if (
            previous.history != prefix
            or previous.card != info.card
            or previous.board != (None if street == 0 else info.board)
            or decision.action != action
            or action not in previous.actions
        ):
            raise ValueError(
                "Own-decision history does not match the current observation"
            )
        path.append(previous)
    return path + [info]


def _encode_path(path: list[InformationSet]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.stack([encode(info) for info in path])),
        torch.from_numpy(np.stack([legal_mask(info) for info in path])),
    )


def _probabilities(
    model: Network, features: torch.Tensor, masks: torch.Tensor
) -> np.ndarray:
    return regret_matching(
        predict(model, features, masks, strategy=False), masks.numpy()
    )


def _actions(
    info: InformationSet, probabilities: np.ndarray
) -> tuple[tuple[Action, float], ...]:
    return tuple(
        (action, float(probabilities[ACTION_SLOT[action]])) for action in info.actions
    )


class AveragePolicy:
    def __init__(self, game: str, player: int, models: tuple[Network, ...]):
        if (
            game not in {"kuhn", "leduc"}
            or type(player) is not int
            or player not in (0, 1)
            or not models
        ):
            raise ValueError("An average policy requires a game, player and snapshots")
        self.game, self.player = game, player
        self._models = tuple(models)

    def distribution(
        self, info: InformationSet, history: tuple[OwnDecision, ...] = ()
    ) -> tuple[tuple[Action, float], ...]:
        _check_information(info, self.game, self.player)
        features, masks = _encode_path(_decision_path(info, history))
        weights = np.log(np.arange(1, len(self._models) + 1, dtype=np.float64))
        current = []
        for index, model in enumerate(self._models):
            probabilities = _probabilities(model, features, masks)
            # Only our earlier actions update the posterior over iteration policies.
            with np.errstate(divide="ignore"):
                weights[index] += sum(
                    np.log(probabilities[row, ACTION_SLOT[decision.action]])
                    for row, decision in enumerate(history)
                )
            current.append(probabilities[-1])
        if not np.isfinite(weights).any():
            mask = legal_mask(info)
            return _actions(info, mask / mask.sum())
        weights = np.exp(weights - weights.max())
        return _actions(info, weights @ np.stack(current) / weights.sum())

    def sample_hand(self, seed: int) -> "SampledHandPolicy":
        choice = Random(stream_seed(seed, "snapshot-choice", player=self.player))
        index = choice.choices(
            range(len(self._models)), weights=range(1, len(self._models) + 1)
        )[0]
        return SampledHandPolicy(
            self.game,
            self.player,
            self._models[index],
            index + 1,
            stream_seed(seed, "snapshot-actions", player=self.player),
        )


class SampledHandPolicy:
    """A single mixture component, selected once and retained for the whole hand."""

    def __init__(
        self, game: str, player: int, model: Network, iteration: int, seed: int
    ):
        self.game, self.player, self.iteration = game, player, iteration
        self._model = model
        self._random = Random(seed)

    def distribution(self, info: InformationSet) -> tuple[tuple[Action, float], ...]:
        _check_information(info, self.game, self.player)
        return _actions(info, _probabilities(self._model, *_encode_path([info]))[0])

    def choose_action(self, info: InformationSet) -> Action:
        actions, probabilities = zip(*self.distribution(info))
        return self._random.choices(actions, weights=probabilities)[0]


def _copy_network(weights: dict, hidden: int) -> Network:
    if not isinstance(weights, dict) or any(
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
        for value in weights.values()
    ):
        raise ValueError("Snapshot weights must be finite CPU float32 tensors")
    model = new_network(hidden, 0)
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as error:
        raise ValueError("Snapshot network shape differs from the archive") from error
    return model.eval().requires_grad_(False)


class StrategyArchive:
    def __init__(self, game: str, hidden: int):
        if (
            game not in {"kuhn", "leduc"}
            or type(hidden) is not int
            or not 1 <= hidden <= 512
        ):
            raise ValueError("Unsupported snapshot game or network width")
        self.game, self.hidden = game, hidden
        self._snapshots: list[tuple[Network, Network]] = []
        self._source: ReferenceType[DeepCFR] | None = None

    @property
    def iterations(self) -> int:
        return len(self._snapshots)

    @property
    def parameter_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for pair in self._snapshots
            for model in pair
            for value in model.parameters()
        )

    def append(self, iteration: int, models: tuple[Network, Network]) -> None:
        if (
            type(iteration) is not int
            or iteration != self.iterations + 1
            or len(models) != 2
        ):
            raise ValueError(
                "Archive requires one complete pair per consecutive iteration"
            )
        copied = tuple(
            _copy_network(model.state_dict(), self.hidden) for model in models
        )
        self._snapshots.append(copied)

    def policy(self, player: int) -> AveragePolicy:
        if type(player) is not int or player not in (0, 1) or not self.iterations:
            raise ValueError("A policy requires a valid player and a nonempty archive")
        return AveragePolicy(
            self.game, player, tuple(pair[player] for pair in self._snapshots)
        )


def record_iteration(
    solver: DeepCFR, archive: StrategyArchive, deadline: float = float("inf")
) -> None:
    if (
        archive.game != solver.tree.game
        or archive.hidden != solver.config.hidden
        or archive.iterations != solver.iterations
    ):
        raise ValueError(
            "Archive must match the solver and contain its complete iteration history"
        )
    if (archive._source is None and archive.iterations) or (
        archive._source is not None and archive._source() is not solver
    ):
        raise ValueError(
            "Recording requires the original live solver; inference archives cannot resume training"
        )
    archive._source = ref(solver)
    # Player 1 contributes before its update; player 0 contributes after its update.
    previous_player1 = solver.advantages[1]
    solver.step(deadline)
    try:
        archive.append(solver.iterations, (solver.advantages[0], previous_player1))
    except BaseException:
        solver.failed = True
        raise


def archive_state(archive: StrategyArchive) -> dict:
    if not archive.iterations:
        raise ValueError("Cannot export an empty strategy archive")
    return {
        "format": FORMAT,
        "encoding": "public-small-game-48-v1",
        "weighting": "linear-own-reach",
        "alignment": "opponent-collection-v1",
        "game": archive.game,
        "hidden": archive.hidden,
        "iterations": archive.iterations,
        "snapshots": [
            [model.state_dict() for model in pair] for pair in archive._snapshots
        ],
    }


def save_archive(archive: StrategyArchive, path: Path) -> str:
    data = BytesIO()
    torch.save(archive_state(archive), data)
    return atomic_write(path, data.getvalue())


def load_archive(path: Path, digest: str) -> StrategyArchive:
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError("Strategy archive hash mismatch")
    payload = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    return restore_archive(payload)


def restore_archive(payload: dict) -> StrategyArchive:
    if (
        not isinstance(payload, dict)
        or payload.get("format") != FORMAT
        or payload.get("encoding") != "public-small-game-48-v1"
        or payload.get("weighting") != "linear-own-reach"
        or payload.get("alignment") != "opponent-collection-v1"
    ):
        raise ValueError("Unsupported strategy archive")
    archive = StrategyArchive(payload["game"], payload["hidden"])
    snapshots = payload["snapshots"]
    if (
        type(payload["iterations"]) is not int
        or payload["iterations"] < 1
        or not isinstance(snapshots, list)
        or len(snapshots) != payload["iterations"]
    ):
        raise ValueError("Archive snapshot count differs from its iteration count")
    for pair in snapshots:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Archive must contain both players at every iteration")
        archive._snapshots.append(
            tuple(_copy_network(weights, archive.hidden) for weights in pair)
        )
    return archive


def tabulate(archive: StrategyArchive, tree) -> np.ndarray:
    """Evaluate the exported mixture in batches over public information sets."""
    if archive.game != tree.game or not archive.iterations:
        raise ValueError("Archive and evaluation game must match")
    features = torch.from_numpy(
        np.stack([encode(info) for info in tree.information_sets])
    )
    masks = torch.from_numpy(
        np.stack([legal_mask(info) for info in tree.information_sets])
    )
    totals = np.zeros(len(tree.information_sets))
    summed = np.zeros(tree.mask.shape)
    for iteration, pair in enumerate(archive._snapshots, 1):
        probabilities = np.zeros(masks.shape)
        for player, model in enumerate(pair):
            rows = tree.owners == player
            probabilities[rows] = _probabilities(model, features[rows], masks[rows])
        for index, info in enumerate(tree.information_sets):
            own_reach = 1.0
            for prior, action in tree.own_sequences[index]:
                slot = ACTION_SLOT[tree.information_sets[prior].actions[action]]
                own_reach *= probabilities[prior, slot]
            weight = iteration * own_reach
            totals[index] += weight
            summed[index, : len(info.actions)] += (
                weight * probabilities[index, [ACTION_SLOT[a] for a in info.actions]]
            )
    result = tree.mask / tree.mask.sum(axis=1, keepdims=True)
    reached = totals > 0
    result[reached] = summed[reached] / totals[reached, None]
    tree.validate_policy(result)
    return result
