"""Frozen standard-network inference through the current public observation adapter."""

import pickle
from contextlib import contextmanager
from decimal import ROUND_HALF_UP, Decimal
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from random import Random

import numpy as np
import torch

from src.arena.catalog import Checkpoint
from src.arena.historical import PokerNetwork, encode_observation
from src.game.observation import Observation
from src.game.types import Action, ActionKind


@contextmanager
def inference_runtime():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


def _metadata(checkpoint, key):
    nested = checkpoint.get("metadata", {})
    if not isinstance(nested, dict):
        raise TypeError("Invalid checkpoint metadata")
    outer, inner = checkpoint.get(key), nested.get(key)
    if outer is not None and inner is not None and outer != inner:
        raise ValueError(f"Conflicting checkpoint metadata: {key}")
    return outer if outer is not None else inner


class FrozenNetwork:
    def __init__(self, spec: Checkpoint, path: Path):
        self.spec = spec
        self.data = path.read_bytes()
        if sha256(self.data).hexdigest() != spec.sha256:
            raise ValueError(f"Checkpoint hash mismatch: {spec.name}")
        try:
            checkpoint = torch.load(
                BytesIO(self.data), map_location="cpu", weights_only=True
            )
        except (
            RuntimeError,
            ValueError,
            TypeError,
            pickle.UnpicklingError,
            EOFError,
        ) as exc:
            raise ValueError(f"Unsupported checkpoint: {spec.name}") from exc
        if not isinstance(checkpoint, dict):
            raise TypeError("Expected a standard checkpoint dictionary")
        if (
            _metadata(checkpoint, "agent_type") not in (None, "standard")
            or "history_encoder" in checkpoint
            or "opponent_model" in checkpoint
        ):
            raise ValueError("This adapter supports standard networks only")
        if _metadata(checkpoint, "schema_version") not in (None, 1):
            raise ValueError("Unsupported checkpoint metadata version")
        state = checkpoint.get("strategy_net")
        if not isinstance(state, dict) or "base.0.weight" not in state:
            raise ValueError("Checkpoint has no standard strategy network")
        if any(
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
            for tensor in state.values()
        ):
            raise ValueError("Strategy weights must be finite float32 tensors")
        first = state["base.0.weight"]
        if first.ndim != 2:
            raise ValueError("Invalid input layer")
        hidden, inputs = first.shape
        players, remainder = divmod(inputs - 120, 6)
        if remainder or not 2 <= players <= 10 or not 2 <= hidden <= 4096:
            raise ValueError("Unsupported standard network dimensions")
        recorded_players = _metadata(checkpoint, "num_players")
        if recorded_players is not None and (
            type(recorded_players) is not int or recorded_players != players
        ):
            raise ValueError("Player count disagrees with the saved network")
        self.players = players
        self.min_bet = checkpoint.get("min_bet_size", 0.1)
        self.max_bet = checkpoint.get("max_bet_size", 3.0)
        if (
            any(
                type(v) not in (int, float) or not np.isfinite(v)
                for v in (self.min_bet, self.max_bet)
            )
            or not 0 < self.min_bet <= self.max_bet
        ):
            raise ValueError("Invalid checkpoint sizing bounds")
        # Loading must not consume a trainer's global Torch random stream.
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            self.network = PokerNetwork(inputs, hidden).float()
        try:
            self.network.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError("Unsupported strategy architecture") from exc
        self.network.eval().requires_grad_(False)
        self.description = {
            "kind": spec.format,
            "weights_sha256": spec.sha256,
            "num_players": players,
            "input_size": inputs,
            "hidden_size": hidden,
            "iteration": _metadata(checkpoint, "iteration"),
            "training_seed": _metadata(checkpoint, "training_seed"),
            "min_bet_size": self.min_bet,
            "max_bet_size": self.max_bet,
            "encoding": "legacy-absolute-seat-v1",
            "sampling": "masked-softmax-private-python-rng-v1",
        }
        for key in ("iteration", "training_seed"):
            value = self.description[key]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Invalid checkpoint {key}")

    def policy(self, seed: int):
        return FrozenPolicy(self, seed)


class FrozenPolicy:
    def __init__(self, model: FrozenNetwork, seed: int):
        self._model = model
        self._random = Random(seed)

    def distribution(self, view: Observation) -> tuple[tuple[Action, float], ...]:
        if not isinstance(view, Observation):
            raise TypeError("A frozen policy requires a player observation")
        if view.finished or view.actor != view.seat or len(view.hole_cards) != 2:
            raise ValueError("Policy needs its own current decision")
        if len(view.players) != self._model.players:
            raise ValueError("Checkpoint does not support this player count")
        encoded = encode_observation(view)
        if not np.isfinite(encoded).all():
            raise ValueError("Non-finite policy input")
        with torch.inference_mode():
            logits, sizing = self._model.network(torch.from_numpy(encoded).unsqueeze(0))
        if not torch.isfinite(logits).all() or not torch.isfinite(sizing).all():
            raise ValueError("Non-finite policy output")
        choices, indices = [], []
        legal = view.legal_actions
        if ActionKind.FOLD in legal.kinds:
            choices.append(Action(ActionKind.FOLD))
            indices.append(0)
        if ActionKind.CHECK in legal.kinds or ActionKind.CALL in legal.kinds:
            choices.append(
                Action(
                    ActionKind.CHECK
                    if ActionKind.CHECK in legal.kinds
                    else ActionKind.CALL
                )
            )
            indices.append(1)
        if ActionKind.RAISE in legal.kinds:
            multiplier = min(
                self._model.max_bet, max(self._model.min_bet, sizing.item())
            )
            unit = Decimal(view.chip_unit)
            # Preserve the old additional-raise convention and its one-unit pot floor.
            desired = Decimal(str(max(1.0, view.pot * float(unit)) * multiplier))
            additional = int((desired / unit).to_integral_value(rounding=ROUND_HALF_UP))
            wager = view.players[view.seat].street_bet + legal.call_amount
            target = min(
                legal.max_raise_to, max(legal.min_raise_to, wager + additional)
            )
            choices.append(Action(ActionKind.RAISE, target))
            indices.append(2)
        if not indices:
            raise ValueError("No legal model action")
        probabilities = torch.softmax(logits[0, indices].double(), dim=0).tolist()
        return tuple(zip(choices, probabilities, strict=True))

    def choose_action(self, view: Observation) -> Action:
        actions, weights = zip(*self.distribution(view), strict=True)
        return self._random.choices(actions, weights)[0]
