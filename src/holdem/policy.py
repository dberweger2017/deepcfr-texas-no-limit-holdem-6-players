"""Isolated current-policy snapshots for a single collection phase."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256

import torch

from src.holdem.actions import SCHEMA as ACTION_SCHEMA
from src.holdem.actions import BetCandidates
from src.holdem.betting import BettingNetwork
from src.holdem.encoding import SCHEMA as DECISION_SCHEMA

FORMAT = "holdem-current-profile-v1"


def _fingerprint(models: tuple[BettingNetwork | None, ...]) -> str:
    digest = sha256(f"{FORMAT}/{ACTION_SCHEMA}/{DECISION_SCHEMA}".encode())
    for seat, model in enumerate(models):
        digest.update(f"/seat-{seat}/".encode())
        if model is None:
            digest.update(b"uniform-candidates")
            continue
        digest.update(f"width-{model.encoder.width}".encode())
        for name, value in sorted(model.state_dict().items()):
            digest.update(f"{name}/{tuple(value.shape)}/{value.dtype}/".encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class FrozenProfile:
    """One independent model per physical seat; None explicitly means uniform."""

    _models: tuple[BettingNetwork | None, ...]
    fingerprint: str

    def __init__(self, models: Sequence[BettingNetwork | None]):
        if not 2 <= len(models) <= 6 or any(
            model is not None and type(model) is not BettingNetwork for model in models
        ):
            raise ValueError(
                "Provide two to six BettingNetwork models or uniform slots"
            )
        copies = tuple(
            None
            if model is None
            else deepcopy(model).cpu().float().eval().requires_grad_(False)
            for model in models
        )
        if any(
            not torch.isfinite(p).all()
            for model in copies
            if model is not None
            for p in model.parameters()
        ):
            raise ValueError("Cannot freeze non-finite model parameters")
        object.__setattr__(self, "_models", copies)
        object.__setattr__(self, "fingerprint", _fingerprint(copies))

    @property
    def capacity(self) -> int:
        return len(self._models)

    def assert_unchanged(self) -> None:
        if (
            any(
                module.training
                for model in self._models
                if model is not None
                for module in model.modules()
            )
            or _fingerprint(self._models) != self.fingerprint
        ):
            raise RuntimeError("The collection policy profile changed")

    def _model(self, candidates: BetCandidates) -> BettingNetwork | None:
        if not isinstance(candidates, BetCandidates):
            raise TypeError("A current policy accepts public bet candidates only")
        view = candidates.decision.source
        if view.capacity != self.capacity or view.actor != view.seat or view.finished:
            raise ValueError("Decision does not belong to this profile's live table")
        return self._models[view.seat_numbers[view.seat]]

    def distribution(self, candidates: BetCandidates) -> tuple[float, ...]:
        model = self._model(candidates)
        if model is None:
            return (1 / len(candidates.actions),) * len(candidates.actions)
        with torch.inference_mode():
            return tuple(model([candidates])[0].probabilities().tolist())

    def action_values(self, candidates: BetCandidates) -> tuple[float, ...]:
        """Frozen own-seat value predictions in BB, never another role's payoff."""
        model = self._model(candidates)
        if model is None:
            return (0.0,) * len(candidates.actions)
        with torch.inference_mode():
            values = model([candidates])[0].values
        if (
            values.shape != (len(candidates.actions),)
            or not torch.isfinite(values).all()
        ):
            raise FloatingPointError("Invalid frozen action values")
        return tuple(values.tolist())
