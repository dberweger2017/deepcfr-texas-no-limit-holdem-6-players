"""Benchmark-only uniform control over the unchanged HU20 restricted menu."""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar
import json

from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, choices
from src.blueprint.solver import HU20_UNCAPPED_GAME

DEFINITION = {
    "adapter": "uniform-restricted-v1",
    "game": HU20_UNCAPPED_GAME,
    "schema": HU20_UNCAPPED_SCHEMA,
    "menu": "choices(view, raise_cap=None, free_fold=False)",
    "sampling": "uniform weights; persisted per-session Python Random.choices",
    "benchmarkOnly": True,
}
DEFINITION_SHA256 = sha256(json.dumps(DEFINITION, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class _Identity:
    sha256: str = DEFINITION_SHA256


@dataclass(frozen=True, slots=True)
class UniformRestrictedPolicy:
    # The service owns and persists sampling randomness; this object has no RNG
    # or cross-session state. The hash identifies the definition, not a model file.
    spec: ClassVar[_Identity] = _Identity()
    name: ClassVar[str] = "Uniform restricted random"
    adapter_id: ClassVar[str] = "uniform-restricted-v1"
    format_id: ClassVar[str] = "builtin-uniform-restricted-v1"
    benchmark_only: ClassVar[bool] = True
    game: ClassVar[str] = HU20_UNCAPPED_GAME
    abstraction: ClassVar[str] = HU20_UNCAPPED_SCHEMA
    players: ClassVar[int] = 2

    @property
    def description(self):
        return {"strategy": "uniform-restricted"}

    def distribution(self, view):
        menu = choices(view, raise_cap=None, free_fold=False)
        if not menu:
            raise ValueError("Uniform control requires an acting player")
        # None means lookup telemetry is inapplicable, not blueprint fallback.
        return menu, (1 / len(menu),) * len(menu), None
