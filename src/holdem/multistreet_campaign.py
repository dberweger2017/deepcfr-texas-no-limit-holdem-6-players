"""Campaign planning, calibration, and resumable reference caching.

This module deliberately sits beside the bounded multi-street pilot.  The
pilot's artifacts and command line remain reproducible; this campaign adds a
larger, cost-controlled orchestration layer with a frozen calibration phase.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from src.arena.schedule import digest
from src.game.hand import card_name
from src.holdem.actions import bet_candidates
from src.holdem.multistreet_reference import (
    MultiStreetContext,
    enumerate_reference,
    split_specs,
)
from src.holdem.river_reference import ReferenceProfile
from src.holdem.targets import CandidateTargets


STREETS = ("flop", "turn", "river")
SITUATIONS = ("open", "facing")
PROFILES = ("uniform", "increasing")
NESTED_WORLD_COUNTS = (8, 16, 32, 64, 128)


def _atomic_bytes(path: Path, payload: bytes) -> str:
    """Publish one cache object atomically and return its content hash."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _atomic_bytes(path, payload)


def _serial_target(target: CandidateTargets) -> dict[str, Any]:
    return {
        "policy": list(target.policy),
        "values_bb": list(target.values_bb),
        "regrets_bb": list(target.regrets_bb),
        "actions": [repr(action) for action in target.candidates.actions],
    }


def campaign_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Expand compact family/holding declarations into 1,152 contexts.

    A family owns all of its descendants.  The resulting context order is
    stable and is the sole source of production cache keys and stream prefixes.
    """

    families = plan.get("families")
    holdings = plan.get("holdings")
    if not isinstance(families, list) or (holdings is not None and not isinstance(holdings, list)):
        # Already-expanded plans are useful in tests and for archived reruns.
        if "contexts" in plan:
            return dict(plan)
        raise ValueError("campaign plan needs families and holdings")
    if len(families) != 48 or (holdings is not None and len(holdings) != 4):
        raise ValueError("campaign requires 48 families and four holdings per family")
    contexts = []
    for family_index, family in enumerate(families):
        split = family.get("split")
        board = tuple(family.get("flop", ()))
        continuation = tuple(family.get("continuation", ()))
        if split not in {"train", "tuning", "validation", "test"}:
            raise ValueError("unknown campaign split")
        if len(board) != 3 or len(continuation) != 2:
            raise ValueError("each family needs a flop and two continuation cards")
        if len(set(board + continuation)) != 5:
            raise ValueError("family board cards must be distinct")
        family_holdings = family.get("holdings", holdings)
        if not isinstance(family_holdings, list) or len(family_holdings) != 4:
            raise ValueError("each campaign family needs four holdings")
        for holding_index, holding in enumerate(family_holdings):
            holding = tuple(holding)
            if len(holding) != 2 or len(set(holding)) != 2:
                raise ValueError("holdings must be distinct two-card tuples")
            if set(holding) & set(board + continuation):
                raise ValueError("holding overlaps a family board")
            for street, visible_count in zip(STREETS, (3, 4, 5), strict=True):
                for facing in SITUATIONS:
                    contexts.append(
                        {
                            "split": split,
                            "family": family_index,
                            "holding": list(holding),
                            "street": street,
                            "board": list(board + continuation[: visible_count - 3]),
                            "facing": facing == "facing",
                        }
                    )
    counts = {split: sum(row["split"] == split for row in contexts) for split in ("train", "tuning", "validation", "test")}
    if counts != {"train": 576, "tuning": 192, "validation": 192, "test": 192}:
        raise ValueError(f"unexpected campaign split counts: {counts}")
    expanded = dict(plan)
    expanded["contexts"] = contexts
    expanded["format"] = "holdem-multistreet-campaign-v1-expanded"
    return expanded


def _validate_nested(values: Sequence[Sequence[float]], maximum: int) -> None:
    if len(values) < maximum:
        raise ValueError("world stream is shorter than the declared maximum")
    width = len(values[0]) if values else 0
    if width == 0 or any(len(row) != width for row in values[:maximum]):
        raise ValueError("world action arrays have inconsistent widths")
    if not np.isfinite(np.asarray(values[:maximum], dtype=float)).all():
        raise ValueError("world action values must be finite")


def paired_action_difference_se(values: Sequence[Sequence[float]], n: int) -> float:
    """Return the worst paired SE over all unordered action pairs.

    Every column comes from the same world row.  This is deliberately a
    within-world action contrast, rather than a marginal standard error for a
    single action or a contrast between two continuation profiles.
    """

    if n < 2:
        raise ValueError("paired SE needs at least two worlds")
    values = np.asarray(values[:n], dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("paired action arrays need at least two actions")
    differences = values[:, :, None] - values[:, None, :]
    standard_errors = differences.std(axis=0, ddof=1) / np.sqrt(n)
    return float(np.max(np.abs(standard_errors)))


@dataclass(frozen=True)
class CalibrationDecision:
    n: int
    status: str
    aggregate_se_bb: float
    strata: tuple[str, ...]


def calibrate_world_counts(
    strata: Mapping[str, Sequence[Sequence[Sequence[float]]]],
    *,
    precision_bb: float = 0.10,
    minimum_n: int = 16,
    maximum_n: int = 128,
    multiplier: float = 1.0,
) -> dict[str, Any]:
    """Freeze one nested world count per stratum before fitting.

    Contexts are summarized by the worst unordered action-pair SE, then the
    90th percentile of those context summaries is compared to the absolute BB
    threshold.  Marginal SEs and between-profile differences are never used.
    The eight-world trace is retained, while freezing starts at ``minimum_n``.
    """

    trace_counts = tuple(n for n in NESTED_WORLD_COUNTS if n <= maximum_n)
    allowed = tuple(n for n in trace_counts if n >= minimum_n)
    if not allowed or allowed[-1] != maximum_n or 8 not in trace_counts:
        raise ValueError("minimum/maximum counts must select nested standard counts")
    if precision_bb <= 0 or not isfinite(precision_bb):
        raise ValueError("precision_bb must be positive and finite")
    if set(strata) != {f"{street}:{situation}" for street in STREETS for situation in SITUATIONS}:
        raise ValueError("calibration must cover six street/situation strata")
    decisions = {}
    for name in sorted(strata):
        contexts = strata[name]
        if not contexts:
            raise ValueError(f"calibration stratum {name} is empty")
        for values in contexts:
            # A caller may provide the two frozen-profile matrices as a pair;
            # combine them using the declared 1:2 continuation schedule before
            # forming within-world action contrasts.  No profile difference is
            # ever used as a precision estimate.
            if len(values) == 2 and np.asarray(values[0]).ndim == 2 and np.asarray(values[1]).ndim == 2:
                values = [
                    [(a + 2 * b) / 3 for a, b in zip(row_a, row_b, strict=True)]
                    for row_a, row_b in zip(values[0], values[1], strict=True)
                ]
            _validate_nested(values, maximum_n)
        traces = []
        for n in trace_counts:
            normalized = []
            for values in contexts:
                if len(values) == 2 and np.asarray(values[0]).ndim == 2 and np.asarray(values[1]).ndim == 2:
                    values = [
                        [(a + 2 * b) / 3 for a, b in zip(row_a, row_b, strict=True)]
                        for row_a, row_b in zip(values[0], values[1], strict=True)
                    ]
                normalized.append(values)
            context_se = [paired_action_difference_se(values, n) for values in normalized]
            aggregate = float(np.percentile(context_se, 90))
            bound = multiplier * aggregate
            traces.append({
                "n": n,
                "context_worst_pair_se_bb": context_se,
                "context_worst_pair_se_p90_bb": aggregate,
                "bound_bb": bound,
            })
        chosen = next(
            (
                n
                for n in allowed
                if all(row["bound_bb"] <= precision_bb for row in traces if row["n"] >= n)
            ),
            None,
        )
        n = chosen or maximum_n
        final = next(row for row in traces if row["n"] == n)
        decisions[name] = {
            "n": n,
            "status": "resolved" if chosen is not None else "unresolved_at_maximum",
            "aggregate_paired_se_bb": final["context_worst_pair_se_p90_bb"],
            "precision_bound_bb": final["bound_bb"],
            "trace": traces,
        }
    return {
        "precision_bb": precision_bb,
        "minimum_n": minimum_n,
        "maximum_n": maximum_n,
        "multiplier": multiplier,
        "aggregation": "p90 across contexts of each context's worst unordered action-pair paired SE; equal strata in report",
        "decisions": decisions,
        "unresolved": [name for name, row in decisions.items() if row["status"] != "resolved"],
    }


class ReferenceCache:
    """Per-context/profile cache with provenance and atomic publication."""

    def __init__(
        self,
        root: Path,
        *,
        source_sha256: str,
        plan_sha256: str,
        stream_namespace: str = "production",
        stream_seed: int | None = None,
        selected_n: Any = None,
    ):
        self.root = Path(root)
        self.source_sha256 = source_sha256
        self.plan_sha256 = plan_sha256
        self.stream_namespace = stream_namespace
        self.stream_seed = stream_seed
        self.selected_n = selected_n
        self.stream_key = sha256(
            f"{stream_namespace}|{stream_seed}|{plan_sha256}|{selected_n}".encode()
        ).hexdigest()

    @staticmethod
    def _world_fingerprint(context: MultiStreetContext) -> dict[str, Any]:
        observation = repr(context.worlds[0].observe(context.hero_seat))
        return {
            "visible_observation_sha256": sha256(observation.encode()).hexdigest(),
            "assignments_sha256": digest(context.assignments),
            # Decks and assignments are privileged cache provenance. They are
            # never passed to model inputs or held-out metric code.
            "world_decks_sha256": digest(
                [
                    tuple(card_name(card) for card in world._state.deck)
                    for world in context.worlds
                ]
            ),
            "world_seeds": list(context.world_seeds),
            "worlds": len(context.worlds),
        }

    def path(self, context: MultiStreetContext, profile: str) -> Path:
        key = digest({"name": context.name, "split": context.split, "street": context.street, "facing": context.facing, "profile": profile, "plan": self.plan_sha256, "stream_namespace": self.stream_namespace, "stream_seed": self.stream_seed, "stream_key": self.stream_key, "selected_n": self.selected_n, "worlds": len(context.worlds)})
        return self.root / f"{key}.json"

    def load(self, context: MultiStreetContext, profile: str) -> dict[str, Any] | None:
        path = self.path(context, profile)
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value.get("source_sha256") != self.source_sha256 or value.get("plan_sha256") != self.plan_sha256:
            raise ValueError(f"cache provenance mismatch: {path}")
        if value.get("profile") != profile or value.get("context") != context.name:
            raise ValueError(f"cache identity mismatch: {path}")
        if value.get("stream_namespace") != self.stream_namespace or value.get("stream_seed") != self.stream_seed or value.get("stream_key") != self.stream_key or value.get("selected_n") != self.selected_n:
            raise ValueError(f"cache stream mismatch: {path}")
        if value.get("world_fingerprint") != self._world_fingerprint(context):
            raise ValueError(f"cache world fingerprint mismatch: {path}")
        if value.get("worlds") != len(context.worlds) or len(value.get("world_action_values_bb", ())) != len(context.worlds):
            raise ValueError(f"cache world count mismatch: {path}")
        expected_actions = [repr(action) for action in bet_candidates(context.worlds[0].observe(context.hero_seat)).actions]
        if value.get("target", {}).get("actions") != expected_actions:
            raise ValueError(f"cache action ordering mismatch: {path}")
        if value.get("sha256") != sha256(json.dumps({k: v for k, v in value.items() if k != "sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest():
            raise ValueError(f"cache content hash mismatch: {path}")
        return value

    def get_or_compute(self, context: MultiStreetContext, profile: str, *, max_nodes: int, deadline: float) -> dict[str, Any]:
        cached = self.load(context, profile)
        if cached is not None:
            return cached
        reference = enumerate_reference(context, ReferenceProfile(profile), max_nodes=max_nodes, deadline=deadline)
        value = {
            "format": "multistreet-reference-cache-v1",
            "source_sha256": self.source_sha256,
            "plan_sha256": self.plan_sha256,
            "stream_namespace": self.stream_namespace,
            "stream_seed": self.stream_seed,
            "stream_key": self.stream_key,
            "selected_n": self.selected_n,
            "context": context.name,
            "split": context.split,
            "street": context.street,
            "situation": "facing" if context.facing else "open",
            "profile": profile,
            "world_seeds": list(context.world_seeds),
            "world_fingerprint": self._world_fingerprint(context),
            "worlds": reference.worlds,
            "world_action_values_bb": [list(row) for row in reference.world_action_values_bb],
            "action_standard_error_bb": list(reference.action_standard_error_bb),
            "uncertainty_status": reference.uncertainty_status,
            "nodes": reference.nodes,
            "seconds": reference.seconds,
            "target": _serial_target(reference.target),
        }
        identity = dict(value)
        value["sha256"] = sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        _atomic_json(self.path(context, profile), value)
        return value


def build_cached_rows(plan, cache, *, deadline, context_filter=None, reference_workers=1, progress=None):
    from src.holdem.multistreet_collection import collect_rows

    return collect_rows(
        plan, cache, deadline=deadline, context_filter=context_filter,
        reference_workers=reference_workers, progress=progress,
    )
