"""Bounded stability audit for independent multi-street reference batches.

The campaign keeps two independent saved batches for the same training
contexts: the calibration stream and the production stream.  This module
compares their saved world values on the frozen per-stratum world prefix.  It
does not rebuild worlds, resample, or change any campaign qualification rule.

Metrics are defined here before aggregate results are read:

* Q and regret RMSE are reported per context and averaged by stratum; pooled
  RMSE and correlation are also reported. Pearson correlation is ``None``
  when either vector is constant (or has fewer than two values).
* Pairwise sign agreement compares every unordered action-value difference.
  Absolute differences <= 1e-12 are ties; a tie agrees only with a tie.
  Top-action agreement compares the complete argmax sets, so tied maxima are
  handled explicitly.
* Regret-matched TV is half the L1 distance between policies formed from
  positive regrets.  If no regret is positive, the first maximum-regret action
  receives probability one, matching the trainer's tie break.
* A decision cost is ``max(Q) - policy @ Q``.  Within-batch costs use each
  batch's own regret-matched policy and Q.  Cross-batch costs evaluate each
  policy against the other batch's Q. Signed paired policy-value differences
  are ``(pi_calibration - pi_production) @ Q_batch`` for each Q batch.

The campaign's two continuation profiles are retained separately.  A third
``combined_1_to_2`` view combines uniform and increasing Q/regret targets with
the declared 1:2 schedule before regret matching.
"""

from __future__ import annotations

import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FROZEN_WORLD_BUDGETS = {
    "flop:open": 128,
    "flop:facing": 128,
    "turn:open": 128,
    "turn:facing": 128,
    "river:open": 32,
    "river:facing": 16,
}
PROFILES = ("uniform", "increasing")
COMBINED_PROFILE = "combined_1_to_2"
FORMAT = "multistreet-reference-stability-v1"
_TIE_TOLERANCE = 1e-12


def _finite_vector(values: Sequence[Any], name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _profile_policy(profile: str, width: int) -> np.ndarray:
    if profile == "uniform":
        result = np.full(width, 1.0 / width)
    elif profile == "increasing":
        weights = np.arange(1, width + 1, dtype=float)
        result = weights / weights.sum()
    else:
        raise ValueError(f"unknown reference profile: {profile}")
    return result


def _stream_key(entry: Mapping[str, Any]) -> str:
    return sha256(
        f"{entry['stream_namespace']}|{entry['stream_seed']}|"
        f"{entry['plan_sha256']}|{entry['selected_n']}".encode()
    ).hexdigest()


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.ptp(left) <= _TIE_TOLERANCE or np.ptp(right) <= _TIE_TOLERANCE:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _regret_matching(regrets: np.ndarray) -> np.ndarray:
    positive = np.maximum(regrets, 0.0)
    total = float(positive.sum())
    if total > 0:
        return positive / total
    result = np.zeros(regrets.size, dtype=float)
    result[int(np.argmax(regrets))] = 1.0
    return result


def _sign(value: float) -> int:
    if value > _TIE_TOLERANCE:
        return 1
    if value < -_TIE_TOLERANCE:
        return -1
    return 0


def _pairwise_and_top(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape or left.size < 2:
        raise ValueError("pairwise comparison needs equally sized action vectors")
    signs_left, signs_right = [], []
    for i in range(left.size):
        for j in range(i + 1, left.size):
            signs_left.append(_sign(float(left[i] - left[j])))
            signs_right.append(_sign(float(right[i] - right[j])))
    agreements = [a == b for a, b in zip(signs_left, signs_right, strict=True)]
    non_tied = [
        a == b
        for a, b in zip(signs_left, signs_right, strict=True)
        if a and b
    ]
    left_top = {i for i, value in enumerate(left) if abs(value - left.max()) <= _TIE_TOLERANCE}
    right_top = {i for i, value in enumerate(right) if abs(value - right.max()) <= _TIE_TOLERANCE}
    union = left_top | right_top
    return {
        "pairwise_sign_agreement": float(np.mean(agreements)),
        "pairwise_total": len(agreements),
        "pairwise_non_tie_agreement": (
            float(np.mean(non_tied)) if non_tied else None
        ),
        "pairwise_non_tie_total": len(non_tied),
        "tie_rate_left": float(np.mean(np.asarray(signs_left) == 0)),
        "tie_rate_right": float(np.mean(np.asarray(signs_right) == 0)),
        "top_action_agreement": float(left_top == right_top),
        "top_action_overlap": float(len(left_top & right_top) / len(union)),
        "top_actions_left": sorted(left_top),
        "top_actions_right": sorted(right_top),
    }


def _decision_metrics(left_values: np.ndarray, left_regrets: np.ndarray,
                      right_values: np.ndarray, right_regrets: np.ndarray) -> dict[str, Any]:
    if not (
        left_values.shape == left_regrets.shape == right_values.shape == right_regrets.shape
    ):
        raise ValueError("batch target vectors must have the same action width")
    left_policy = _regret_matching(left_regrets)
    right_policy = _regret_matching(right_regrets)
    left_within = float(left_values.max() - left_policy @ left_values)
    right_within = float(right_values.max() - right_policy @ right_values)
    left_on_right = float(right_values.max() - left_policy @ right_values)
    right_on_left = float(left_values.max() - right_policy @ left_values)
    return {
        "q_rmse_bb": float(np.sqrt(np.mean((left_values - right_values) ** 2))),
        "q_correlation": _correlation(left_values, right_values),
        "regret_rmse_bb": float(np.sqrt(np.mean((left_regrets - right_regrets) ** 2))),
        "regret_correlation": _correlation(left_regrets, right_regrets),
        "regret_matched_tv": float(0.5 * np.abs(left_policy - right_policy).sum()),
        **_pairwise_and_top(left_values, right_values),
        "within_cost_left_bb": left_within,
        "within_cost_right_bb": right_within,
        "cross_cost_left_policy_on_right_bb": left_on_right,
        "cross_cost_right_policy_on_left_bb": right_on_left,
        # Signed as (pi_calibration - pi_production) @ Q_batch. On left Q this
        # equals cross(right policy on left Q) minus the left within cost. On
        # right Q it equals the right within cost minus cross(left policy on
        # right Q).
        "paired_policy_value_difference_on_left_bb": float((left_policy - right_policy) @ left_values),
        "paired_policy_value_difference_on_right_bb": float((left_policy - right_policy) @ right_values),
        "left_policy": left_policy.tolist(),
        "right_policy": right_policy.tolist(),
        "_q_left": left_values.tolist(),
        "_q_right": right_values.tolist(),
        "_regret_left": left_regrets.tolist(),
        "_regret_right": right_regrets.tolist(),
    }


def _mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty stability stratum")
    keys = (
        "q_rmse_bb", "q_correlation", "regret_rmse_bb", "regret_correlation",
        "regret_matched_tv", "pairwise_sign_agreement", "pairwise_non_tie_agreement",
        "tie_rate_left", "tie_rate_right", "top_action_agreement", "top_action_overlap",
        "within_cost_left_bb", "within_cost_right_bb",
        "cross_cost_left_policy_on_right_bb", "cross_cost_right_policy_on_left_bb",
        "paired_policy_value_difference_on_left_bb", "paired_policy_value_difference_on_right_bb",
    )
    result = {key: _mean_metric(rows, key) for key in keys}
    result.update({"contexts": len(rows), "pairwise_total": sum(row["pairwise_total"] for row in rows)})
    result["pairwise_non_tie_total"] = sum(row["pairwise_non_tie_total"] for row in rows)
    q_left = np.concatenate([np.asarray(row["_q_left"]) for row in rows])
    q_right = np.concatenate([np.asarray(row["_q_right"]) for row in rows])
    regret_left = np.concatenate([np.asarray(row["_regret_left"]) for row in rows])
    regret_right = np.concatenate([np.asarray(row["_regret_right"]) for row in rows])
    result["pooled_q_rmse_bb"] = float(np.sqrt(np.mean((q_left - q_right) ** 2)))
    result["pooled_q_correlation"] = _correlation(q_left, q_right)
    result["pooled_regret_rmse_bb"] = float(np.sqrt(np.mean((regret_left - regret_right) ** 2)))
    result["pooled_regret_correlation"] = _correlation(regret_left, regret_right)
    return result


def _public_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def _validate_entry(entry: Mapping[str, Any], *, batch: str, profile: str) -> None:
    required = {
        "format", "source_sha256", "plan_sha256", "stream_namespace", "stream_seed",
        "stream_key", "selected_n", "context", "split", "street", "situation",
        "profile", "worlds", "world_action_values_bb", "target", "world_fingerprint", "sha256",
    }
    if not required <= set(entry):
        raise ValueError(f"{batch} entry {entry.get('context')} is missing cache fields")
    if entry["format"] != "multistreet-reference-cache-v1":
        raise ValueError("unexpected reference cache format")
    expected_stream = "calibration" if batch == "calibration" else "production"
    if entry["stream_namespace"] != expected_stream:
        raise ValueError(f"{batch} cache has the wrong stream namespace")
    if entry["stream_key"] != _stream_key(entry):
        raise ValueError(f"{entry.get('context')} has an invalid stream key")
    expected_hash = sha256(
        json.dumps(
            {key: value for key, value in entry.items() if key not in {"sha256", "_cache_file_sha256"}},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if entry["sha256"] != expected_hash:
        raise ValueError(f"{entry.get('context')} has an invalid cache content hash")
    if entry["profile"] != profile:
        raise ValueError("cache profile identity mismatch")
    if entry["split"] != "train":
        raise ValueError("stability overlap is restricted to training contexts")
    if entry["street"] not in {"flop", "turn", "river"} or entry["situation"] not in {"open", "facing"}:
        raise ValueError("invalid street or situation")
    stratum = f"{entry['street']}:{entry['situation']}"
    budget = FROZEN_WORLD_BUDGETS[stratum]
    if batch == "production":
        if entry["selected_n"] != FROZEN_WORLD_BUDGETS:
            raise ValueError(f"{entry['context']} does not use the frozen production budgets")
    elif entry["selected_n"] != 128:
        raise ValueError(f"{entry['context']} does not use the calibration 128-world budget")
    rows = entry["world_action_values_bb"]
    expected_worlds = budget if batch == "production" else 128
    if not isinstance(rows, list) or len(rows) != expected_worlds:
        raise ValueError(f"{entry['context']} does not contain the frozen world budget")
    if entry["worlds"] != len(rows) or entry["world_fingerprint"].get("worlds") != len(rows):
        raise ValueError(f"{entry['context']} has inconsistent world counts")
    seeds = entry["world_fingerprint"].get("world_seeds")
    if not isinstance(seeds, list) or len(seeds) != len(rows) or len(set(seeds)) != len(seeds):
        raise ValueError(f"{entry['context']} has an invalid world stream")
    target = entry["target"]
    actions = target.get("actions")
    values = _finite_vector(target.get("values_bb"), "target values")
    regrets = _finite_vector(target.get("regrets_bb"), "target regrets")
    policy = _finite_vector(target.get("policy"), "target policy")
    if (
        not isinstance(actions, list)
        or len(actions) != values.size
        or values.shape != regrets.shape
        or regrets.shape != policy.shape
    ):
        raise ValueError(f"{entry['context']} has inconsistent action vectors")
    if not np.isclose(policy.sum(), 1.0, rtol=0, atol=1e-10) or (policy < 0).any():
        raise ValueError(f"{entry['context']} has an invalid continuation policy")
    expected_policy = _profile_policy(profile, values.size)
    if not np.allclose(policy, expected_policy, rtol=0, atol=1e-12):
        raise ValueError(f"{entry['context']} has a profile-policy mismatch")
    world_values = np.asarray(rows, dtype=float)
    if world_values.ndim != 2 or world_values.shape[1] != values.size or not np.isfinite(world_values).all():
        raise ValueError(f"{entry['context']} has invalid world action values")
    expected_values = world_values.mean(axis=0)
    expected_regrets = expected_values - float(policy @ expected_values)
    if not np.allclose(values, expected_values, rtol=0, atol=1e-10) or not np.allclose(
        regrets, expected_regrets, rtol=0, atol=1e-10
    ):
        raise ValueError(f"{entry['context']} target disagrees with saved world values")
    if not isinstance(entry["stream_key"], str) or not entry["stream_key"]:
        raise ValueError("cache stream key is missing")
    if not isinstance(entry["source_sha256"], str) or not isinstance(entry["plan_sha256"], str):
        raise ValueError("cache source fingerprints are missing")


def _target_from_prefix(entry: Mapping[str, Any], profile: str, budget: int) -> tuple[np.ndarray, np.ndarray]:
    world_values = np.asarray(entry["world_action_values_bb"][:budget], dtype=float)
    if world_values.ndim != 2 or not np.isfinite(world_values).all():
        raise ValueError(f"{entry['context']} has invalid world action values")
    policy = _profile_policy(profile, world_values.shape[1])
    values = world_values.mean(axis=0)
    regrets = values - float(policy @ values)
    return values, regrets


def _batch_identity(entries: Sequence[Mapping[str, Any]], batch: str) -> dict[str, Any]:
    return {
        "stream_namespace": batch,
        "source_sha256": sorted({entry["source_sha256"] for entry in entries}),
        "plan_sha256": sorted({entry["plan_sha256"] for entry in entries}),
        "stream_keys": sorted({entry["stream_key"] for entry in entries}),
        # Keep the cache's own content fingerprints verbatim so the audit
        # remains tied to the archived source objects.
        "cache_sha256": sorted(
            entry.get("_cache_file_sha256", entry["sha256"]) for entry in entries
        ),
    }


def audit_cache_entries(
    calibration_entries: Iterable[Mapping[str, Any]],
    production_entries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit two batches of saved cache entries and return JSON-ready metrics."""

    batches = {
        "calibration": list(calibration_entries),
        "production": list(production_entries),
    }
    indexed: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    for batch, entries in batches.items():
        if not entries:
            raise ValueError(f"{batch} cache is empty")
        by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
        for entry in entries:
            profile = entry.get("profile")
            if profile not in PROFILES:
                raise ValueError(f"unknown or missing profile in {batch} cache")
            _validate_entry(entry, batch=batch, profile=profile)
            key = (str(entry["context"]), profile)
            if key in by_key:
                raise ValueError(f"duplicate {batch} context/profile {key}")
            by_key[key] = entry
        indexed[batch] = by_key
        stream_keys = {entry["stream_key"] for entry in entries}
        stream_seeds = {entry["stream_seed"] for entry in entries}
        if len(stream_keys) != 1 or len(stream_seeds) != 1:
            raise ValueError(f"{batch} entries do not share one declared stream")
    overlap = set(indexed["calibration"]) & set(indexed["production"])
    if not overlap:
        raise ValueError("calibration and production caches have no overlap")
    if not set(indexed["calibration"]) <= set(indexed["production"]):
        raise ValueError("every calibration context/profile must exist in production")
    # Production includes the full campaign train split; only the 72 contexts
    # present in the independent calibration batch are in scope here.
    overlap = set(indexed["calibration"])
    calibration_stream_keys = {indexed["calibration"][key]["stream_key"] for key in overlap}
    production_stream_keys = {indexed["production"][key]["stream_key"] for key in overlap}
    calibration_seeds = {indexed["calibration"][key]["stream_seed"] for key in overlap}
    production_seeds = {indexed["production"][key]["stream_seed"] for key in overlap}
    if calibration_stream_keys & production_stream_keys or calibration_seeds & production_seeds:
        raise ValueError("calibration and production streams are not independent")
    source_fingerprints = {
        indexed[batch][key]["source_sha256"]
        for batch in ("calibration", "production")
        for key in overlap
    }
    if len(source_fingerprints) != 1:
        raise ValueError("calibration and production source fingerprints differ")
    contexts = sorted({context for context, _ in overlap})
    if len(contexts) * 2 != len(overlap):
        raise ValueError("every overlapping context needs both profiles")

    context_rows = []
    by_stratum: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for context in contexts:
        profile_rows: dict[str, dict[str, Any]] = {}
        for profile in PROFILES:
            left = indexed["calibration"][context, profile]
            right = indexed["production"][context, profile]
            for field in ("street", "situation", "split"):
                if left[field] != right[field]:
                    raise ValueError(f"{context} disagrees on {field}")
            if left["world_fingerprint"].get("visible_observation_sha256") != right["world_fingerprint"].get("visible_observation_sha256"):
                raise ValueError(f"{context} has different public observations")
            if left["target"]["actions"] != right["target"]["actions"]:
                raise ValueError(f"{context} has mismatched action ordering")
            for batch, entry in (("calibration", left), ("production", right)):
                for other_profile in PROFILES:
                    other = indexed[batch][context, other_profile]
                    if entry["target"]["actions"] != other["target"]["actions"]:
                        raise ValueError(f"{batch} profiles have mismatched actions for {context}")
                    if entry["world_fingerprint"].get("visible_observation_sha256") != other["world_fingerprint"].get("visible_observation_sha256"):
                        raise ValueError(f"{batch} profiles have mismatched observations for {context}")
            stratum = f"{left['street']}:{left['situation']}"
            budget = FROZEN_WORLD_BUDGETS[stratum]
            left_values, left_regrets = _target_from_prefix(left, profile, budget)
            right_values, right_regrets = _target_from_prefix(right, profile, budget)
            profile_rows[profile] = _decision_metrics(
                left_values, left_regrets, right_values, right_regrets
            )
        for batch in ("calibration", "production"):
            for profile in PROFILES:
                for other in PROFILES:
                    entry = indexed[batch][context, profile]
                    other_entry = indexed[batch][context, other]
                    if entry["world_fingerprint"] != other_entry["world_fingerprint"]:
                        raise ValueError(f"{batch} profile streams differ for {context}")
        combined = {}
        for batch in ("calibration", "production"):
            values_by_profile, regrets_by_profile = {}, {}
            stratum = f"{indexed[batch][context, 'uniform']['street']}:{indexed[batch][context, 'uniform']['situation']}"
            budget = FROZEN_WORLD_BUDGETS[stratum]
            for profile in PROFILES:
                values_by_profile[profile], regrets_by_profile[profile] = _target_from_prefix(
                    indexed[batch][context, profile], profile, budget
                )
            values = (values_by_profile["uniform"] + 2 * values_by_profile["increasing"]) / 3
            regrets = (regrets_by_profile["uniform"] + 2 * regrets_by_profile["increasing"]) / 3
            combined[batch] = (values, regrets)
        profile_rows[COMBINED_PROFILE] = _decision_metrics(
            *combined["calibration"], *combined["production"]
        )
        stratum = f"{indexed['calibration'][context, 'uniform']['street']}:{indexed['calibration'][context, 'uniform']['situation']}"
        row = {
            "context": context,
            "stratum": stratum,
            "profiles": {profile: _public_metrics(metrics) for profile, metrics in profile_rows.items()},
        }
        context_rows.append(row)
        for profile, metrics in profile_rows.items():
            by_stratum[stratum][profile].append(metrics)

    strata = {
        stratum: {
            "world_budget": FROZEN_WORLD_BUDGETS[stratum],
            "profiles": {profile: _aggregate(rows) for profile, rows in sorted(profile_rows.items())},
        }
        for stratum, profile_rows in sorted(by_stratum.items())
    }
    return {
        "format": FORMAT,
        "metrics_definition": {
            "q_and_regret_rmse": "per-context mean and pooled root mean square differences across action values",
            "correlation": "Pearson correlation; null for a constant vector or fewer than two values",
            "pairwise_sign": "all unordered action differences; absolute differences <= 1e-12 are ties and agree only with ties",
            "top_action": "exact agreement of complete argmax action sets, with overlap also reported",
            "regret_matched_tv": "half L1 distance after positive-regret matching; first max regret wins when all are non-positive",
            "decision_cost": "max(Q) minus policy dot Q in BB",
            "cross_batch_cost": "each batch policy evaluated against the other batch's Q",
            "paired_policy_value_difference": "(pi_calibration - pi_production) dot Q_batch, signed separately on each batch Q",
            "profile_schedule": "combined_1_to_2 uses (uniform + 2 * increasing) / 3 for Q and regret targets",
            "limitations": "descriptive training-context audit; calibration helped select world budgets, the 12 families are shared across strata, no CI is inferred across correlated contexts, noisy-max bias and non-greedy regret matching remain interpretation limits",
        },
        "frozen_world_budgets": dict(FROZEN_WORLD_BUDGETS),
        "batches": {batch: _batch_identity(entries, batch) for batch, entries in batches.items()},
        "overlap": {"contexts": len(contexts), "entries": len(overlap), "training_only": True},
        "strata": strata,
        "contexts": context_rows,
    }


def load_cache_directory(path: Path, *, batch: str) -> list[dict[str, Any]]:
    """Load all JSON cache entries for a declared independent stream."""

    files = sorted(Path(path).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no cache JSON files under {path}")
    entries = []
    for file in files:
        raw = file.read_bytes()
        value = json.loads(raw)
        # The audit is intentionally bounded to the training overlap.  Held
        # out production entries remain archived, but never enter this audit.
        if value.get("stream_namespace") != batch or value.get("split") != "train":
            continue
        value["_cache_file_sha256"] = sha256(raw).hexdigest()
        entries.append(value)
    if not entries:
        raise ValueError(f"no {batch} cache entries under {path}")
    return entries


def run_audit(calibration_cache: Path, production_cache: Path, output: Path) -> dict[str, Any]:
    result = audit_cache_entries(
        load_cache_directory(calibration_cache, batch="calibration"),
        load_cache_directory(production_cache, batch="production"),
    )
    if result["overlap"]["contexts"] != 72 or any(
        value["profiles"][COMBINED_PROFILE]["contexts"] != 12
        for value in result["strata"].values()
    ):
        raise ValueError("archived campaign audit is missing the expected 72/12 context roster")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result
