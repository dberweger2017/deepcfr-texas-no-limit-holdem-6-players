"""Select fitting duration on tuning data, then qualify frozen recipes."""

from math import isfinite


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def choose_durations(measurements, plan):
    """Choose one duration per architecture across every declared seed."""
    fields = {"variant", "seed", "duration", "weighted_decision_cost_bb"}
    expected = {
        (variant, seed, duration)
        for variant in plan["variants"]
        for seed in plan["seeds"]
        for duration in plan["durations"]
    }
    indexed = {}
    for row in measurements:
        # This interface deliberately cannot accept validation or test curves.
        if set(row) != fields:
            raise ValueError("Duration selection accepts tuning costs only")
        key = (row["variant"], row["seed"], row["duration"])
        if key in indexed or key not in expected:
            raise ValueError("Unexpected or duplicate tuning measurement")
        indexed[key] = _number(row["weighted_decision_cost_bb"], "Tuning cost")
    if not expected or set(indexed) != expected:
        raise ValueError("Incomplete tuning matrix")
    chosen, scores = {}, {}
    for variant in plan["variants"]:
        options = [
            (
                sum(indexed[variant, seed, duration] for seed in plan["seeds"])
                / len(plan["seeds"]),
                duration,
            )
            for duration in plan["durations"]
        ]
        chosen[variant] = min(options)[1]
        scores[variant] = [
            {"duration": duration, "mean_tuning_cost_bb": cost}
            for cost, duration in sorted(options, key=lambda item: item[1])
        ]
    return {"durations": chosen, "scores": scores}


def qualify(measurements, durations, plan):
    """Qualify every candidate using only its already selected checkpoint.

    Paired gain uncertainty concerns Monte Carlo reference noise conditional on
    these board groups. It does not measure generalization to a poker population.
    """
    expected = {(variant, seed) for variant in plan["variants"] for seed in plan["seeds"]}
    if set(durations) != set(plan["variants"]):
        raise ValueError("Every architecture needs a selected duration")
    indexed = {}
    for row in measurements:
        key = (row["variant"], row["seed"])
        if key in indexed or key not in expected:
            raise ValueError("Unexpected or duplicate validation measurement")
        if row["duration"] != durations[row["variant"]]:
            raise ValueError("Validation must use the selected duration")
        indexed[key] = row
    if not expected or set(indexed) != expected:
        raise ValueError("Incomplete validation matrix")
    baseline = plan["variants"][0]
    details, eligible = {}, []
    for variant in plan["variants"][1:]:
        checks = []
        for seed in plan["seeds"]:
            candidate, control = indexed[variant, seed], indexed[baseline, seed]
            cost = _number(candidate["weighted_decision_cost_bb"], "Candidate cost")
            control_cost = _number(control["weighted_decision_cost_bb"], "Control cost")
            gain = control_cost - cost
            se = _number(candidate["paired_gain_standard_error_bb"], "Paired gain SE")
            if se < 0:
                raise ValueError("Paired gain SE cannot be negative")
            lower = gain - plan["reference_se_multiplier"] * se
            error, control_error = candidate["relative_rmse"], control["relative_rmse"]
            error_ok = (
                error is not None and control_error is not None
                and _number(error, "Candidate error")
                <= _number(control_error, "Control error") + plan["allowed_relative_rmse_increase"]
            )
            passed = (
                gain >= plan["minimum_absolute_cost_gain_bb"]
                and gain >= plan["minimum_relative_cost_gain"] * control_cost
                and lower > 0
                and error_ok
            )
            checks.append({
                "seed": seed,
                "cost_gain_bb": gain,
                "paired_gain_standard_error_bb": se,
                "reference_noise_lower_bound_bb": lower,
                "error_passed": error_ok,
                "passed": passed,
            })
        details[variant] = {"seeds": checks, "passed": all(row["passed"] for row in checks)}
        if details[variant]["passed"]:
            eligible.append(variant)
    return {"eligible": eligible, "candidates": details}
