"""Analyze saved multi-street decision errors and coarse training coverage.

This is deliberately a saved-data analysis.  It does not load checkpoints,
run model inference, fit models, or inspect the sealed test split.  The fit
report contains the probabilities and reference values needed to reconstruct
each decision cost.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev

from src.game.showdown import hand_value


ANALYZED_SPLITS = ("train", "tuning", "validation")
EXPECTED_COUNTS = {"train": 576, "tuning": 192, "validation": 192}
ACTION_KIND_RE = re.compile(r"Action\(kind=<ActionKind\.([A-Z_]+): '([^']+)'>")


def reconstruct_cost(probabilities, values):
    """Return max-Q policy loss and its action-level contributions."""

    p = [float(x) for x in probabilities]
    q = [float(x) for x in values]
    if len(p) != len(q) or not p:
        raise ValueError("probabilities and values must have the same nonzero length")
    if any(not math.isfinite(x) or x < 0 for x in p) or any(not math.isfinite(x) for x in q):
        raise ValueError("probabilities and values must be finite; probabilities nonnegative")
    if not math.isclose(sum(p), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("probabilities must sum to one")
    maximum = max(q)
    contributions = [weight * (maximum - value) for weight, value in zip(p, q)]
    return {
        "max_value_bb": maximum,
        "policy_value_bb": sum(weight * value for weight, value in zip(p, q)),
        "cost_bb": sum(contributions),
        "contributions_bb": contributions,
    }


def paired_best_action_gap(world_values):
    """Compute best-vs-second-best gap and paired-world standard error."""

    worlds = [[float(x) for x in row] for row in world_values]
    if not worlds or not worlds[0] or any(len(row) != len(worlds[0]) for row in worlds):
        raise ValueError("world values must be a nonempty rectangular array")
    if any(not math.isfinite(value) for row in worlds for value in row):
        raise ValueError("world values must be finite")
    means = [sum(row[index] for row in worlds) / len(worlds) for index in range(len(worlds[0]))]
    order = sorted(range(len(means)), key=lambda index: means[index], reverse=True)
    best, second = order[0], order[1] if len(order) > 1 else order[0]
    paired = [row[best] - row[second] for row in worlds]
    se = None if len(paired) < 2 else stdev(paired) / math.sqrt(len(paired))
    return {
        "best_action_index": best,
        "second_action_index": second,
        "best_value_bb": means[best],
        "second_value_bb": means[second],
        "gap_bb": means[best] - means[second],
        "gap_se_bb": se,
        "world_count": len(worlds),
    }


def parse_action(text):
    """Extract the semantic action kind from an Action repr."""

    match = ACTION_KIND_RE.search(text)
    if not match:
        raise ValueError(f"Unrecognized saved action representation: {text!r}")
    kind_name, kind = match.groups()
    raise_to = None
    if kind == "raise":
        suffix = text.split("raise_to=", 1)[-1].rstrip(")")
        raise_to = suffix if suffix != "None" else None
    return {"label": text, "kind": kind, "kind_name": kind_name, "raise_to": raise_to}


def visible_classes(holding, board):
    """Return transparent made-hand and board texture labels."""

    category_names = (
        "high_card", "pair", "two_pair", "trips", "straight", "flush",
        "full_house", "quads", "straight_flush",
    )
    category = hand_value(tuple(holding) + tuple(board))[0]
    suits = Counter(card[1] for card in board)
    maximum_suit = max(suits.values())
    if maximum_suit >= 3:
        flush_texture = "three_or_more_same_suit"
    elif maximum_suit == 2:
        flush_texture = "two_same_suit"
    else:
        flush_texture = "all_distinct_suits"
    return {
        "visible_hand_category": category_names[category],
        "board_paired": len({card[0] for card in board}) < len(board),
        "board_flush_texture": flush_texture,
    }


def _file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _load_inputs(report_path, contexts_path, manifest_path, cache_dir):
    report = json.loads(Path(report_path).read_text())
    contexts = json.loads(Path(contexts_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if report.get("status") != "completed":
        raise ValueError("Only a completed fit report can be analyzed")
    specs = manifest["expanded_plan"]["contexts"]
    if len(contexts) != len(specs):
        raise ValueError("contexts.json and expanded_plan context counts differ")
    by_name = {}
    for index, (row, spec) in enumerate(zip(contexts, specs, strict=True)):
        expected_name = f"{spec['street']}-{index}"
        if row["name"] != expected_name:
            raise ValueError(f"Context ordering/name join failed at {expected_name}")
        if any(row[key] != expected for key, expected in (
            ("split", spec["split"]),
            ("street", spec["street"]),
            ("situation", "facing" if spec["facing"] else "open"),
        )):
            raise ValueError(f"Context metadata join failed for {row['name']}")
        if row["split"] in ANALYZED_SPLITS:
            enriched = dict(row)
            enriched.update({"board": spec["board"], "holding": spec["holding"], "family": spec["family"]})
            enriched.update(visible_classes(spec["holding"], spec["board"]))
            by_name[row["name"]] = enriched
    counts = Counter(row["split"] for row in contexts)
    if any(counts[split] != expected for split, expected in EXPECTED_COUNTS.items()):
        raise ValueError(f"Unexpected campaign split counts: {dict(counts)}")

    cache_dir = Path(cache_dir)
    cache_entries = {}
    for entry in manifest["production_cache"]:
        context = entry["context"]
        if context not in by_name:
            continue
        key = (context, entry["profile"])
        if key in cache_entries:
            raise ValueError(f"Duplicate production cache entry: {key}")
        path = cache_dir / Path(entry["path"]).name
        if not path.exists() or _file_hash(path) != entry["sha256"]:
            raise ValueError(f"Production cache hash/path mismatch: {path}")
        value = json.loads(path.read_text())
        if value.get("sha256") is None:
            raise ValueError(f"Cache entry has no identity hash: {path}")
        if value.get("context") != context or value.get("profile") != entry["profile"]:
            raise ValueError(f"Cache manifest join failed: {path}")
        if value.get("source_sha256") != report.get("source_sha256"):
            raise ValueError(f"Cache source fingerprint mismatch: {path}")
        if value.get("plan_sha256") != report.get("plan_sha256"):
            raise ValueError(f"Cache plan fingerprint mismatch: {path}")
        cache_entries[key] = value
    expected_cache_keys = {(name, profile) for name in by_name for profile in ("uniform", "increasing")}
    if set(cache_entries) != expected_cache_keys:
        raise ValueError("Production cache is incomplete for analyzed contexts")
    for name in by_name:
        left = cache_entries[name, "uniform"]
        right = cache_entries[name, "increasing"]
        if left["target"]["actions"] != right["target"]["actions"]:
            raise ValueError(f"Action order differs between profiles for {name}")
        if left["worlds"] != by_name[name]["world_count"]:
            raise ValueError(f"Frozen duration/world count mismatch for {name}")
    return report, by_name, cache_entries


def _selected_decisions(report, durations):
    records = {}
    variants = report["plan"]["variants"]
    seeds = report["plan"]["seeds"]
    for fit in report["fits"]:
        if fit["duration"] != durations[fit["variant"]]:
            continue
        for split in ("train", "tuning"):
            key = (fit["variant"], fit["seed"], split)
            if key in records:
                raise ValueError(f"Duplicate selected fit record: {key}")
            records[key] = fit["metrics"][split]["decisions"]
    for row in report["validation"]:
        expected = durations[row["variant"]]
        if row["duration"] != expected:
            raise ValueError(f"Validation duration is not frozen selected duration: {row}")
        key = (row["variant"], row["seed"], "validation")
        if key in records:
            raise ValueError(f"Duplicate selected validation record: {key}")
        records[key] = row["metric"]["decisions"]
    expected_keys = {(variant, seed, split) for variant in variants for seed in seeds for split in ANALYZED_SPLITS}
    if set(records) != expected_keys:
        raise ValueError("Selected fit/validation roster is incomplete")
    for key, decisions in records.items():
        if len(decisions) != EXPECTED_COUNTS[key[2]]:
            raise ValueError(f"Decision record count mismatch for {key}")
    return records


def _action_metadata(cache_entry):
    return [parse_action(action) for action in cache_entry["target"]["actions"]]


def _regret_matching(regrets):
    if not regrets or any(not math.isfinite(float(regret)) for regret in regrets):
        raise ValueError("Predicted regrets must be nonempty and finite")
    positive = [max(float(regret), 0.0) for regret in regrets]
    maximum = max(positive)
    if maximum > 0:
        total = sum(positive)
        return [value / total for value in positive]
    best = max(range(len(regrets)), key=lambda index: float(regrets[index]))
    return [1.0 if index == best else 0.0 for index in range(len(regrets))]


def _enrich_decision(decision, context, cache_entry):
    actions = _action_metadata(cache_entry)
    probabilities = decision["probabilities"]
    values = decision["reference_values_bb"]
    if len(actions) != len(probabilities) or len(values) != len(actions):
        raise ValueError(f"Saved action/probability/value lengths differ for {decision['context']}")
    if "predicted_regrets_bb" not in decision:
        raise ValueError(f"Saved decision lacks predicted regrets: {decision['context']}")
    expected_probabilities = _regret_matching(decision["predicted_regrets_bb"])
    if len(expected_probabilities) != len(probabilities) or any(
        not math.isclose(expected, actual, rel_tol=0.0, abs_tol=2e-6)
        for expected, actual in zip(expected_probabilities, probabilities, strict=True)
    ):
        raise ValueError(f"Saved probabilities do not match regret matching for {decision['context']}")
    world_values = context["world_action_values_bb"]
    if len(world_values) != context["world_count"] or any(
        len(row) != len(actions) for row in world_values
    ):
        raise ValueError(f"Saved world-Q shape differs from actions for {decision['context']}")
    world_means = [
        sum(float(row[index]) for row in world_values) / len(world_values)
        for index in range(len(actions))
    ]
    if any(
        not math.isclose(expected, actual, rel_tol=0.0, abs_tol=2e-7)
        for expected, actual in zip(world_means, values, strict=True)
    ):
        raise ValueError(f"Saved world-Q means do not match reference values for {decision['context']}")
    reconstructed = reconstruct_cost(probabilities, values)
    if not math.isclose(reconstructed["cost_bb"], decision["decision_cost_bb"], rel_tol=0.0, abs_tol=2e-8):
        raise ValueError(f"Decision cost reconstruction failed for {decision['context']}")
    contributions = reconstructed["contributions_bb"]
    by_kind = defaultdict(float)
    for action, contribution in zip(actions, contributions, strict=True):
        by_kind[action["kind"]] += contribution
    world_gap = paired_best_action_gap(context["world_action_values_bb"])
    labels = [action["label"] for action in actions]
    enriched = {
        "context": decision["context"],
        "split": context["split"],
        "street": context["street"],
        "situation": context["situation"],
        "family": context["family"],
        "board": context["board"],
        "holding": context["holding"],
        "visible_hand_category": context["visible_hand_category"],
        "board_paired": context["board_paired"],
        "board_flush_texture": context["board_flush_texture"],
        "actions": actions,
        "probabilities": probabilities,
        "reference_values_bb": values,
        "decision_cost_bb": decision["decision_cost_bb"],
        "action_contributions_bb": contributions,
        "cost_by_action_kind_bb": dict(sorted(by_kind.items())),
        "missed_value_folds_bb": by_kind.get("fold", 0.0),
        "costly_calls_or_raises_bb": by_kind.get("call", 0.0) + by_kind.get("raise", 0.0),
        "reference_best_action": labels[world_gap["best_action_index"]],
        "reference_second_action": labels[world_gap["second_action_index"]],
        "reference_best_action_gap_bb": world_gap["gap_bb"],
        "reference_best_action_gap_se_bb": world_gap["gap_se_bb"],
        "reference_best_action_world_count": world_gap["world_count"],
    }
    return enriched


def _summarize(rows):
    costs = [row["decision_cost_bb"] for row in rows]
    total = sum(costs)
    sorted_costs = sorted(costs, reverse=True)
    return {
        "contexts": len(rows),
        "mean_cost_bb": mean(costs) if costs else 0.0,
        "total_cost_bb": total,
        "cost_concentration": {
            str(k): (sum(sorted_costs[:k]) / total if total else 0.0)
            for k in (5, 10)
        },
        "cost_by_action_kind_bb": dict(sorted({
            kind: sum(row["cost_by_action_kind_bb"].get(kind, 0.0) for row in rows)
            for kind in {kind for row in rows for kind in row["cost_by_action_kind_bb"]}
        }.items())),
        "missed_value_folds_bb": sum(row["missed_value_folds_bb"] for row in rows),
        "costly_calls_or_raises_bb": sum(row["costly_calls_or_raises_bb"] for row in rows),
    }


def _coverage(rows):
    # Coverage is a count of concrete campaign contexts, independent of how
    # many variants and seeds evaluated that context.
    unique = {}
    for row in rows:
        unique.setdefault(row["context"], row)
    rows = list(unique.values())
    def key(row):
        return (
            row["street"], row["situation"], row["visible_hand_category"],
            row["board_paired"], row["board_flush_texture"],
        )

    train_rows = [row for row in rows if row["split"] == "train"]
    train = Counter(key(row) for row in train_rows)
    train_families = defaultdict(set)
    for row in train_rows:
        train_families[key(row)].add(row["family"])
    by_split = {}
    for split in ANALYZED_SPLITS:
        counts = Counter(
            key(row)
            for row in rows if row["split"] == split
        )
        by_split[split] = [
            {
                "street": key[0], "situation": key[1], "visible_hand_category": key[2],
                "board_paired": key[3], "board_flush_texture": key[4],
                "context_count": count, "training_context_count": train.get(key, 0),
                "training_family_count": len(train_families.get(key, ())),
                "coarse_training_seen": train.get(key, 0) > 0,
            }
            for key, count in sorted(counts.items(), key=lambda item: item[0])
        ]
    return {
        "by_split": by_split,
        "interpretation": "Counts share coarse visible categories; they do not prove nearest-information-set equivalence.",
        "nearest_equivalence_proof": False,
    }


def _family_concentration(records):
    """Rank family losses after averaging each concrete family across seeds."""

    per_seed = defaultdict(list)
    for row in records:
        per_seed[row["variant"], row["split"], row["family"], row["seed"]].append(row)
    by_family = defaultdict(list)
    for (variant, split, family, seed), rows in per_seed.items():
        by_family[variant, split, family].append({
            "seed": seed,
            "total_cost_bb": sum(row["decision_cost_bb"] for row in rows),
            "mean_context_cost_bb": mean(row["decision_cost_bb"] for row in rows),
            "context_count": len(rows),
        })
    result = {}
    for variant, split in sorted({key[:2] for key in by_family}):
        families = []
        for (family_variant, family_split, family), values in by_family.items():
            if (family_variant, family_split) != (variant, split):
                continue
            families.append({
                "family": family,
                "seed_count": len(values),
                "mean_total_cost_bb": mean(value["total_cost_bb"] for value in values),
                "mean_context_cost_bb": mean(value["mean_context_cost_bb"] for value in values),
                "context_count": values[0]["context_count"],
            })
        families.sort(key=lambda row: row["mean_total_cost_bb"], reverse=True)
        total = sum(row["mean_total_cost_bb"] for row in families)
        result[f"{variant}/{split}"] = {
            "family_count": len(families),
            "concentration": {
                str(k): (sum(row["mean_total_cost_bb"] for row in families[:k]) / total if total else 0.0)
                for k in (5, 10)
            },
            "families": families,
        }
    return result


def analyze(report_path, contexts_path, manifest_path, cache_dir, out=None):
    report, contexts, cache_entries = _load_inputs(report_path, contexts_path, manifest_path, cache_dir)
    durations = report["duration_selection"]["durations"]
    decisions = _selected_decisions(report, durations)
    records = []
    for (variant, seed, split), rows in sorted(decisions.items()):
        expected_names = {name for name, context in contexts.items() if context["split"] == split}
        if {row["context"] for row in rows} != expected_names:
            raise ValueError(f"Decision roster differs from declared split: {variant}/{seed}/{split}")
        for decision in rows:
            if decision["context"] not in contexts:
                raise ValueError(f"Decision references excluded or unknown context: {decision['context']}")
            context = contexts[decision["context"]]
            cache = cache_entries[decision["context"], "uniform"]
            record = _enrich_decision(decision, context, cache)
            record.update({"variant": variant, "seed": seed, "duration": durations[variant]})
            records.append(record)

    summaries = []
    grouped = defaultdict(list)
    for row in records:
        grouped[row["variant"], row["seed"], row["split"], row["street"], row["situation"]].append(row)
    for key, rows in sorted(grouped.items()):
        variant, seed, split, street, situation = key
        summaries.append({
            "variant": variant, "seed": seed, "split": split,
            "street": street, "situation": situation, "duration": durations[variant],
            **_summarize(rows),
        })

    aggregate = []
    aggregate_groups = defaultdict(list)
    for row in records:
        aggregate_groups[row["variant"], row["split"], row["street"], row["situation"]].append(row)
    for key, rows in sorted(aggregate_groups.items()):
        variant, split, street, situation = key
        aggregate.append({
            "variant": variant, "split": split, "street": street, "situation": situation,
            "duration": durations[variant], "seed_count": len({row["seed"] for row in rows}),
            **_summarize(rows),
        })

    top_cases = {}
    for key, rows in sorted(grouped.items()):
        variant, seed, split, street, situation = key
        top_cases[f"{variant}/{seed}/{split}/{street}/{situation}"] = sorted(
            rows, key=lambda row: row["decision_cost_bb"], reverse=True
        )[:10]
    top_concrete_hands = {}
    for key, rows in sorted(aggregate_groups.items()):
        variant, split, street, situation = key
        by_context = defaultdict(list)
        for row in rows:
            by_context[row["context"]].append(row)
        averaged = []
        for context_name, values in by_context.items():
            first = values[0]
            averaged.append({
                "context": context_name, "board": first["board"], "holding": first["holding"],
                "visible_hand_category": first["visible_hand_category"], "board_paired": first["board_paired"],
                "board_flush_texture": first["board_flush_texture"], "seed_count": len(values),
                "mean_cost_bb": mean(row["decision_cost_bb"] for row in values),
                "mean_missed_value_folds_bb": mean(row["missed_value_folds_bb"] for row in values),
                "mean_costly_calls_or_raises_bb": mean(row["costly_calls_or_raises_bb"] for row in values),
            })
        top_concrete_hands[f"{variant}/{split}/{street}/{situation}"] = sorted(
            averaged, key=lambda row: row["mean_cost_bb"], reverse=True
        )[:10]

    result = {
        "format": "holdem-multistreet-decision-errors-v1",
        "source": {
            "report": str(report_path), "contexts": str(contexts_path),
            "manifest": str(manifest_path), "cache_dir": str(cache_dir),
            "report_sha256": _file_hash(report_path),
            "contexts_sha256": _file_hash(contexts_path),
            "manifest_sha256": _file_hash(manifest_path),
            "report_plan_sha256": report["plan_sha256"], "report_source_sha256": report["source_sha256"],
        },
        "scope": {
            "splits": list(ANALYZED_SPLITS), "excluded_split": "test",
            "variants": report["plan"]["variants"], "seeds": report["plan"]["seeds"],
            "selected_durations": durations, "model_inference": False,
            "training_or_fitting": False, "sealed_evaluation": False,
        },
        "validation": {
            "context_counts": dict(EXPECTED_COUNTS), "selected_record_count": len(records),
            "cost_reconstruction": "sum(probability[action] * (max(reference_values) - reference_values[action]))",
            "cache_profiles_joined": ["uniform", "increasing"],
            "action_order_source": "production cache target.actions; no array-index action inference",
        },
        "summaries_per_seed_variant": summaries,
        "summaries_average_across_seeds": aggregate,
        "top_cases_per_seed_variant": top_cases,
        "top_concrete_hands_average_across_seeds": top_concrete_hands,
        "family_cost_concentration_average_across_seeds": _family_concentration(records),
        "coverage": _coverage(records),
        "records": records,
    }
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("results/multistreet-retrieved/poker/results/multistreet-campaign")
    parser.add_argument("--report", type=Path, default=root / "fit/report.json")
    parser.add_argument("--contexts", type=Path, default=root / "fit/contexts.json")
    parser.add_argument("--manifest", type=Path, default=root / "campaign-manifest.json")
    parser.add_argument("--cache-dir", type=Path, default=root.parent / "multistreet-campaign-cache")
    parser.add_argument("--out", type=Path, default=Path("results/decision-errors/multistreet.json"))
    args = parser.parse_args()
    result = analyze(args.report, args.contexts, args.manifest, args.cache_dir, args.out)
    print(json.dumps({"output": str(args.out), "records": len(result["records"]), "status": "completed"}))


if __name__ == "__main__":
    main()
