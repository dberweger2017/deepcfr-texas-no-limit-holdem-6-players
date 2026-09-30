"""One frozen scientific stage; M4-only, durable rows, no outcome selection."""

import argparse
import gc
import gzip
import json
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from random import Random
from time import perf_counter, time

from scripts.diagnose_hu20_decisions import _node_data, _selected_views, _trace
from scripts.evaluate_hu20_reopening import Target
from scripts.run_exact_ranker_experiment import file_hash, guard
from scripts.validate_reverse_lbr_acceleration import SUITS, _PairedDeck
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import information_key
from src.blueprint.search import _sample_world
from src.diagnostics.cached_lbr import CachedLocalBestResponse, SharedProbabilityCache
from src.diagnostics.conditional_values import summarize, visible_fingerprint
from src.diagnostics.exact_ranker import RankedCachedLocalBestResponse, exact_seven_card
from src.diagnostics.posterior_audit_v2 import (
    DurableRows, HIGHER_ROOT, MAIN_ROOT, SELECTION_DIGEST, STABILITY_ROOTS,
    atomic_json, likelihood_seed, posterior_from_counts, stability_gate,
)
from src.diagnostics.posterior_worlds import conditional_row
from src.diagnostics.reverse_lbr import compatible_holdings, observed_lbr_actions
from src.diagnostics.robustness import LBRConfig, checkdown_payoffs
from src.game.observation import BoardDealt, replay
from src.game.types import Action, ActionKind

CONFIG = LBRConfig(4, 5)
SEEDS = (2026093001, 2026093002, 2026093003)


class ScientificStop(Exception):
    """A prospective validity gate stopped further inference."""


def load_inputs(args):
    selection = json.loads(args.selection.read_text())
    plan = json.loads(args.plan.read_text())
    if selection["selection_digest"] != SELECTION_DIGEST or digest(selection["selected"]) != SELECTION_DIGEST:
        raise ValueError("Original 24 selected coordinates changed")
    if len(selection["selected"]) != 24 or plan["parent_selection_digest"] != SELECTION_DIGEST:
        raise ValueError("Frozen sample/plan mismatch")
    if plan["rng_roots"] != {"main_likelihood": MAIN_ROOT, "stability": list(STABILITY_ROOTS),
                             "higher_count": HIGHER_ROOT, "conditional_worlds": 202610050131}:
        raise ValueError("Frozen roots changed")
    views = list(_selected_views(args.raw_dir, selection))
    if len(views) != 24:
        raise ValueError("Missing selected visible decision")
    inventory = {e["rank"]: len(compatible_holdings(v))*len(observed_lbr_actions(v)) for e, _, v in views}
    if sum(inventory.values()) != 58047 or sum(inventory[e["rank"]] for e in plan["stability_selected"]) != 14639:
        raise ValueError("Frozen main/stability likelihood inventory changed")
    if sum(inventory[rank] for rank in plan["suit_selected_ranks"]) != 12477:
        raise ValueError("Frozen coupled-suit likelihood inventory changed")
    specs = {s["seed"]: s for s in json.loads(args.models.read_text())
             if s.get("arm") == "B" and s.get("milestone") == 100000000}
    if tuple(sorted(specs)) != SEEDS:
        raise ValueError("Three retained B100M lineages required")
    return selection, plan, views, specs


def suit_view(view):
    card = lambda c: c[0] + SUITS[c[1]]
    history = tuple(replace(e, cards=tuple(map(card, e.cards))) if isinstance(e, BoardDealt) else e
                    for e in view.history)
    return replay(history, view.seat, tuple(map(card, view.hole_cards)))


def execution(source, cache, view, seed, ranked):
    cls = RankedCachedLocalBestResponse if ranked else CachedLocalBestResponse
    attacker = cls(source, seed, cache, CONFIG)
    before = perf_counter()
    action = attacker.choose_action(view)
    elapsed = perf_counter() - before
    telemetry = attacker.telemetry[-1]
    return {"action": repr(action), "requested": telemetry["requested_samples"],
            "completed": telemetry["samples"], "limited": not telemetry["completed"],
            "values_chips": telemetry["values_chips"],
            "weights_digest": digest(list(map(float, attacker.weights))),
            "rng_digest": digest(attacker.random.getstate()),
            "zero_likelihood": attacker.zero_likelihood, "seconds": elapsed}


def timer(args, selection, views, specs, check):
    result = {"status": "running", "prefixes": [], "checks": [], "differences": []}
    wanted = defaultdict(dict)
    for entry, _, view in views:
        wanted[entry["file"]][entry["line"]] = True
    recorded = {}
    for filename, lines in wanted.items():
        with gzip.open(args.raw_dir / filename, "rt") as handle:
            for number, payload in enumerate(handle, 1):
                if number in lines:
                    recorded[(filename, number)] = json.loads(payload)
    journal = DurableRows(args.root / "timer" / "rows", {"selection_digest": SELECTION_DIGEST,
                            "source": args.source_head, "protocol": "real-clock-v2"})
    try:
        for model_seed in SEEDS:
            check()
            source = Target(specs[model_seed])
            cache = SharedProbabilityCache(source)
            for entry, _, view in views:
                if entry["seed"] != model_seed:
                    continue
                row = recorded[(entry["file"], entry["line"])]
                prior = [item for item in row["actions"]
                         if item["index"] < entry["action_index"] and item["logical_player"] == 1]
                events = observed_lbr_actions(view)
                if len(prior) != len(events):
                    raise ValueError("Recorded/public attacker prefixes differ")
                for item, (event_index, prefix, observed) in zip(prior, events):
                    lbr = item.get("lbr")
                    status = bool(lbr and lbr["completed"] and lbr["samples"] == lbr["requested_samples"])
                    result["prefixes"].append({"rank": entry["rank"], "public_event_index": event_index,
                            "recorded_action_index": item["index"], "complete": status,
                            "requested": lbr.get("requested_samples") if lbr else None,
                            "completed": lbr.get("samples") if lbr else None})
                    if not status:
                        result["differences"].append(f"{entry['rank']}/{event_index}: recorded limited/missing attacker")
                    holdings = sorted(compatible_holdings(view), key=lambda pair:
                                      sha256(repr((entry["rank"], event_index, pair)).encode()).hexdigest())
                    for holding_index in (0, len(holdings)//2, len(holdings)-1):
                        pair = holdings[holding_index]
                        opponent_view = replay(prefix, 1-view.seat, pair)
                        for repetition in range(2):
                            check()
                            identifier = f"{entry['rank']}:{event_index}:{holding_index}:{repetition}"
                            if identifier not in journal.rows:
                                internal = stream_seed(MAIN_ROOT, "validation", "opponent", "timing-v2",
                                                       entry["rank"], event_index, pair, repetition)
                                original = execution(source, cache, opponent_view, internal, False)
                                candidate = execution(source, cache, opponent_view, internal, True)
                                differences = [key for key in ("action", "requested", "completed", "rng_digest",
                                               "weights_digest", "zero_likelihood") if original[key] != candidate[key]]
                                if len(original["values_chips"]) != len(candidate["values_chips"]) or any(
                                        abs(a-b) > 1e-10 for a, b in zip(original["values_chips"], candidate["values_chips"])):
                                    differences.append("values_chips")
                                if original["limited"] or candidate["limited"]:
                                    differences.append("limited")
                                journal.add({"id": identifier, "rank": entry["rank"],
                                    "public_event_index": event_index, "holding": list(pair), "seed": internal,
                                    "original": original, "candidate": candidate, "differences": differences,
                                    "status": "complete"})
                                journal.checkpoint()
                            record = journal.rows[identifier]
                            result["checks"].append(record)
                            if record["differences"]:
                                result["differences"].append({"id": identifier, "fields": record["differences"]})
                atomic_json(args.root / "timer" / "result.json", result)
            del source, cache
            gc.collect()
    finally:
        journal.close()
    result["status"] = "passed" if not result["differences"] else "stopped"
    result["check_count"] = len(result["checks"])
    atomic_json(args.root / "timer" / "result.json", result)
    if result["differences"]:
        raise ScientificStop("Real-clock/recorded attacker identity gate failed; values prohibited")


def estimate(args, entry, view, source, cache, root, samples, label, check, *, suit=False):
    path = args.root / "likelihood" / label / entry["rank"]
    path.mkdir(parents=True, exist_ok=True)
    if (path / "posterior.json").exists():
        return json.loads((path / "posterior.json").read_text())
    base_holdings = compatible_holdings(view)
    current = suit_view(view) if suit else view
    holdings = tuple(tuple(c[0] + SUITS[c[1]] for c in pair) for pair in base_holdings) if suit else base_holdings
    events = []
    for event_index, prefix, observed in observed_lbr_actions(current):
        check()
        metadata = {"rank": entry["rank"], "public_event_index": event_index,
            "public_prefix": [repr(e) for e in prefix], "observed_action": repr(observed),
            "root": root, "denominator": samples, "source": args.source_head,
            "model_sha256": source.source.description.get("sha256", specs_hash(source)),
            "holdings_digest": digest(holdings), "suit": suit}
        journal = DurableRows(path / f"event-{event_index:03d}", metadata)
        counts, limited = [], 0
        try:
            with _PairedDeck(suit):
                for holding_index, (pair, seed_pair) in enumerate(zip(holdings, base_holdings)):
                    check()
                    matched = 0
                    for sample in range(samples):
                        identifier = f"{holding_index}:{sample}"
                        if identifier not in journal.rows:
                            internal = likelihood_seed(root, entry["rank"], event_index, seed_pair, sample)
                            try:
                                observed_view = replay(prefix, 1-current.seat, pair)
                                record = execution(source, cache, observed_view, internal, True)
                                journal.add({"id": identifier, "holding": list(pair), "sample": sample,
                                    "seed": internal, "match": int(record["action"] == repr(observed)),
                                    "predicted_action": record["action"], "requested": record["requested"],
                                    "completed": record["completed"], "limited": record["limited"],
                                    "seconds": record["seconds"], "status": "complete"})
                            except Exception as exc:
                                journal.add({"id": identifier, "holding": list(pair), "sample": sample,
                                             "seed": internal, "status": "failed", "failure": repr(exc)})
                                journal.checkpoint()
                                raise
                        record = journal.rows[identifier]
                        if record["status"] != "complete":
                            raise ScientificStop("Retained failed likelihood ID; no blind retry")
                        matched += record["match"]
                        limited += int(record["limited"])
                        if record["limited"]:
                            journal.checkpoint()
                            raise ScientificStop("Simulated LBR limited; inference stopped")
                    counts.append(matched)
                    journal.sync()
                    if holding_index % 64 == 0:
                        journal.checkpoint()
                if len(journal.rows) != len(holdings) * samples:
                    raise ValueError("Unexpected extra likelihood IDs")
        finally:
            journal.close()
        event = {"public_event_index": event_index, "counts": counts, "denominator": samples,
                 "limited_samples": limited, "public_prefix_digest": digest([repr(e) for e in prefix]),
                 "observed_action": repr(observed)}
        atomic_json(path / f"event-{event_index:03d}-counts.json", event)
        events.append(event)
    result = posterior_from_counts(holdings, events, samples)
    result.update({"rank": entry["rank"], "root": root, "label": label,
                   "event_count": len(events), "likelihood_calls": len(holdings)*len(events)*samples,
                   "cache": cache.telemetry(), "suit": suit})
    atomic_json(path / "posterior.json", result)
    return result


def specs_hash(source):
    return digest(source.source.description)


def likelihood_stage(args, plan, views, specs, check):
    stability = {e["rank"] for e in plan["stability_selected"]}
    results = []
    for model_seed in SEEDS:
        check()
        source = Target(specs[model_seed])
        cache = SharedProbabilityCache(source)
        for entry, _, view in views:
            if entry["seed"] != model_seed or (args.phase == "stability" and entry["rank"] not in stability):
                continue
            if args.phase == "stability":
                four = [estimate(args, entry, view, source, cache, root, 4, f"repeat-{i+1}", check)
                        for i, root in enumerate(STABILITY_ROOTS)]
                higher = estimate(args, entry, view, source, cache, HIGHER_ROOT, 16, "higher-16", check)
                gate = stability_gate(four, higher)
                results.append({"selection": entry, **gate})
                atomic_json(args.root / "stability" / "result.json", {"status": "running", "cases": results})
            else:
                main = estimate(args, entry, view, source, cache, MAIN_ROOT, 4, "main-4", check)
                if main["status"] != "usable":
                    raise ScientificStop(f"Main posterior unusable at {entry['rank']}")
                if entry["rank"] in stability:
                    four = [json.loads((args.root / "likelihood" / f"repeat-{i+1}" / entry["rank"] / "posterior.json").read_text())
                            for i in range(3)]
                    higher = json.loads((args.root / "likelihood" / "higher-16" / entry["rank"] / "posterior.json").read_text())
                    gate = stability_gate([*four, main], higher)
                    results.append({"selection": entry, **gate})
                atomic_json(args.root / "main" / "result.json", {"status": "running", "cases": results})
        del source, cache
        gc.collect()
    passed = len(results) == 5 and all(row["passed"] for row in results)
    stage = "stability" if args.phase == "stability" else "main"
    atomic_json(args.root / stage / "result.json", {"status": "passed" if passed else "stopped", "cases": results})
    if not passed:
        raise ScientificStop(f"{stage} five-case posterior stability gate failed; primary values prohibited")


def suit_likelihood(args, plan, views, specs, check):
    results = []
    for model_seed in SEEDS:
        check()
        source = Target(specs[model_seed])
        cache = SharedProbabilityCache(source)
        for entry, _, view in views:
            if entry["seed"] != model_seed or entry["rank"] not in plan["suit_selected_ranks"]:
                continue
            original = json.loads((args.root / "likelihood" / "main-4" / entry["rank"] / "posterior.json").read_text())
            permuted = estimate(args, entry, view, source, cache, MAIN_ROOT, 4, "suit-main-4", check, suit=True)
            maximum = max(abs(a-b) for a, b in zip(original["weights"], permuted["weights"]))
            opts, probs, trained = source.distribution(view)
            alternative, other_probs, other_trained = source.distribution(suit_view(view))
            key = information_key(view, opts, schema=specs[model_seed]["abstraction"])
            other_key = information_key(suit_view(view), alternative, schema=specs[model_seed]["abstraction"])
            passed = (maximum <= 1e-10 and key == other_key and opts == alternative and
                      probs == other_probs and trained == other_trained and permuted["status"] == "usable")
            results.append({"rank": entry["rank"], "maximum_weight_difference": maximum,
                            "key_equal": key == other_key, "passed": passed})
        del source, cache
        gc.collect()
    atomic_json(args.root / "suit-likelihood" / "result.json", {"status": "passed" if all(r["passed"] for r in results) else "stopped",
                                                             "cases": results})
    if len(results) != 4 or not all(r["passed"] for r in results):
        raise ScientificStop("Coupled suit likelihood/policy control failed")


def require_gates(args):
    for phase in ("timer", "stability", "main", "suit-likelihood"):
        if json.loads((args.root / phase / "result.json").read_text())["status"] != "passed":
            raise ScientificStop(f"{phase} gate not passed; values prohibited")


def value_stage(args, plan, views, specs, check):
    require_gates(args)
    results, suit_controls = [], []
    for model_seed in SEEDS:
        check()
        source = Target(specs[model_seed])
        cache = SharedProbabilityCache(source)
        selected = [(e, i, v) for e, i, v in views if e["seed"] == model_seed]
        keys = {information_key(v, source.distribution(v)[0], schema=specs[model_seed]["abstraction"])
                for _, _, v in selected}
        nodes = _node_data(Path(specs[model_seed]["checkpoint_path"]), keys)
        for entry, item, view in selected:
            check()
            posterior = json.loads((args.root / "likelihood" / "main-4" / entry["rank"] / "posterior.json").read_text())
            holdings = tuple(tuple(pair) for pair in posterior["holdings"])
            weights = {"uniform": [1/len(holdings)]*len(holdings), "posterior": posterior["weights"]}
            menu, probabilities, trained = source.distribution(view)
            key = information_key(view, menu, schema=specs[model_seed]["abstraction"])
            if key != item["target_key"] or trained != entry["trained"] or nodes[key]["visits"] != entry["visits"]:
                raise ValueError("Saved selected lookup/update identity changed")
            record = {"selection": entry, "key": key, "node": nodes[key], "trained": trained,
                      "menu": [repr(c.action) for c in menu], "probabilities": list(probabilities),
                      "recorded_selected_action": repr(Action(ActionKind(item["kind"]), item["raise_to"])),
                      "hole_cards": list(view.hole_cards), "board": list(view.board),
                      "public_history": [repr(e) for e in view.history], "visible_fingerprint": visible_fingerprint(view),
                      "model_sha256": specs[model_seed]["sha256"],
                      "checkpoint_sha256": specs[model_seed]["checkpoint_sha256"], "ranges": {}}
            rows_by_range = {}
            for range_name in ("uniform", "posterior"):
                journal = DurableRows(args.root / "values" / entry["rank"] / range_name,
                                      {"rank": entry["rank"], "range": range_name, "source": args.source_head,
                                       "weights_digest": digest(weights[range_name]), "root": 202610050131})
                try:
                    for index in range(96):
                        check()
                        identifier = str(index)
                        if identifier not in journal.rows:
                            try:
                                row = conditional_row(view, source, cache, holdings, weights[range_name],
                                                      entry["rank"], index, check)
                                journal.add({"id": identifier, "index": index, "status": "complete", **row})
                            except Exception as exc:
                                journal.add({"id": identifier, "index": index, "status": "failed", "failure": repr(exc)})
                                journal.checkpoint()
                                raise
                            journal.checkpoint()
                        if journal.rows[identifier]["status"] != "complete":
                            raise ScientificStop("Retained failed world; no blind retry")
                    rows = [journal.rows[str(i)] for i in range(96)]
                    rows_by_range[range_name] = rows
                    record["ranges"][range_name] = summarize([r["values_bb"] for r in rows], probabilities)
                    record["ranges"][range_name]["limited_batches"] = sum(r["limited"] for r in rows)
                    if record["ranges"][range_name]["limited_batches"]:
                        raise ScientificStop("Continuation LBR limited; retain worlds and stop inference")
                finally:
                    journal.close()
            identity = not observed_lbr_actions(view)
            record["no_prior_action_identity_control"] = identity
            if identity and rows_by_range["uniform"] != rows_by_range["posterior"]:
                raise ScientificStop("Uniform/posterior preflop identity control failed")
            if entry["rank"] in plan["suit_selected_ranks"]:
                permuted_holdings = tuple(tuple(c[0] + SUITS[c[1]] for c in pair) for pair in holdings)
                differences = []
                for range_name in ("uniform", "posterior"):
                    journal = DurableRows(args.root / "suit-values" / entry["rank"] / range_name,
                                          {"rank": entry["rank"], "range": range_name, "source": args.source_head,
                                           "weights_digest": digest(weights[range_name]), "root": 202610050131})
                    try:
                        with _PairedDeck(True):
                            for index in range(96):
                                check()
                                if str(index) not in journal.rows:
                                    row = conditional_row(suit_view(view), source, cache, permuted_holdings,
                                                          weights[range_name], entry["rank"], index, check)
                                    journal.add({"id": str(index), "index": index, "status": "complete", **row})
                                    journal.checkpoint()
                                row = journal.rows[str(index)]
                                native = rows_by_range[range_name][index]
                                if row["menu"] != native["menu"] or row["probabilities"] != native["probabilities"]:
                                    differences.append(f"{range_name}/{index}: menu/probabilities")
                                if row["limited"] or any(abs(a-b)*view.big_blind > 1e-10
                                        for a, b in zip(row["values_bb"], native["values_bb"])):
                                    differences.append(f"{range_name}/{index}: return/limited")
                        suit_summary = summarize([journal.rows[str(i)]["values_bb"] for i in range(96)], probabilities)
                        if suit_summary["selected_action_index"] != record["ranges"][range_name]["selected_action_index"]:
                            differences.append(f"{range_name}: selected action")
                    finally:
                        journal.close()
                suit_controls.append({"rank": entry["rank"], "passed": not differences, "differences": differences})
                atomic_json(args.root / "suit-values" / "result.json", {"cases": suit_controls})
                if differences:
                    raise ScientificStop("Coupled suit conditional returns failed")
            atomic_json(args.root / "values" / entry["rank"] / "decision.json", record)
            results.append(record)
            atomic_json(args.root / "values" / "result.json", {"status": "running", "decisions": results})
        del source, cache
        gc.collect()
    atomic_json(args.root / "values" / "result.json", {"status": "complete", "decisions": results,
               "identity_controls": sum(r["no_prior_action_identity_control"] for r in results)})
    atomic_json(args.root / "suit-values" / "result.json", {"status": "passed", "cases": suit_controls})


def river_stage(args, plan, views, check):
    import numpy as np
    results = []
    ranks = {e["rank"] for e in plan["river_reference_selected"]}
    for entry, _, view in views:
        if entry["rank"] not in ranks:
            continue
        holdings = compatible_holdings(view)
        # Only terminal fold/call and matched checkdown plumbing, not strategy.
        actions = [Action(ActionKind.CALL)] if view.legal_actions.call_amount else [Action(ActionKind.CHECK)]
        if ActionKind.FOLD in view.legal_actions.kinds:
            actions.insert(0, Action(ActionKind.FOLD))
        journal = DurableRows(args.root / "river" / entry["rank"], {"rank": entry["rank"],
                  "source": args.source_head, "node_ceiling": 100000, "boundary": "fold/call/matched-checkdown"})
        nodes, maximum, ceiling = 0, 0.0, False
        try:
            for i, pair in enumerate(holdings):
                for j, action in enumerate(actions):
                    check()
                    identifier = f"{i}:{j}"
                    if identifier not in journal.rows:
                        needed = 1 + len([e for e in view.history if hasattr(e, "action")]) + 3
                        if nodes + needed > 100000:
                            ceiling = True
                            break
                        hand = _sample_world(view, {1-view.seat: ((pair, 1.0),)}, Random(0))
                        steps = 1 + len([e for e in view.history if hasattr(e, "action")])
                        hand = hand.apply(action)
                        steps += 1
                        while not hand.finished:
                            observation = hand.observe(hand.actor)
                            reply = Action(ActionKind.CALL if observation.legal_actions.call_amount else ActionKind.CHECK)
                            hand = hand.apply(reply)
                            steps += 1
                            if steps > needed:
                                raise ValueError("River reference unexpectedly needs strategic continuation")
                        a = exact_seven_card(view.hole_cards + view.board)
                        b = exact_seven_card(tuple(pair) + view.board)
                        ledger = float(checkdown_payoffs(view, action, np.asarray([(a>b)-(a<b)]))[0])
                        native = hand.events[-1].stacks[view.seat] - view.players[view.seat].starting_stack
                        if sum(hand.events[-1].stacks) != sum(hand.table.stacks):
                            raise ValueError("River native settlement is not zero-sum")
                        journal.add({"id": identifier, "holding": list(pair), "action": repr(action),
                                     "nodes": steps, "native_chips": native, "ledger_chips": ledger,
                                     "difference_chips": abs(native-ledger), "status": "complete"})
                        journal.sync()
                    row = journal.rows[identifier]
                    nodes += row["nodes"]
                    maximum = max(maximum, row["difference_chips"])
                if ceiling:
                    break
                if i % 64 == 0:
                    journal.checkpoint()
        finally:
            journal.close()
        results.append({"selection": entry, "nodes": nodes, "node_ceiling": 100000,
                        "holding_actions_completed": len(journal.rows), "maximum_difference_chips": maximum,
                        "status": "incomplete_ceiling" if ceiling else "passed" if maximum <= 1e-10 else "failed"})
    atomic_json(args.root / "river" / "result.json", {"cases": results,
                   "status": "passed" if len(results) == 3 and all(r["status"] == "passed" for r in results) else "limited_or_failed"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("timer", "stability", "main", "suit-likelihood", "values", "river"))
    for name in ("root", "selection", "plan", "raw-dir", "models"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    clock = json.loads((args.root / "scientific-clock.json").read_text())
    scientific_clock = dict(clock, deadline=clock["science_deadline"])
    last_guard = [0.0]
    def check():
        if time() >= scientific_clock["deadline"]:
            raise TimeoutError("Immutable nine-hour scientific cutoff; final 30 minutes reserved")
        if time()-last_guard[0] > 5:
            guard(args.root, scientific_clock)
            last_guard[0] = time()
    summary = {"phase": args.phase, "started": time(), "status": "running"}
    atomic_json(args.root / f"{args.phase}-attempt.json", summary)
    try:
        check()
        prerequisites = {"stability": "timer", "main": "stability", "suit-likelihood": "main"}
        if args.phase in prerequisites:
            required = prerequisites[args.phase]
            if json.loads((args.root / required / "result.json").read_text())["status"] != "passed":
                raise ScientificStop(f"{required} prerequisite failed; phase prohibited")
        selection, plan, views, specs = load_inputs(args)
        if args.phase == "timer":
            timer(args, selection, views, specs, check)
        elif args.phase in ("stability", "main"):
            likelihood_stage(args, plan, views, specs, check)
        elif args.phase == "suit-likelihood":
            suit_likelihood(args, plan, views, specs, check)
        elif args.phase == "values":
            value_stage(args, plan, views, specs, check)
        else:
            require_gates(args)
            river_stage(args, plan, views, check)
        summary["status"] = "complete"
    except ScientificStop as exc:
        summary.update(status="scientific_stop", reason=str(exc))
    except Exception as exc:
        summary.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    summary["finished"] = time()
    atomic_json(args.root / f"{args.phase}-attempt.json", summary)
    print(json.dumps(summary))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
