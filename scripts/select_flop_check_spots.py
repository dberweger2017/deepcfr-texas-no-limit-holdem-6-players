"""Outcome-blind Set A extraction and stratified fresh self-play root selection."""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random

from src.diagnostics.flop_check import atomic_json
from src.diagnostics.saved_hu20 import file_hash
from src.arena.endgame_quality import _world
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken, BoardDealt, replay
from src.game.types import Action, ActionKind, Street
from scripts.prepare_flop_check import load_policy

DEAL_SEED = 202610020001
ACTION_SEED = 202610020002
SELECTION_SEED = 202610020003


def root_record(history):
    view = replay(history, 0, ())
    if view.finished or view.street != Street.FLOP:
        raise ValueError("Need a live public flop root")
    line = [{"position": (e.seat - view.button) % 2,
             "kind": e.action.kind.value, "raise_to": e.action.raise_to}
            for e in history if isinstance(e, ActionTaken) and e.street == Street.PREFLOP]
    raises = [e for e in line if e["kind"] == "raise"]
    kind = ("limped" if not raises else "3-bet" if len(raises) >= 2
            else "min-raised" if raises[0]["raise_to"] == 200 else "pot-raised")
    public = {"board": sorted(view.board), "preflop_line": line,
              "pot": view.pot, "stacks_by_position": [view.players[(view.button + i) % 2].stack
                                                        for i in (0, 1)]}
    identity = sha256(json.dumps(public, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(public, spot=identity, kind=kind, button=view.button)


def replay_root(record):
    """Reconstruct betting from public actions and replace only the public flop."""
    hand = Hand.start(Table(("a", "b"), (2000, 2000), button=record["button"]),
                      hand_id="flop-selection", seed=0)
    for item in record["preflop_line"]:
        if (hand.actor - record["button"]) % 2 != item["position"]:
            raise ValueError("Public line actor differs")
        hand = hand.apply(Action(ActionKind(item["kind"]), item["raise_to"]))
    events = tuple(replace(e, cards=tuple(record["board"]))
                   if isinstance(e, BoardDealt) and e.street == Street.FLOP else e
                   for e in hand.events)
    root = _world(events, tuple(record["board"]), {}).events
    if root_record(root)["spot"] != record["spot"]:
        raise ValueError("Public root does not replay exactly")
    return root


def stored_set_a(plan, repo):
    roots = {}; decisions = []; counts = Counter(); opponent = defaultdict(Counter)
    sources = []
    for relative, expected in sorted(plan["stored_hands"].items()):
        path = Path(repo) / "docs/reports/hu20-card-v2-artifacts/production" / relative
        actual = file_hash(path)
        if actual != expected:
            raise ValueError("Stored hands SHA-256 differs before use")
        sources.append({"path": str(path), "sha256": actual})
        with gzip.open(path, "rt") as source:
            for number, line in enumerate(source, 1):
                row = json.loads(line)
                if row["panel"] != "lbr" or row["version"] != "v1":
                    continue
                counts["lbr_v1_hands"] += 1
                hand = Hand.start(Table(("a", "b"), (2000, 2000), button=row["button"]),
                                  hand_id="flop-selection", seed=row["deal_seed"])
                root = None
                for action in row["actions"]:
                    if action["street"] != "preflop":
                        break
                    if hand.actor != action["seat"]:
                        raise ValueError("Stored preflop actor differs from native replay")
                    observed = hand.observe(hand.actor)
                    if tuple(action["observation"]["hole_cards"]) != observed.hole_cards:
                        raise ValueError("Stored preflop holding differs from native deal")
                    hand = hand.apply(Action(ActionKind(action["kind"]), action["raise_to"]))
                    if not hand.finished and hand.observe(hand.actor).street == Street.FLOP:
                        root = root_record(hand.events)
                if root is None:
                    if len(hand.observe(0).board) >= 3:
                        counts["hands_dealt_flop"] += 1
                        counts["preflop_all_in_showdowns"] += 1
                    continue
                counts["hands_with_live_flop_root"] += 1
                counts["hands_dealt_flop"] += 1
                for action in row["actions"]:
                    if action["street"] == "flop" and action["observation"]["board"] != list(
                            replay(hand.events, 0, ()).board):
                        raise ValueError("Stored flop differs from native replay")
                line_id = json.dumps(root["preflop_line"], sort_keys=True)
                first_opp = next((a for a in row["actions"] if a["logical_player"] == 1), None)
                if first_opp is not None:
                    position = (first_opp["seat"] - row["button"]) % 2
                    holding = tuple(sorted(first_opp["observation"]["hole_cards"]))
                    opponent[(line_id, position)][holding] += 1
                selected = [a for a in row["actions"] if a["street"] == "flop"
                            and a["logical_player"] == 0 and a["observation"]["call_amount"] > 0]
                if selected:
                    roots.setdefault(root["spot"], dict(root, multiplicity=0))
                    roots[root["spot"]]["multiplicity"] += 1
                for action in selected:
                    decisions.append({"spot": root["spot"], "source_sha256": actual,
                                      "source_line": number, "action_index": action["index"],
                                      "target_position": (action["seat"] - row["button"]) % 2,
                                      "public_context_id": action["observation"]["public_context_id"]})
    if len(decisions) != 273:
        raise ValueError(f"Set A differs from pinned 273 decisions: {len(decisions)}")
    ranges = [{"preflop_line": json.loads(line), "opponent_position": pos,
               "samples": sum(hist.values()), "raw_hand_counts": [
                   {"hand": list(h), "count": n} for h, n in sorted(hist.items())]}
              for (line, pos), hist in sorted(opponent.items())]
    return {"set": "A", "roots": list(roots.values()), "decisions": decisions,
            "counts": dict(counts), "sources": sources, "empirical_lbr_ranges": ranges,
            "flop_reach_definition": "live flop decision root; preflop all-in terminals excluded",
            "selection_uses_payoffs": False}


def select_strata(population, n, seed):
    groups = defaultdict(list)
    for root in population:
        groups[(root["kind"], root["button"])].append(root)
    keys = sorted(groups); sizes = {k: len(groups[k]) for k in keys}
    if sum(sizes.values()) < n:
        raise ValueError("Fresh corpus has fewer unique roots than requested")
    allocation = {k: min(sizes[k], n // len(keys)) for k in keys}
    while sum(allocation.values()) < n:
        candidates = [k for k in keys if allocation[k] < sizes[k]]
        key = max(candidates, key=lambda k: (sizes[k] / (allocation[k] + 1), k))
        allocation[key] += 1
    rng = Random(seed); selected = []
    for key in keys:
        shuffled = sorted(groups[key], key=lambda r: r["spot"]); rng.shuffle(shuffled)
        for root in shuffled[:allocation[key]]:
            selected.append(dict(root, reach_weight=root["multiplicity"] * sizes[key] / allocation[key],
                                 inclusion_probability=allocation[key] / sizes[key]))
    return selected, {f"{k[0]}/button{k[1]}": {"population_unique": sizes[k],
             "selected": allocation[k]} for k in keys}


def self_play_set_b(source, *, deals=10_000, n=150):
    deal_rng = Random(DEAL_SEED); action_rng = Random(ACTION_SEED); roots = {}
    counts = Counter()
    for index in range(deals):
        seed = deal_rng.getrandbits(64); button = index % 2
        hand = Hand.start(Table(("a", "b"), (2000, 2000), button=button),
                          hand_id="flop-selection", seed=seed)
        while not hand.finished and hand.observe(hand.actor).street == Street.PREFLOP:
            menu, p, _ = source.distribution(hand.observe(hand.actor))
            hand = hand.apply(action_rng.choices(menu, weights=p, k=1)[0].action)
        if hand.finished:
            counts["preflop_terminal"] += 1; continue
        root = root_record(hand.events)
        if root["spot"] not in roots:
            roots[root["spot"]] = dict(root, deal_seed=seed, multiplicity=0)
        roots[root["spot"]]["multiplicity"] += 1
        counts["live_flop_roots"] += 1
    selected, strata = select_strata(list(roots.values()), n, SELECTION_SEED)
    return {"set": "B", "roots": selected, "corpus_deals": deals, "counts": dict(counts),
            "corpus_unique_roots": len(roots), "strata": strata,
            "deal_seed": DEAL_SEED, "action_seed": ACTION_SEED,
            "selection_seed": SELECTION_SEED, "source": source.description,
            "target_positions": ["OOP", "IP"], "selection_uses_payoffs": False,
            "weights": "multiplicity / inclusion probability; common corpus for all policies",
            "postflop_actions_sampled": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", choices=("A", "B"), required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    if args.set == "A":
        result = stored_set_a(plan, args.repo)
    else:
        if not args.inputs:
            parser.error("Set B requires --inputs")
        result = self_play_set_b(load_policy(plan["policies"][0], args.inputs))
    atomic_json(args.out, result)


if __name__ == "__main__":
    main()
