"""Freeze outcome-blind native-LBR equivalence cases before optimization."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path

from scripts.diagnose_hu20_decisions import _trace
from scripts.evaluate_hu20 import write_json
from src.arena.schedule import digest, stream_seed
from src.blueprint.search import DECK
from src.game.observation import replay

ROOT = 202610030101
SALT = "reverse-lbr-corpus-v1"


def _holding_rank(selected_rank, action_index, pair):
    return sha256(f"{SALT}|{selected_rank}|{action_index}|{pair}".encode()).hexdigest()


def freeze(raw_dir, selection_path, output):
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads(selection_path.read_text())
    if digest(selection["selected"]) != selection["selection_digest"]:
        raise ValueError("#119 selected-list digest changed")
    by_file = defaultdict(lambda: defaultdict(list))
    for entry in selection["selected"]:
        by_file[entry["file"]][entry["line"]].append(entry)
    cases = []
    coverage = Counter()
    for filename, wanted in sorted(by_file.items()):
        path = raw_dir / filename
        if sha256(path.read_bytes()).hexdigest() != selection["source_hashes"][filename]:
            raise ValueError("#117 raw-hand hash changed")
        with gzip.open(path, "rt") as handle:
            for line_number, line in enumerate(handle, 1):
                if line_number not in wanted:
                    continue
                row = json.loads(line)
                traced = list(_trace(row))
                for selected in wanted[line_number]:
                    target = [(item, view) for item, view in traced
                              if item["index"] == selected["action_index"]]
                    if len(target) != 1 or target[0][0]["logical_player"] != 0:
                        raise ValueError("Selected target decision does not replay")
                    target_view = target[0][1]
                    attacker = [(item, view) for item, view in traced
                                if item["logical_player"] == 1]
                    prior = [(item, view) for item, view in attacker
                             if item["index"] < selected["action_index"]]
                    subsequent = [(item, view) for item, view in attacker
                                  if item["index"] > selected["action_index"]]
                    picked = prior[:1] + prior[-1:] + subsequent[:1]
                    seen = set()
                    for item, attacker_view in picked:
                        if item["index"] in seen:
                            continue
                        seen.add(item["index"])
                        prefix = attacker_view.history
                        target_at_prefix = replay(prefix, target_view.seat, target_view.hole_cards)
                        visible = set(target_at_prefix.hole_cards + target_at_prefix.board)
                        holdings = sorted(
                            combinations((card for card in DECK if card not in visible), 2),
                            key=lambda pair: _holding_rank(selected["rank"], item["index"], pair),
                        )
                        indices = sorted({0, len(holdings)//2, len(holdings)-1})
                        for index in indices:
                            pair = holdings[index]
                            for sample in (0, 1):
                                internal_seed = stream_seed(ROOT, "validation", "opponent",
                                                            "equivalence", selected["rank"],
                                                            item["index"], pair, sample)
                                body = {"selected": {k:selected[k] for k in
                                         ("seed", "file", "line", "block", "rotation",
                                          "action_index", "street", "position", "rank")},
                                        "attacker_action_index": item["index"],
                                        "attacker_action_kind": item["kind"],
                                        "prefix_digest": digest([repr(event) for event in prefix]),
                                        "target_cards": target_view.hole_cards,
                                        "public_board_at_prefix": target_at_prefix.board,
                                        "attacker_holding": pair,
                                        "internal_seed": internal_seed,
                                        "sample_index": sample,
                                        "holding_rank_position": index,
                                        "suit_control": "cyclic c->d->h->s->c, paired deck order"}
                                body["case_id"] = digest(body)
                                cases.append(body)
                                coverage[(selected["seed"], selected["street"], selected["position"])] += 1
    if not cases:
        raise ValueError("No attacker prefixes in fixed selected decisions")
    report = {"schema": "reverse-lbr-equivalence-corpus-v1",
              "selection_digest": selection["selection_digest"],
              "source_hashes": selection["source_hashes"],
              "root": ROOT,
              "case_rule": "first/last prior attacker decision and first subsequent after each selected target decision; three hash-ranked compatible holdings; two RNG seeds",
              "suit_permutation": {"c":"d", "d":"h", "h":"s", "s":"c"},
              "value_absolute_tolerance_chips": 1e-10,
              "required_exact_fields": ["action including raise_to", "menu order",
                  "requested/completed samples", "zero-likelihood events",
                  "compatible/positive support", "repeat determinism"],
              "synthetic_fixtures": ["preflop-open", "preflop-facing-minraise",
                                      "river-check", "river-facing-minraise",
                                      "zero-evidence-Bayes", "first-index-tie"],
              "cases": cases, "case_digest": digest(cases),
              "coverage": [{"seed": seed, "street": street, "position": position, "cases": count}
                           for (seed, street, position), count in sorted(coverage.items())],
              "case_count": len(cases)}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return {"case_count": len(cases), "case_digest": report["case_digest"],
            "covered_cells": len(coverage)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.raw_dir, args.selection, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
