"""Compare three same-seed HU20 training checkpoints key by key, as exported policies would play.

T (traverser-reach average) and O (opponent-sampled average) come from lockstep runs: their
regrets and visits must be identical, so any difference in play is the averaging rule. F adds
the CFR+ regret floor. Checkpoint rows are sorted by key, so the three files are merge-joined
as streams.

Situations come from the stored menu: facing a jam is exactly `fold, call`; facing a bet or
raise has `fold` and more; an unopened decision starts with `check`. Each table cell is a
situation and a visit band (by T's visits); exported policies use the export rule, so a
zero-mass average plays uniformly.
"""

import argparse
from collections import defaultdict
import gzip
import json
from math import fsum
from pathlib import Path

BANDS = ((1, 1), (2, 9), (10, 99), (100, 999), (1000, None))


def rows(path):
    with gzip.open(path, "rt") as stream:
        header = json.loads(stream.readline())
        yield header
        for line in stream:
            key, names, regrets, average, visits = json.loads(line)
            yield key, tuple(names), regrets, average, visits


def situation(names):
    if names == ("fold", "call"):
        return "facing-jam"
    if names[0] == "fold":
        return "facing-bet"
    if names[0] == "check":
        return "unopened"
    return "other"


def band(visits):
    for low, high in BANDS:
        if visits >= low and (high is None or visits <= high):
            return f"{low}+" if high is None else (str(low) if low == high else f"{low}-{high}")
    return "0"


def exported_average(average):
    mass = fsum(average)
    return [v / mass for v in average] if mass > 0 else [1 / len(average)] * len(average), mass == 0


def current(regrets):
    positive = [max(r, 0.0) for r in regrets]
    total = fsum(positive)
    return [p / total for p in positive] if total > 0 else [1 / len(regrets)] * len(regrets), total == 0


def distance(p, q):
    return 0.5 * fsum(abs(a - b) for a, b in zip(p, q))


def merged(*paths):
    """(key, row-or-None per checkpoint) in key order."""
    streams = [rows(p) for p in paths]
    headers = [next(s) for s in streams]
    heads = [next(s, None) for s in streams]
    yield headers
    while any(h is not None for h in heads):
        key = min(h[0] for h in heads if h is not None)
        yield key, [h if h is not None and h[0] == key else None for h in heads]
        heads = [next(s, None) if h is not None and h[0] == key else h for s, h in zip(streams, heads)]


class Cell:
    def __init__(self):
        self.n = defaultdict(float)

    def add(self, name, value, weight):
        self.n[name + ":sum"] += value
        self.n[name + ":wsum"] += value * weight
        self.n[name + ":count"] += 1
        self.n[name + ":weight"] += weight

    def mean(self, name, weighted=False):
        count = self.n[name + (":weight" if weighted else ":count")]
        return self.n[name + (":wsum" if weighted else ":sum")] / count if count else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traverser", type=Path, required=True, help="T: traverser-reach average checkpoint")
    p.add_argument("--opponent", type=Path, required=True, help="O: opponent-sampled lockstep partner of T")
    p.add_argument("--floor", type=Path, required=True, help="F: regret-floor checkpoint, same seed")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    stream = merged(a.traverser, a.opponent, a.floor)
    headers = next(stream)
    if headers[1].get("average_rule") != "opponent-sampled" or "average_rule" in headers[0]:
        raise ValueError("Expected T (traverser-reach) then O (opponent-sampled)")
    if headers[0]["iteration"] != headers[1]["iteration"] or headers[0]["config"] != headers[1]["config"]:
        raise ValueError("T and O are not lockstep partners")
    cells = defaultdict(Cell)
    keys = defaultdict(int)
    for key, (t, o, f) in stream:
        keys[("T" if t else "") + ("O" if o else "") + ("F" if f else "")] += 1
        if t is not None and (o is None or t[1] != o[1] or t[2] != o[2] or t[4] != o[4]):
            raise ValueError(f"T and O regrets/visits differ at {key}: the runs are not lockstep")
        if t is None and o is not None and (o[4] != 0 or any(o[2])):
            raise ValueError(f"O-only key {key} has traverser visits or regrets")
        base = t or o or f
        # O also stores keys seen only as the sampled opponent; T has no entry there and plays uniformly.
        where = (situation(base[1]), band(t[4]) if t else ("T-missing" if o else "F-only"))
        cell = cells[where]
        if o is not None:
            weight = max(t[4] if t else 0, 1)
            pt, zt = exported_average(t[3]) if t else ([1 / len(o[1])] * len(o[1]), True)
            po, zo = exported_average(o[3])
            cell.add("visits", t[4] if t else 0, 1)
            cell.add("T zero-mass", zt, weight)
            cell.add("O zero-mass", zo, weight)
            if "fold" in o[1]:
                cell.add("T fold", pt[0], weight)
                cell.add("O fold", po[0], weight)
            cell.add("O-T distance", distance(po, pt), weight)
        if f is not None:
            weight = max(f[4], 1)
            pf, zf = exported_average(f[3])
            cf, uniform = current(f[2])
            cell.add("F zero-mass", zf, weight)
            cell.add("F all-zero regrets", uniform, weight)
            if "fold" in f[1]:
                cell.add("F fold", pf[0], weight)
                cell.add("F current fold", cf[0], weight)
            pt = exported_average(t[3])[0] if t else [1 / len(f[1])] * len(f[1])
            cell.add("F-T distance", distance(pf, pt), weight)
    order = {s: i for i, s in enumerate(("facing-jam", "facing-bet", "unopened", "other"))}
    bands = ["T-missing"] + [band(low) for low, _ in BANDS] + ["F-only"]
    table = []
    for (where, visits), cell in sorted(cells.items(), key=lambda kv: (order[kv[0][0]], bands.index(kv[0][1]))):
        row = {"situation": where, "T visits": visits, "keys T/O": int(cell.n["visits:count"]),
               "keys F": int(cell.n["F zero-mass:count"]), "visit share T": cell.n["visits:sum"]}
        for name in ("T zero-mass", "O zero-mass", "F zero-mass", "F all-zero regrets", "T fold", "O fold",
                     "F fold", "F current fold", "O-T distance", "F-T distance"):
            row[name] = cell.mean(name)
            row[name + " (visit-weighted)"] = cell.mean(name, weighted=True)
        table.append(row)
    total = fsum(r["visit share T"] for r in table)
    for r in table:
        r["visit share T"] = r["visit share T"] / total if total else None
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {"checkpoints": {"T": str(a.traverser), "O": str(a.opponent), "F": str(a.floor)},
               "iterations": {"T": headers[0]["iteration"], "O": headers[1]["iteration"], "F": headers[2]["iteration"]},
               "F options": headers[2].get("training_options"), "key presence": dict(keys), "cells": table,
               "rule": "exported average = normalized stored average, uniform when its mass is zero"}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    show = ("keys T/O", "keys F", "visit share T", "T zero-mass", "O zero-mass", "F zero-mass", "F all-zero regrets",
            "T fold", "O fold", "F fold", "F current fold", "O-T distance", "F-T distance")
    fmt = lambda v: "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| situation | T visits | " + " | ".join(show) + " |", "|" + "---|" * (len(show) + 2)]
    lines += [f"| {r['situation']} | {r['T visits']} | " + " | ".join(fmt(r[s]) for s in show) + " |" for r in table]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"iterations": summary["iterations"], "key presence": summary["key presence"]}))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
