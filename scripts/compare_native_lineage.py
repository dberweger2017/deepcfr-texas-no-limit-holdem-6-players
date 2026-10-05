"""Check that a native HU20 checkpoint (and optionally its current-policy export) equals a Python one.

Compares the iteration and every table row (names, regrets, averages, visits) as parsed values,
so float text formatting doesn't matter; for exports, every entry's names and probabilities.
Resource limits in the config (max nodes, entries, seconds) are reported, not compared.
"""

import argparse
import gzip
import json
from pathlib import Path


def checkpoint(path):
    with gzip.open(path, "rt") as f:
        header = json.loads(f.readline())
        rows = {}
        for line in f:
            row = json.loads(line)
            rows[row[0]] = row[1:]
    return header, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("python", type=Path)
    p.add_argument("native", type=Path)
    p.add_argument("--python-current", type=Path)
    p.add_argument("--native-current", type=Path)
    a = p.parse_args()
    ph, prows = checkpoint(a.python)
    nh, nrows = checkpoint(a.native)
    semantic = ("seed", "raise_cap", "roots_per_seat", "abstraction", "game")
    result = {
        "iteration": [ph["iteration"], nh["iteration"]],
        "config_equal": all(ph["config"].get(k) == nh["config"].get(k) for k in semantic),
        "identity_equal": ph["identity"] == nh["identity"] and ph["table"] == nh["table"],
        "entries": [len(prows), len(nrows)],
        "rows_equal": prows == nrows,
    }
    if not result["rows_equal"]:
        result["differing_rows"] = sum(prows.get(k) != v for k, v in nrows.items()) + len(prows.keys() - nrows.keys())
    del prows, nrows
    if a.python_current and a.native_current:
        pc = json.loads(gzip.open(a.python_current, "rt").read())
        nc = json.loads(gzip.open(a.native_current, "rt").read())
        result["current_entries_equal"] = pc["entries"] == nc["entries"]
        result["current_description_equal"] = {k: pc[k] == nc.get(k) for k in pc if k not in ("entries", "config")}
    result["identical"] = (result["iteration"][0] == result["iteration"][1] and result["config_equal"]
                           and result["identity_equal"] and result["rows_equal"]
                           and result.get("current_entries_equal", True))
    print(json.dumps(result))
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
