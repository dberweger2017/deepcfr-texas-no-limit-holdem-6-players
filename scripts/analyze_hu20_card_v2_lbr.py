"""Post-hoc LBR partitions from committed generated hands; no model loads."""
import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path


STREETS = ("preflop", "flop", "turn", "river")


def accumulate(rows):
    arms = defaultdict(lambda: {"hands": 0, "chips": 0,
        "last_betting_street": defaultdict(Counter),
        "minimum_target_late_visits": defaultdict(Counter),
        "decision_visits": defaultdict(Counter)})
    seen = set()
    for row in rows:
        if row["panel"] != "lbr":
            continue
        if row["status"] != "complete" or not row.get("native_replay_verified"):
            raise ValueError("Incomplete or unreplayed LBR record")
        coordinate = (row["seed"], row["version"], row["block"], row["rotation"])
        if coordinate in seen:
            raise ValueError("Duplicate LBR coordinate")
        seen.add(coordinate)
        arm = arms[row["version"]]
        chips = row["target_chips"]
        arm["hands"] += 1
        arm["chips"] += chips
        # An all-in may run out later board cards. This describes when betting
        # stopped, not terminal board street or the location of an error.
        street = row["actions"][-1]["street"]
        if street not in STREETS:
            raise ValueError("Unknown betting street")
        arm["last_betting_street"][street].update(hands=1, chips=chips)
        late = []
        for action in row["actions"]:
            if action["logical_player"] != 0:
                continue
            visits = action["observation"]["visits"]
            if type(visits) is not int or visits < 0:
                raise ValueError("Invalid target visit count")
            band = "0" if visits == 0 else "1-9" if visits < 10 else "10-99" if visits < 100 else "100+"
            arm["decision_visits"][action["street"]][band] += 1
            if action["street"] in ("turn", "river"):
                late.append(visits)
        group = "no_target_late_decision" if not late else "under_10" if min(late) < 10 else "10-99" if min(late) < 100 else "100+"
        arm["minimum_target_late_visits"][group].update(hands=1, chips=chips)
    return json.loads(json.dumps(arms))


def analyze(root):
    paths = sorted(root.glob("*/evaluation/hands.jsonl.gz"))
    if not paths:
        raise ValueError("No published hands found")

    def rows():
        for path in paths:
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    yield json.loads(line)

    return {"post_hoc": True, "model_loads": 0,
        "analysis_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "definition": "Whole-hand returns grouped by last recorded betting action or minimum target turn/river visits; not action EV or causal street attribution",
        "inputs": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in paths},
        "arms": accumulate(rows())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
