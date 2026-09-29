"""Explicit immutable input validation for deployed scaling workers."""

import gzip
import json
from pathlib import Path

from src.blueprint.windowed import _hash


def validate_fixture(plan):
    path = Path(plan["independent_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Required independent fixture: {path}")
    if _hash(path) != plan["independent_sha256"]:
        raise ValueError("Independent fixture hash mismatch")


def validate_inputs(plan, *, parse=False):
    """Fail before collection; all paths in recovery plans are absolute."""
    validate_fixture(plan)
    verified = []
    for entry in plan.get("runtime_inputs", []):
        path = Path(entry["path"])
        if not path.is_absolute():
            raise ValueError("Runtime input requires an explicit absolute path")
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or wrong-size runtime input: {path}")
        if _hash(path) != entry["sha256"]:
            raise ValueError(f"Runtime input hash mismatch: {path}")
        if parse and entry.get("kind") in ("json", "gzip-json", "gzip-jsonl"):
            opener = gzip.open if entry["kind"].startswith("gzip") else open
            with opener(path, "rt") as saved:
                if entry["kind"] == "gzip-jsonl":
                    count = 0
                    for line in saved:
                        row = json.loads(line)
                        if entry.get("schema") == "independent-observations":
                            required = {"actions", "seat", "seed", "button", "observation_sha256", "hand_id", "path"}
                            if not required <= row.keys():
                                raise ValueError("Independent observation schema")
                        if entry.get("schema") == "training-checkpoint":
                            if count == 0:
                                if row.get("kind") != "training" or row.get("checkpoint_format") != "jsonl-v2":
                                    raise ValueError("Training checkpoint header")
                            elif not isinstance(row, list) or len(row) != 5:
                                raise ValueError("Training checkpoint row")
                        count += 1
                    if "rows" in entry and count != entry["rows"]:
                        raise ValueError("Runtime fixture row count")
                else:
                    row = json.load(saved)
                    if entry.get("required_keys") and not set(entry["required_keys"]) <= row.keys():
                        raise ValueError(f"Runtime input schema: {path}")
        verified.append({"path": str(path), "sha256": entry["sha256"], "bytes": entry["bytes"]})
    return verified
