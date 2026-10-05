"""Require exact scientific response parity for an external engineering build."""

import argparse
import hashlib
import json
from pathlib import Path

from src.diagnostics.flop_check import atomic_json
from src.diagnostics.saved_hu20 import file_hash


# Only native telemetry may differ. Nested fields and every other field count.
TELEMETRY = {"elapsed_seconds", "solver_peak_rss_bytes"}


def scientific_fingerprint(path):
    digest = hashlib.sha256()
    events = 0
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            scientific = {k: v for k, v in row.items() if k not in TELEMETRY}
            encoded = json.dumps(scientific, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()
            digest.update(encoded + b"\n")
            events += 1
    return {"events": events, "scientific_sha256": digest.hexdigest(),
            "response_sha256": file_hash(path)}


def compare(reference, actual):
    old, new = scientific_fingerprint(reference), scientific_fingerprint(actual)
    if (old["events"], old["scientific_sha256"]) != (new["events"], new["scientific_sha256"]):
        raise ValueError("Scientific responses differ; no tolerance or partial reuse is admitted")
    return {"passed": True, "reference": str(Path(reference).resolve()),
            "actual": str(Path(actual).resolve()), "reference_fingerprint": old,
            "actual_fingerprint": new, "ignored_top_level_telemetry": sorted(TELEMETRY),
            "comparison": "Exact canonical JSON; float values and signed zero retained"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "actual", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    atomic_json(a.out, compare(a.reference, a.actual))


if __name__ == "__main__":
    main()
