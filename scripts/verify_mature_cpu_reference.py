"""Independent streaming verification of a closed M4 reference archive."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import time


def verify(root, out):
    clock = json.loads((root / "clock.json").read_text())
    def deadline():
        if time.time() >= clock["deadline"]:
            raise TimeoutError("Original M4 reference deadline")
    def digest(path, compressed=False):
        h = sha256()
        opener = gzip.open if compressed else open
        with opener(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                deadline()
                h.update(chunk)
        return h.hexdigest()
    manifest_path = root.with_name(root.name + "-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert actual == set(manifest), "Archive membership changed"
    for name, record in manifest.items():
        path = root / name
        assert path.stat().st_size == record["bytes"]
        assert digest(path) == record["sha256"], name
    direct = json.loads((root / "direct/result.json").read_text())
    resumed = json.loads((root / "resumed/result.json").read_text())
    assert direct["status"] == resumed["status"] == "complete"
    assert digest(Path(direct["parent"])) == direct["parent_fingerprint"]["sha256"]
    for name in ("final", "current", "next"):
        assert direct[name] == resumed[name]
        for phase in ("direct", "resumed"):
            path = root / phase / (name + ".json.gz")
            assert digest(path) == direct[name]["sha256"]
            assert digest(path, True) == direct[name]["uncompressed_sha256"]
    rows = [json.loads(line) for line in (root / "direct/iterations.jsonl").read_text().splitlines()]
    suffix = [json.loads(line) for line in (root / "resumed/iterations.jsonl").read_text().splitlines()]
    assert suffix == [r for r in rows if r["added_nodes"] > resumed["resume_added_nodes"]]
    assert sum(r["nodes"] for r in rows) == direct["added_nodes"]
    assert len(rows) == direct["iteration"] - direct["origin_iteration"]
    assert sum(r["new_entries"] for r in rows) == direct["new_entries"]
    assert all(r["nodes"] == r["attempted_work"]["nodes"] for r in rows)
    assert direct["next_streams"] == resumed["next_streams"]
    state = json.loads((root / "supervisor/campaign.json").read_text())
    assert state["status"] == "complete" and len(state["attempts"]) == 3
    assert all(a["status"] == "complete" and a["guard_failure"] is None for a in state["attempts"])
    resources = [json.loads(line) for line in (root / "supervisor/resources.jsonl").read_text().splitlines()]
    assert all("AC Power" in r["power"] for r in resources)
    peak = max(r["aggregate_job_rss_bytes"] for r in resources)
    swap = max(r["swap_growth_bytes"] for r in resources)
    free = min(r["free_disk_bytes"] for r in resources)
    assert peak < 10.5 * 2**30 and swap <= .5 * 2**30 and free >= 8 * 2**30
    deadline()
    result = {"passed": True, "finished": time.time(), "deadline": clock["deadline"],
              "manifest_sha256": digest(manifest_path), "verified_files": len(manifest),
              "complete_iterations": len(rows), "resumed_suffix_iterations": len(suffix),
              "added_nodes": direct["added_nodes"], "new_entries": direct["new_entries"],
              "peak_sampled_aggregate_rss_bytes": peak, "max_swap_growth_bytes": swap,
              "minimum_free_disk_bytes": free, "all_power_samples_ac": True,
              "all_payloads_and_artifact_bytes_match": True, "original_parent_unchanged": True,
              "discarded_work": 0, "source": clock["runtime_source"],
              "verification_script_sha256": digest(Path(__file__))}
    out.write_text(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.out)
