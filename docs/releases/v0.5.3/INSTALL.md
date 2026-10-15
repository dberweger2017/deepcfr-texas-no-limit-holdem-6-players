# Install and run v0.5.3 (unpublished review bundle)

Verify the complete asset set and source binding before loading. The owner
separately authorizes publication; this bundle carries no publication approval.
Use Python 3.11 with requirements-play.txt at source PACKAGE_SOURCE_COMMIT.

Download all declared assets into one directory. The model consists of three
ordered parts below 2 GiB each; flop, turn and river K50 tables are separate assets.
The verifier uses only Python's standard library and checks the pinned whole model.

```sh
python3.11 -I -S models/v053/verify_hu100_bundle.py models/v053 \
  --expect-source PACKAGE_SOURCE_COMMIT \
  --assemble models/v053-model/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz
```

Keep the assembled model outside the immutable asset directory. The runtime
verifies all assets before loading and uses translation 512/128 as part of the
session identity. The downloaded payload is about 4.58 GB; the assembled model adds 3.20 GB.
Uncached JSON-average loads take minutes. Actual M1 load time, peak RSS and
verification receipts are in [PR298’s readiness record](https://github.com/dberweger2017/CFR-NLHE-6max/pull/298).

```sh
.venv/bin/python -m src.play_api.server --hu100-release models/v053 \
  --data-dir results/hu100-play --source-version v0.5.3
```

Open the printed local URL and use the token from the private data directory.
Choose human free/restricted sizing or self-play spectator. Other depths are
rejected. Sessions retain their exact model and inference identity across restart.
Independent journal verification:

```sh
.venv/bin/python -m src.play_api.server --hu100-release models/v053 \
  --data-dir results/hu100-play --verify-session SESSION_ID
```

There is no publication approval or release tag for this preparation. Older
v0.5.0–v0.5.2 bundles retain their manifests; HU20's default remains v0.4.2.
