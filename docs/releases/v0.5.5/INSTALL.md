# Install and run v0.5.5

To play, use Python 3.11, Git and a Rust toolchain for the pinned engine's source
build, on a 16 GB machine. The blend was built and tested on a base
16 GB MacBook Pro M4; #306 measured 12.75 GB peak family RSS for its research
pipeline. First start verifies the bundle and builds a binary play cache by
streaming all 156,208,331 rows with `scripts/build_streaming_average_cache.py`.
Every start verifies the canonical model SHA256, header, K50 tables and all cache
files, then maps the cache read-only. Later starts skip JSON row parsing.
On Linux with one CPU core and local SSD storage, two fresh CLI starts measured:

| Start | Time until ready | Process peak RSS |
|---|---:|---:|
| First: verify, reassemble and build cache | 20 min 15 s | 0.52 GB (0.48 GiB) |
| Later: verify and reuse cache | 81 s | 0.59 GB (0.55 GiB) |

These are process peaks over each CLI's lifetime, including play; operating-system
file caching is additional. Timing varies by CPU and storage. Use a **16 GB machine**.

Allow **25 GB of free disk**, plus the Python environment and session journals:
about **7.67 GB of downloads** (6.28 GB model parts and 1.38 GB card tables),
plus the **6.28 GB reassembled model** and **9.33 GB binary cache** (about 8.69 GiB;
9,331,096,671 bytes including its manifest in the measured build).
Keep the immutable assets, reassembled file and cache. Reassembly uses a
temporary file with an exclusive final link; it does not require a second whole
model copy. The cache is derived locally; it is not a release download.

```sh
git clone https://github.com/dberweger2017/CFR-NLHE-6max.git
cd CFR-NLHE-6max
git checkout v0.5.5
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-play.txt
mkdir -p models/v055
gh release download v0.5.5 --repo dberweger2017/CFR-NLHE-6max --dir models/v055
python3.11 -I -S models/v055/verify_hu200_bundle.py models/v055 \
  --expect-source "$(git rev-parse HEAD)" --require-publication
```

Without `gh`, download every asset from the v0.5.5 release page into `models/v055/`.
The verifier uses only the standard library. It checks every ordered part,
the reassembled whole hash, pinned K50 tables, translation and publication binding.
`SHA256SUMS` alone detects corruption; the pinned whole hash is the model identity.

```sh
.venv/bin/python -m src.play_api.server --hu200-release models/v055 --stack-bb 200 \
  --data-dir results/hu200-play --source-version v0.5.5
```

The first start reassembles `models/v055-model/HU200-v055-three-seed-blend.average.gz`.
It builds the cache next to the bundle in `models/v055-cache/`, under a directory
bound to the model SHA256 and schema. Later starts verify and reuse both.
To assemble ahead of time, add
`--assemble models/v055-model` to the verifier command, using a fresh output directory.
Never put the assembled file inside the asset directory.

A corrupt, incomplete or mismatched cache stops startup with an explicit error;
it is never silently used or replaced. Preserve and move `models/v055-cache/`
to a different local path, then restart to rebuild from the verified model.
An interrupted streaming build retains its `.streaming-partial` directory for
diagnosis. Keep the bundle, reassembled file and cache outside synced folders.

Open the printed local URL and enter the token stored in
`results/hu200-play/access.token`. Choose human play with restricted or free
sizing, or watch the self-play spectator. Restricted sizing uses the trained
minimum/pot/conditional-all-in menu; free sizing accepts every legal whole-chip
wager. Stacks reset to 200 BB each hand. Translation is fixed at 512/128.

Restart with the same bundle and data directory to resume a session. A pending
bot decision preserves its sampled reply and idempotent acknowledgement. To
replay a session's completed hands:

```sh
.venv/bin/python -m src.play_api.server --hu200-release models/v055 \
  --data-dir results/hu200-play --verify-session SESSION_ID
```

This release is **HU200 only**. `--hu100-release` still runs v0.5.0–v0.5.3;
`--models-dir` still runs HU20 with v0.4.2 as its default. The research seeds
in the separate pre-release are verification-only assets for the table app.
