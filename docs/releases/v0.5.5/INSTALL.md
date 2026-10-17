# Install and run v0.5.5

To play, use Python 3.11, Git and a Rust toolchain for the pinned engine's source
build, on a 16 GB machine. The blend was built and tested on a base
16 GB MacBook Pro M4; #306 measured 12.75 GB peak family RSS for its research
pipeline. Two fresh CLI starts took **22m41s and 22m16s**, with peak process RSS
of **9,412,120,576 bytes** (9.41 GB /8.77 GiB) on the packaging test machine.
Startup validates 156,208,331 rows; speed varies by CPU and storage. Use a
**16 GB machine with about 12 GB RAM available** before starting. Memory used
by validation is released when the process exits.

Allow **15 GB of free disk**, plus the Python environment and session journals:
about **7.67 GB of downloads** (6.28 GB model parts and 1.38 GB card tables),
plus the **6.28 GB reassembled model**. The default runtime makes no play cache.
Keep both the immutable assets and the reassembled file. Reassembly uses a
temporary file with an exclusive final link; it does not require a second whole
model copy. Independent cache workflows are optional and require about 9.4 GB
additional disk; they are not needed by the table app.

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
Later starts verify and reuse it. To assemble ahead of time, add
`--assemble models/v055-model` to the verifier command, using a fresh output directory.
Never put the assembled file inside the asset directory.

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
