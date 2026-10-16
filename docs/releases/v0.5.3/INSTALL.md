# Install and run v0.5.3

v0.5.3 needs Python 3.11, about 2 GB of free memory and 8 GB of disk: 4.6 GB of downloads, plus the 3.2 GB whole model, which the first start reassembles from its parts. Every start checks the whole bundle and all 97.3 million stored entries, which took about ten minutes on an M1 MacBook Pro.

```sh
git clone https://github.com/dberweger2017/CFR-NLHE-6max.git
cd CFR-NLHE-6max
git checkout PACKAGE_SOURCE_COMMIT    # the v0.5.3 tag
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-play.txt
```

Download all twelve release assets into one directory, then verify them against this release's source commit. The verifier needs only the Python 3.11 standard library; it checks each model part, the whole model's fixed hash and the three card tables.

```sh
mkdir -p models/v053
gh release download v0.5.3 --repo dberweger2017/CFR-NLHE-6max --dir models/v053
python3.11 models/v053/verify_hu100_bundle.py models/v053 \
  --expect-source PACKAGE_SOURCE_COMMIT --require-publication
```

Without `gh`, download each asset from the release page into `models/v053/`.

Start the local table:

```sh
.venv/bin/python -m src.play_api.server --hu100-release models/v053 \
  --data-dir results/hu100-play --source-version v0.5.3
```

The first start writes the reassembled model to `models/v053-model/`, next to the asset directory, and verifies it against the fixed hash; later starts reuse it after checking it again. Keep the asset directory itself unchanged.

Open the printed local URL and enter the token stored in `results/hu100-play/access.token`. Choose human play (restricted or free sizing) or the self-play spectator. Restricted sizing offers this release's menu; free sizing accepts any legal whole-chip wager. Translation is part of this release and can't be turned off. Other stack depths are rejected.

Sessions are stored per release, so one data directory can hold v0.5.0–v0.5.3 sessions, and a session never changes model. Restarting with the same bundle and data directory resumes them. To replay a finished session independently:

```sh
.venv/bin/python -m src.play_api.server --hu100-release models/v053 \
  --data-dir results/hu100-play --verify-session SESSION_ID
```

`--hu100-release` also loads v0.5.0–v0.5.2 bundles. The HU20 releases run as before, with `--models-dir`; v0.4.2 is their default.

`SHA256SUMS` only detects corruption. Trust comes from the verifier's fixed model and provenance pins, and from the publication manifest bound to the tagged source.
