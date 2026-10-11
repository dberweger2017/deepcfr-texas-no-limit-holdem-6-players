# Install and run v0.5.1

v0.5.1 needs Python 3.11, about 4 GB of free memory and 3.5 GB of disk. Every start checks all 54.6 million stored entries, which took about five minutes on an M1 MacBook Pro.

```sh
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git
cd deepcfr-texas-no-limit-holdem-6-players
git checkout PACKAGE_SOURCE_COMMIT    # the v0.5.1 tag
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-play.txt
```

Download all seven release assets into one directory, then verify them against this release's source commit. The verifier needs only the Python 3.11 standard library.

```sh
mkdir -p models/v051
gh release download v0.5.1 --repo dberweger2017/deepcfr-texas-no-limit-holdem-6-players --dir models/v051
python3.11 models/v051/verify_hu100_bundle.py models/v051 \
  --expect-source PACKAGE_SOURCE_COMMIT --require-publication
```

Without `gh`, download each asset from the release page into `models/v051/`.

Start the local table:

```sh
.venv/bin/python -m src.play_api.server --hu100-release models/v051 \
  --data-dir results/hu100-play --source-version v0.5.1
```

Open the printed local URL and enter the token stored in `results/hu100-play/access.token`. Choose human play (restricted or free sizing) or the self-play spectator. Translation is part of this release and can't be turned off. Other stack depths are rejected.

Sessions are stored per release, so one data directory can hold both v0.5.0 and v0.5.1 sessions, and a session never changes model. Restarting with the same bundle and data directory resumes them. To replay a finished session independently:

```sh
.venv/bin/python -m src.play_api.server --hu100-release models/v051 \
  --data-dir results/hu100-play --verify-session SESSION_ID
```

`--hu100-release` also loads a v0.5.0 bundle. The HU20 releases run as before, with `--models-dir`; v0.4.2 is their default.

`SHA256SUMS` only detects corruption. Trust comes from the verifier's fixed model and provenance pins, and from the publication manifest bound to the tagged source.
