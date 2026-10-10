# Install and run v0.5.0

v0.5.0 needs Python 3.11, about 4 GB of free memory and 2.5 GB of disk. Every start checks all 41 million stored entries, which took about four minutes on an M1 MacBook Pro.

```sh
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git
cd deepcfr-texas-no-limit-holdem-6-players
git checkout PACKAGE_SOURCE_COMMIT    # the v0.5.0 tag
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-play.txt
```

Download all seven release assets into one directory, then verify them against this release's source commit. The verifier needs only the Python 3.11 standard library.

```sh
mkdir -p models/v050
gh release download v0.5.0 --repo dberweger2017/deepcfr-texas-no-limit-holdem-6-players --dir models/v050
python3.11 models/v050/verify_v050_bundle.py models/v050 \
  --expect-source PACKAGE_SOURCE_COMMIT --require-publication
```

Without `gh`, download each asset from the release page into `models/v050/`.

Start the local table:

```sh
.venv/bin/python -m src.play_api.server --v050 models/v050 \
  --data-dir results/v050-play --source-version v0.5.0
```

Open the printed local URL and enter the token stored in `results/v050-play/access.token`. Choose human play (restricted or free sizing) or the self-play spectator. Translation is part of this release and can't be turned off. Other stack depths are rejected.

Use a fresh data directory. Restarting with the same bundle and data directory resumes your sessions. To replay a finished session independently:

```sh
.venv/bin/python -m src.play_api.server --v050 models/v050 \
  --data-dir results/v050-play --verify-session SESSION_ID
```

The HU20 releases run as before, with `--models-dir`; v0.4.2 is their default.

`SHA256SUMS` only detects corruption. Trust comes from the verifier's fixed model and provenance pins, and from the publication manifest bound to the tagged source.
