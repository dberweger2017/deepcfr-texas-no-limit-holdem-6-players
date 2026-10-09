# Retrieve, verify and run the unpublished candidate

There is no v0.5.0 release download yet. Use the package-source commit named
in `candidate-manifest.json` (the repository preparation receipt also records it);
this document in a package is bound by its
`candidate-manifest.json` and `SHA256SUMS`. Python 3.11 is supported.

The builder substitutes `PACKAGE_SOURCE_COMMIT` below with the exact frozen
commit. In the repository template, use `package_source_commit` from the
preparation receipt instead. The builder requires that commit checked out with
clean tracked bytes.

```sh
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git deepcfr-v050
cd deepcfr-v050
git fetch origin feature/v050-release-readiness
git checkout PACKAGE_SOURCE_COMMIT
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-play.txt
```

Download #207's existing [accepted archive](https://drive.google.com/file/d/1iowJoQBQqB3tLRnU6GD0JcniF0qDIvgj/view)
from [its canonical folder](https://drive.google.com/drive/folders/1qhlOHmphBGSFfiM82S7T4B_KhdabyRUS)
to a fresh ignored nonsynced location if it is not already allocated locally.
Do not duplicate it in Research-Cloud. The whole ZIP is 20,517,119,304 bytes,
SHA256 `ba3e82d8fa79be32d445c54eb240069c4a717cb75af9713f3eaf7ac86364fddf`.
The retrieval command checks the whole ZIP, embedded `ARCHIVE-MANIFEST.json`
SHA256 `609d4899a363b384aa58b0daaef2ff85f2bcae7b135c2a4e3604813fc4182d58`,
and member `research/training/1000000000/average.gz`. It copies unchanged bytes
into a fresh destination and verifies the model/header, never re-extracting a
policy from a checkpoint. The failed original partial ZIP is not usable.

```sh
.venv/bin/python -m scripts.retrieve_v050_candidate \
  --archive "$HOME/Local/Research-Cloud/PR-207-hu100-1b/hu100-1b-campaign-M4-retry-20261009.zip" \
  --out models/v050-retrieved
.venv/bin/python -m scripts.build_v050_bundle \
  --source models/v050-retrieved/O1B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz \
  --out models/v050-candidate --source-sha "$(git rev-parse HEAD)"
python3.11 models/v050-candidate/verify_v050_bundle.py models/v050-candidate \
  --expect-source "$(git rev-parse HEAD)"
.venv/bin/python -m src.play_api.server --v050-candidate models/v050-candidate \
  --stack-bb 100 --data-dir results/v050-play --source-version "$(git rev-parse HEAD)"
```

Open the printed loopback URL and enter the token from the printed local file.
Choose human restricted/free or self-play spectator. Translation is fixed enabled
in this candidate; `--translate-off-menu` is only the older explicit research
option and is rejected with this candidate. Other stack depths are rejected.
Use a fresh data directory; never repurpose an older release or research journal.
Restart with the same package/data directory to resume. For independent replay:

```sh
.venv/bin/python -m src.play_api.server --v050-candidate models/v050-candidate \
  --data-dir results/v050-play --verify-session SESSION_ID
```

For a package delivered separately, run its standalone verifier with the
reviewed expected source first. SHA256SUMS detects corruption; trust comes from
the reviewed verifier and fixed model/provenance pins, not a checksum list alone.
A clean runtime load validates all 41,010,014 stored entries and can take several
minutes and a few GiB of RAM. Packaging plus one restored copy needs about
2.4 GB in addition to the archive; retain 15.5 GiB disk headroom. M4 smoke resource
measurements are in READINESS; no universal startup-time guarantee is implied.
