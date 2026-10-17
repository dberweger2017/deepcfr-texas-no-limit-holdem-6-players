# Verify the frozen seeds and rebuild v0.5.5

This research pre-release is verification-only for the local table app. Use the
separate v0.5.5 bundle to play the qualified HU200 blend.

The collection downloads about **25.75 GB**: three average exports, three raw
checkpoints and the K50 tables. Each model/checkpoint part is at most 1 GiB.
The standalone verifier needs only Python 3.11. Assembling all six payloads adds
**24.37 GB**. The default workflow creates no play cache.

For the complete rebuild, start with **100 GB free disk**, a 16 GB machine,
Python 3.11 and a Rust toolchain. Keep at least **70 GB free after the downloads**.
The rebuild retains about 47 GB of assembled inputs and outputs, plus native
build files and logs. It runs sequentially with one Cargo/Rayon worker. The blend
was built and tested on a base 16 GB MacBook Pro M4; #306's full research pipeline
peaked at 12.75 GB family RSS. Reconstruction time depends on CPU and storage;
allow several hours. No training, games or model selection are performed.

```sh
git clone https://github.com/dberweger2017/CFR-NLHE-6max.git
cd CFR-NLHE-6max
git checkout hu200-frozen-20b-seeds
python3.11 -m venv .venv
mkdir -p models/hu200-frozen-20b-seeds
gh release download hu200-frozen-20b-seeds --repo dberweger2017/CFR-NLHE-6max \
  --dir models/hu200-frozen-20b-seeds
python3.11 -I -S models/hu200-frozen-20b-seeds/verify_hu200_bundle.py \
  models/hu200-frozen-20b-seeds --expect-source "$(git rev-parse HEAD)" --require-publication
.venv/bin/python -m scripts.rebuild_hu200_release \
  --bundle models/hu200-frozen-20b-seeds --out results/hu200-blend-rebuild \
  --expect-source "$(git rev-parse HEAD)" --require-publication
```

Download every asset into the same directory; do not mix the v0.5.5 and research
assets. Without `gh`, use the pre-release page's public download links.
The verifier checks each ordered part and each complete reassembled hash, the
K50 tables, original recipe and publication/source binding. No private Drive
archive, access token, model cache or third-party account is needed.

The rebuild command:

1. Verifies and assembles all three averages and all three raw checkpoints.
2. Builds the repository's native `hu200_blend` tool and exporter from the clean
   source commit bound to the release, using its locked dependencies.
3. Freshly exports each seed's average from its raw checkpoint and checks its
   original SHA256. These copies are reconstruction checks, not replacements.
4. Scans the union and checks 156,208,331 keys /482,975,338 action slots, then
   blends raw accumulators with the unchanged scheduled weights.
5. Checks the resulting checkpoint hash, exports its uniform-zero-mass average,
   and requires **`ce3ecc2cb8ebffa1fb3e0a7f11c79d0900b74008b55d85093e1915ff4dde6485`**.

The underlying commands are recorded in `results/hu200-blend-rebuild/commands.json`;
`rebuild-verification.json` records the input/source pins and output hashes.
Every log and partial output remains on failure. Choose a fresh output directory
for another attempt; the script refuses to overwrite one.

`blend-config.json` preserves #306's original provenance strings. The native
`--input-dir` option resolves the actual checkpoint files by seed in the assembled
input directory. You do not need to create the historical absolute paths, edit
that configuration or change the original header. Averaging the normalized
exports instead would produce a different model.
