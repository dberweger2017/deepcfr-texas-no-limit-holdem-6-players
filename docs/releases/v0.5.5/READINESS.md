# v0.5.5 packaging checks

The owner approved publishing #306's exact HU200 three-seed blend after review
and merge of the packaging change. Publication follows the verified draft and
public-download procedure in [v0.5.3's publication record](../v0.5.3/PUBLICATION.md).
The owner reviews and merges this package; publication then needs no second approval
if every declared check passes.

The fixed model is 6,282,968,150 bytes, SHA256
`ce3ecc2cb8ebffa1fb3e0a7f11c79d0900b74008b55d85093e1915ff4dde6485`,
with 156,208,331 entries, PR163 K50 tables and HU200 translation at 512 states /
128 events. No retraining, re-export or selection is part of packaging.

Completed checks:

- The verified-cache runtime independently replays **30 retained hands**:
  16 human and 14 spectator, with no partial hands left. Actual browser controls
  exercised restricted sizing, an off-menu 800-chip raise, a 20,000-chip all-in,
  spectator Play/Pause/Step and a stable journal revision after Pause.
  Pending-bot-decision restart and idempotent acknowledgements pass.
  [Runtime receipt](verification/runtime.json).
- First start verifies, reassembles and builds the streaming cache in
  **1,214.93 seconds / 516,747,264 bytes peak process RSS**. A fresh later start
  verifies and maps it read-only in **80.96 seconds / 586,121,216 bytes**.
  All six cache arrays are verified against the pinned model and mapped read-only.
  The 9,331,096,671-byte cache and both timings are documented in the public
  install guide, model card and release notes. [Measurements](verification/cache-starts.json).
- Wrong valid model, rehashed K50 table and translation identity, wrong stack
  depth and inference adapter, mismatched cache model binding and a corrupted
  probability array are refused. Bad caches are reported and preserved, with
  instructions to move them aside and restart. Every injected fault is restored;
  the complete bundle, model and cache verify afterward.
- Runtime source coverage and the unchanged HU100 policy/translation modules are
  checked separately from native reconstruction. [Source coverage](verification/runtime-source-coverage.json).
- All four old HU100 manifests rebuild byte for byte; the old trace hashes stay
  equal, including a fresh v0.5.3 baseline comparison. All 1,280 trace hand records
  independently replay. The HU20 default remains v0.4.2.
  [Compatibility](verification/compatibility.json), [manifests](verification/old-manifests.json).
- Standalone stdlib verification passes for the blend and the separate collection
  of three averages **and three raw checkpoints**, with ordered parts at most
  1 GiB, pinned tables, recipe, identities and source/publication binding.
  Fresh bundles from clean source `ea5301bb` verify all **49 assets** and all six
  shipped documents byte for byte; both reject a wrong source and a request for
  publication approval. [Review-bundle receipt](verification/review-bundles.json).
  The first complete reconstruction from prepared parts reproduces all three
  averages and both exact audited blend hashes. The original release model is unchanged.
- Public documentation includes protocol-v2 qualification, the failed first
  protocol, AIVAT context, internal results, weaknesses and measured requirements.

Still required before the packaging PR is ready:

- Complete and preserve the second exact reconstruction with the integrated
  native readers; the first reconstruction does not substitute for this check.
- Finish evidence archival and record its accepted restoration receipt.
- Confirm final code and shipped-document source equivalence to the tested
  sources and verified review bundles, run the staged repository artifact guard
  and confirm green CI on the final PR commit. Publication always builds fresh
  approved bundles from the owner's merge commit.

The separate research pre-release is `hu200-frozen-20b-seeds`, never Latest.
Publication requires the owner's packaging merge, publication-approved bundles,
green exact-merge CI, authenticated draft readbacks, unauthenticated public
readbacks of every asset and another full reconstruction from the downloaded
public seed parts. Release assets and raw runtime journals stay outside Git.
The earlier uncached runtime and its repaired spectator timer failure remain
recorded separately in [the historical receipt](verification/runtime-uncached.json).
