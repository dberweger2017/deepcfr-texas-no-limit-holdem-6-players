# v0.4.2 release readiness

**Published and verified as stable Latest on October 8, 2026.** [Release](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/tag/v0.4.2), [final publication/download/runtime record](PUBLICATION.md). Exact tagged source `a53f167fa736a481339034d1a7235202d599418c`; v0.4.0/v0.4.1 remain available.

The owner explicitly authorized the release PR, merge after independent review and green checks, tag and stable Latest publication in chat. This release uses only #188's canonical `research/package/` fixed-first-seed 2026100601 model. The original unpublished preparation and its publication hold remain preserved as provenance; that dated hold has been superseded by the owner's authorization.

The [updated decision table](../../reports/hu20-v042-lbr-confirmation.md) combines #185's direct **+3.50 [+1.63, +5.37] BB/100**, pressure and panel passes with #188's fresh LBR **−1.092 [−4.965, +2.782] BB/100**. All four declared gates pass. The LBR lower bound clears −5 by only 0.034777, with half-width 3.873710. This is not evidence of LBR improvement or each seed's non-regression. The released model supports heads-up 20 BB only, with no professional-strength claim.

## Input and preparation provenance

Canonical archive: [PR188 archive entry](../../../RESULTS_INDEX.md#pr188--fresh-v042-lbr-confirmation-passes-narrowly). Only its six `research/package/` members were retrieved into a fresh ignored nonsynced M4 directory. Whole archive SHA256, embedded manifest SHA256, every selected member's size/SHA256 and the original standalone verifier pass. [Retrieval receipt](verification/retrieval-receipt.json). No bytes from retained `research/package-attempt-01/` are used; no retraining, re-extraction, new arena or model selection occurs.

Original [preparation manifest](PREPARATION_MANIFEST.json) and [preparation checksums](PREPARATION_SHA256SUMS) retain the exact archived provenance, including preparation source **`1d80bd02c2e8acf2dc99b75f6e0349bb55a941d8`** and publication approval false. Those historical checksums describe the archived preparation documents, not the revised publication documents. The exact model remains **249,237,403 bytes**, SHA256 **`15736cc61a72baa1e6722b1566897917ffb6fdf8bea8874a82b5485fe95d4bae`**. No model binary is in Git.

## Runtime and publication binding

v0.4.2 is the release checkout's default. v0.4.0 and v0.4.1 remain selectable and downloadable; existing persisted sessions retain their chosen model. Human restricted/free sizing and spectator playback use the existing observation boundary, legal menus, probabilities and rules. No game or statistical code changes.

The publication bundle contains seven files: model, card, notes, publication manifest, fixed catalog identity manifest, SHA256SUMS and standalone Python 3.11 verifier. The runtime pins `catalog-manifest.json` by SHA256; its bytes identify the exact approved model and preparation. The separately generated `release-manifest.json` binds publication approval to the actual merged/tagged source. This avoids a containing Git commit referring to its own hash. Both manifests are checksummed and the verifier rejects altered catalog identity, lost preparation provenance, invalid approval/status/tag and mismatched source.

The verifier still accepts the exact archived unpublished preparation, with its original manifest pinned by hash. Publication verification additionally requires explicit approval and an independently supplied actual tagged source:

```sh
python3.11 verify_v042_bundle.py . --expect-source TAGGED_COMMIT --require-publication
```

## Verified integration

**89 focused tests pass** (`tests/play_ui` and `tests/test_v042_bundle.py`), including default selection, older sessions/restarts, spectator audit, exact policy distributions and rejection of inconsistent publication metadata. JS syntax and the staged repository artifact guard pass. Independent review identified an old two-model spectator fixture inheriting the new default; it was corrected and the independent rerun passes all 89 tests.

The deterministic M4 smoke completed in **80.45 seconds**, with **four human hands /37 actions /16 bot positions** and **four spectator hands /15 decisions**, all independently replayed and settled. Human play covers both seats, restricted/free sizing and one exact 201-chip raise; spectator covers v0.4.2 against each older release. All recorded bot positions were trained entries in this small sample; this does not establish coverage over all play. Catalog/default/model hashes and all three available versions match. [Summary](verification/smoke-summary.json), [source binding](verification/smoke-source-binding.json), [resource receipt](verification/smoke-resource-receipt.json). Peak RSS **2,959,736,832 bytes /2.76 GiB**, below the 3-GiB guard; minimum free disk **100,125,777,920 bytes**, above 15 GiB. No arena or strength estimate was run. Complete private journals/source inventories remain in the own ignored M4 release root.

[Release PR #198](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/198) records final-head review, CI, merge and subsequent publication/download receipts. The sequence below completed; final download/source/runtime and older-release checks pass as recorded above.

## Review and publication sequence

1. Run focused runtime/bundle tests and a small deterministic load/play/replay smoke on M4. Independently audit every smoke action, policy distribution and settlement. Record independent source review and resolve every finding.
2. Merge the reviewed release PR only after all required final-head checks are green.
3. From that exact merged source, build a fresh explicitly approved publication bundle using the unchanged verified model. Preserve the original preparation; bind `package_source_commit` and `approved_release_source_commit` to the actual merged source.
4. Tag `v0.4.2` at that commit, create a draft, attach all seven assets, download them into a fresh ignored directory and verify the whole publication bundle against the tag's actual commit.
5. Publish stable Latest only after draft-download verification. Repeat public download verification, confirm the supported runtime loads the intended default and older releases remain accessible with unchanged assets. Post final source/hash/download receipts on the release PR.

No publication is inferred from the research pass alone. This sequence executes the owner's explicit release authorization; it does not schedule further research or cleanup.
