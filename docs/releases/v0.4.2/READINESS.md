# v0.4.2 preparation readiness

**Research checks pass; package verified; publication held.** The owner approved only the fixed 12-hour evaluation and merging the completed report PR. A separate explicit chat go is required before any tag or release. v0.4.1 remains Latest and the runtime default.

The [updated decision table](../../reports/hu20-v042-lbr-confirmation.md) combines #185's direct/pressure/panel passes with #188's fresh LBR **−1.091512 [−4.965223, +2.782198] BB/100**, lower >−5 and half-width 3.873710. The safeguard clears by only 0.034777. It does not establish LBR improvement or each lineage's non-regression. All pilot/final hands and actions independently audit; no historical or pilot hands are pooled.

Canonical reviewed package: archive member directory `research/package/` in the [PR188 archive entry](../../../RESULTS_INDEX.md#pr188--fresh-v042-lbr-confirmation-passes-narrowly). Six assets: exact fixed first-seed 2026100601 10B average, model card, release notes, preparation manifest, standalone verifier and SHA256SUMS. [Preparation manifest](PREPARATION_MANIFEST.json), [checksums](SHA256SUMS), [M4 verification receipt](../../reports/hu20-v042-lbr-artifacts/package-verification.json). Preparation source **`1d80bd02c2e8acf2dc99b75f6e0349bb55a941d8`**; all package source/doc bytes come from that commit. The model is byte-identical to #185 and has not been re-extracted. No binary is tracked in Git.

After retrieving and verifying the indexed archive/member hashes, extract only `research/package/` into a fresh ignored directory and run:

```sh
python3.11 verify_v042_bundle.py . --expect-source 1d80bd02c2e8acf2dc99b75f6e0349bb55a941d8
```

Eight focused integrity tests pass, covering exact copy/source binding, overwrite prevention, rehashed manifest/model tampering, duplicate checksums, missing coverage, changed notes and symlink rejection. The actual six-file M4 package independently passes exact model/header/manifest/checksum verification. The first attempt's manually transcribed incorrect source binding was rejected against the actual Git revision, retained as `research/package-attempt-01/`, then corrected in the canonical fresh package. Use only `research/package/`; the incorrect attempt is failure evidence.

Publication approval remains false and approved release source null. On explicit owner go, bind preparation to the owner-approved merged release source and review any desired runtime integration before publishing. This verifier deliberately accepts only an unpublished preparation; a later publication manifest requires a separately reviewed update. No public strength commitment or default/spectator change is implied by preparation.
