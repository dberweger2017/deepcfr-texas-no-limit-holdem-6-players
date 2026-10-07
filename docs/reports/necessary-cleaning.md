# Necessary cleaning: repository and archive audit

October 7, 2026. Inventory source: `5ba66c159ec8e7624772cedf72012dd80bb310f6`
(main after #189). This audit is an inventory and maintenance decision record;
it is not archive acceptance for all research files or permission to delete them.
[Storage contract](../artifact-storage.md), [development map](../development.md).

## Findings and changes

The source tree had 298 top-level scripts, 141 Python-source files under `src`,
and 1761 files / about 240.5 MB under `docs`. Markdown accounted for about
2.85 MB of the documentation tree. The rest includes useful small receipts but
also substantial raw research. A static import audit found library-to-script
imports and released average inference living in `src/diagnostics`.

This change moves the average reader/validation and compact storage into
`src/blueprint`, and release pins/verifiers into `src/policies`. Diagnostic
extraction/audits remain separate. Existing format strings, model hashes, reader
math, probabilities and fallback semantics are preserved. The release catalog
can add a reviewed model without rewriting session routing. The public verifier
commands and single-model launch flags remain usable.

The play-only requirements install NumPy and the pinned engine. Full research
requirements retain PyTorch, SciPy and the existing developer/monitoring setup.
No neural trainer or reference test is removed merely because tabular play no
longer needs its dependencies installed.

The storage guard retains **165 existing payload paths /190,211,521 bytes** by
exact Git blob and size, while rejecting new or changed payloads in those classes.
This number is a subset of tracked documentation, not reclaimed disk space.
Run `python -m scripts.check_repository_artifacts --revision HEAD --inventory`
to inspect current counts and paths. The retention list's source revision fixes
the pre-cleanup inventory; new code/documents naturally change total file counts.

## Evidence retained pending migration review

Largest tracked research groups at the inventory revision (decimal MB):

- Exact turn-check artifacts: 33.19 MB.
- CFR-average artifacts, including an interrupted trace: 29.90 MB.
- Card-v2 artifacts: 28.72 MB.
- Stackoff artifacts: 27.85 MB.
- Seat-symmetry artifacts: 22.44 MB.
- History-alias artifacts: 13.32 MB.
- Turn-search artifacts: 11.26 MB.

These are migration candidates, not obsolete files. Some exact-turn/board-pooling
inputs support open #190; search evidence can support the next stackoff diagnosis.
No archive ownership/member mapping is inferred solely from a directory name.
This PR does not delete these payloads, research originals or synced files.
Removing them from a future tree would not remove their bytes from Git history.

## Google Drive spot checks

Live GitHub checks found #169, #182 and #185 merged. Drive metadata readbacks on
October 7 matched the index's names, byte counts and parent folders for:

- [#169 scoring](https://drive.google.com/file/d/17X9Fpi1gNL0BVU24iN6xGBuoIMG8_Jpa/view):
  `pr169-scoring-6fd63e0.zip`, 1,869,797,825 bytes;
  parent `1j9i0_HYXfiDjJYmwqvIxdkicPowWcYhT`.
- [#182 learning curve](https://drive.google.com/file/d/1V2bbJ9kf0_MTdqwfcoo__XnCMWjAEkbi/view):
  `hu20-learning-curve-complete-20261006.zip`, 1,899,498,972 bytes;
  parent `1wNthhMPoO0XGPeEZbMZmQVIIxF4tJra9`.
- [#185 confirmation](https://drive.google.com/file/d/1o75pBJpeTsynTzMc0ZDQEDPeVk88Yopt/view):
  `hu20-o-10b-confirmation-retry-02-final-9h-20261007.zip`, 1,764,022,787 bytes;
  parent `1uTSTXQKn19JaGWTTx_lbrANfxO01xPtL`.

These are a sample of indexed archives, not a complete reconciliation of all
tracked payloads or Drive contents. This audit did not download archive bytes,
reverify their members, or obtain fresh native upload-status acceptance. The
connector did not expose a SHA256 in these responses. No cleanup eligibility is
claimed from these metadata reads. Existing SHA256s and restoration instructions
remain in [RESULTS_INDEX](../../RESULTS_INDEX.md).

## Remaining retirement work

Completed-campaign launchers such as `hu20_500m_control.py`, `dr2x2_control.py`
and `run_posterior_audit_v2.py` should be reviewed by owning campaign before
retirement. They are not the supported entry points for a new release campaign,
but source pins and replay dependencies can still require them. The inventory
found no identical tracked blobs of at least 100 KB, so there is no demonstrated
large byte-identical duplication to remove from the current tree.

`src/holdem` and `src/solver` still serve checkpoint adapters and reference tests.
Remaining library-to-script imports in pooling/turn diagnostics and multistreet
fitting deserve narrowly scoped refactors when their active consumers are stable.
Do not sweep these into the ongoing #188/#190 scientific snapshots.

There were 53 registered local worktrees at audit time. No process/ownership,
archive or disk-reclamation audit was performed for them. They remain untouched.
The separate roadmap reconciliation is #191; this PR does not duplicate that edit.
