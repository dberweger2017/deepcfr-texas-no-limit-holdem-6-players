# v0.4.1 release readiness

The owner selected #176's exact first-seed O export. #176 and #179 are merged; #180's storage policy is preserved. [Release PR #181](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/181) prepares the report, proposed default, assets and verification. **The owner gave the final go in chat: “yes do the tag and release”. [v0.4.1 release](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/tag/v0.4.1) assets are bound to the actual merged source and verified after download before publication. v0.4.0 remains available.**

## Review packet

- [Comparison, all thirteen panels, lineage results, progress chart and limits](../../reports/v0.4.1-release.md).
- [Final release-notes draft](RELEASE_NOTES.md), [model card](MODEL_CARD.md), [cleanup inventory](CLEANUP.md).
- Exact unchanged O bytes: 142,677,367; SHA256 `571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`. No current-policy zero-mass replacement, training checkpoint or engine binary is included.

## Web verification

Real browser play on the free M4 completed **eight O hands and two v0.4.0 hands**, with both human seats represented for each model. O covers restricted and free sizing; free sizing includes an exact **201-chip / 2.01-BB raise-to**. The final web source is `34aa980e31758fc6bb49ed17d7265fe6e0c0df53`; two earlier completed restricted O hands at `fefd612821dc8139ec083c97af7c332e4319ee8e` are retained and also audited. Later changes add audits, verification and documentation without altering the tested web runtime.

| Check | Result |
| --- | --- |
| v0.4.1 default; v0.4.0 selectable | Both pinned hashes displayed; separate sessions remain bound to their chosen model. |
| New session after settlement | Returns to model selection and preserves completed history. |
| Native replay of every action and settlement | All **10 hands / 83 actions** independently pass actor, legal bounds, event hash, terminal chips and session totals. No unfinished hand is omitted. |
| Recorded web distributions equal arena loader | Exact menu/probability/found equality at **29 O and 8 R bot positions**, loaded independently from the pinned exports. Only the bot's legal observation is passed to either reader. |
| Fallback telemetry | **0 missing keys, 0 stored zero-mass positions, 0 unexpected fallbacks** in these hands. This small smoke does not establish zero fallback over all possible play. |
| Hand history | Clean native blinds, actions, board, revealed/mucked cards and alternating seats; the 201-chip wager is retained exactly. |
| Browser/API/server | **92 captured final-run API responses, all HTTP 200; no window errors or server error output** in the checked interval. |
| Resources | Final server peak RSS **4,943,855,616 bytes**, minimum free disk **28,135,575,552 bytes**. Independent O audit peak **5,735,956,480 bytes**; below the six-GiB guard. One worker at a time, no RunPod. |

The read-only [audit script](../../../scripts/audit_v041_web.py) reconstructs native hands directly instead of invoking the service's replay/reporter. Its automated test uses recorded decisions in both seats and proves that changed settlement and changed web probabilities are rejected. **53 focused web checks pass.** Sanitized [O audit](verification/v0.4.1-web-audit.json), [R audit](verification/v0.4.0-web-audit.json), [browser receipt](verification/browser-check.json) and [resource receipt](verification/web-resource-receipt.json) are committed; complete private journals and positions remain in the member-hashed research archive.

Screenshots: [model selection](screenshots/model_choices.png), [O button](screenshots/final_restricted_button.png), [O big blind](screenshots/final_restricted_bb.png), [restricted history](screenshots/final_restricted_history.png), [exact 2.01-BB input](screenshots/final_free_201.png), [four free hands](screenshots/final_free_history.png), [v0.4.0 selection](screenshots/v040_selection.png), [v0.4.0 completed hands](screenshots/v040_complete.png).

Auxiliary attempts are retained: an early navigation before server readiness reset the connection; one exact-button selector and some waits needed correction; a fixture initially patched only one module's pinned constants. Corrected invocations pass. A bot folding before the human acts is a valid completed hand, and the driver now waits for asynchronous processing to settle before accepting completion. No failed game, resource or release check is bypassed.

## Assets and publication sequence

The unpublished package contains exactly the unchanged model, model card, notes, manifest and SHA256SUMS. A clean detached checkout at `654734e6df88dd36397b5f8605cbb12d54944de2` built the package, downloaded **all five staged files over loopback HTTP**, and passed both standalone model verifiers and the whole-bundle verifier; the checkout stayed clean. [Download and member hashes](verification/clean-checkout-download.json). This verifies download bytes locally; it does not claim unpublished GitHub assets are available. Final-head CI and a final-source download receipt are posted on the PR before approval.

Publication procedure, authorized by the final chat go and conditional on green final-head checks with no unresolved findings:

1. Merge the reviewed PR without bypassing checks. Use the actual merged commit, not a stale preparation head.
2. Build a new publication package from the same model bytes with explicit approval, binding `approved_release_source_commit` and preparation source to that merged commit. Preserve the unpublished package and evidence.
3. Create and push the new `v0.4.1` tag at that commit; never move an existing tag.
4. Create a draft GitHub release, attach the five files, download them into a fresh directory from a clean checkout, and run the model and full-bundle verifiers with `--expect-source` set to the merged SHA.
5. Publish and mark Latest only after that download passes. Keep v0.4.0 and its assets available; post the tag and release links.
6. Open the requested spectator-mode follow-up after publication: pick two models, watch hands and inspect each decision's probabilities and legal information. Do not build that feature in this PR.

Full final-head CI is mandatory. Research/verification files are archived under `~/Local/Research-Cloud/PR-181-v0.4.1-release/` with member hashes and retrieval provenance. All originals and synced Drive files remain; #166's open roots are untouched.
