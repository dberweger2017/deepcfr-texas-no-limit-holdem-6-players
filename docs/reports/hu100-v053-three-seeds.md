# HU100 v0.5.3 candidate: three-seed qualification

**In progress; no qualification result yet.** The owner decides release and merge.

[PR286](https://github.com/dberweger2017/CFR-NLHE-6max/pull/286) repeats
#260 → #272 on fresh seeds 2026101501 then 2026101502, sequentially on the M4.
The [protocol](../hu100-v053-three-seeds-protocol.md) pins the unchanged cap,
recipe, one 14-hour clock and full resource guard. The current-main build is
byte-identical to #272; [receipt](hu100-v053-three-seeds-artifacts/build-receipt.json).

Seed 2026101501 reached **330,748,996 nodes /100,827 iterations /97,291,138 entries**
at its first complete-iteration capacity stop. Its stage-2 continuation reached **5,000,004,085 nodes /1,794,239 iterations**
with identical sorted keys. All **97,291,138** final current/average rows pass
[the full audit](hu100-v053-three-seeds-artifacts/2026101501-5b.audit.json).
Average SHA256 `83771d644b423dc3e32fd313b0bb127b18920b6eb58d7ec6b6f90a7036087000`.
[Final lineage and membership](hu100-v053-three-seeds-artifacts/2026101501-training-verified.json). [Stage-1 receipt](hu100-v053-three-seeds-artifacts/2026101501-stage1-verified.json).

Seed **2026101502** reached its stage-1 capacity terminal at **330,510,042 nodes
/101,154 iterations /97,291,445 entries** and is continuing frozen to 5B.
[Stage-1 receipt](hu100-v053-three-seeds-artifacts/2026101502-stage1-verified.json).

No final hands run before the canonical roots/counts/measured cost plan is
posted and read back in the PR description. Outcomes remain sealed through
independent replay and exact reproduction of every final hand. The final report
will place both new panels beside #272 and report three fresh direct matches
against v0.5.2 with translation on (512/128), v2 off. There is no improvement-
release pass/fail gate; #272’s historical rule is reported for comparison.

Preparation retains worktree/refspec and PATH failures before the first build,
a metadata helper’s system-Python mismatch, and one corrected unit-test coverage
assertion. No scientific retry or failed final-hand rerun has occurred.
Archive and restoration status is in [RESULTS_INDEX](../../RESULTS_INDEX.md).
No release, tag, default change, Slumbot, credential work or merge.
