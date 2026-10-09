# HU100 independent seeds: campaign in progress

[PR215](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/215)
is open and unmerged. [Protocol](../hu100-independent-stages.md).
Merged #211 retained a 1,000,373-node pilot, not new 1B qualification results.
This new campaign admits useful training independently of final evaluation.
No qualification or playing-strength claim exists yet.

## Calibration completed without inspecting winnings

Existing #207 early/terminal policies were evaluated at32 and512 independent
swapped-seat blocks/opponent, a16x sample ratio, fully replayed/reproduced.
One load per checkpoint: early **30.851s**, terminal **171.769s**. The same
validated model serves both sizes and primary/reproduction; terminal also
serves on/off. No model-copy/reload optimization changes any probability,
private draw, observation or action. Fresh tiny-model regression independently
checks reused versus separately loaded policies, translation reset, full replay,
reproduction and corruption rejection.

| Arm | Fixed validation/snapshot/model hashes,32 /512 s | Play/replay/reproduction/raw hash/setup,32 /512 s |
|---|---:|---:|
| Early/off |0.604 /0.601|1.896 /20.383|
| Terminal/off |3.622 /3.623|1.738 /15.878|
| Terminal/on |3.625 /3.625|1.636 /16.035|

Initial loads are separate from this table. Early plays the uniform reference;
other arms reuse its exact retained rows. Replay includes repeated reference
rows, so these aren't like-for-like candidate-only hand rates. The early
primary play/report phase alone is0.445s /7.495s. Snapshot/model hashes are
nearly constant across counts. Block-scaled schedule/stream setup joins the
variable upper bound; its constant manifest overhead is conservatively included.
The final quote uses3x maximum measured variable seconds/block across counts
and arms, plus3x entries-scaled six checkpoint loads and fixed model-bound
validation/snapshot/hashes perarm; strict reporting is separately budgeted.
The quote is an upper planning estimate, not observed final runtime. Full output
inventory hashing, previously outside wall_seconds, is now measured explicitly.
[Cost receipts](hu100-independent-stages-artifacts/calibration-costs.json).

## Preparation failure retained; original budget preserved

Current-main base6e18043 contains merged #210 optional equity-bucket support.
Initial source dd19995 incorrectly demanded that this inactive native source
match #207's binary source. Preparation stopped before any training fixture,
calibration or final hand. The unchanged binary's SHA256 passed. The narrow
independently reviewed repair archives/binds its actual frozen bd0e7a native
source rather than changing or rebuilding the executed trainer. Python's
legacy HU100 v1 branch remains unchanged; optional bucket behavior is unused.

Original deadline **October9 10:34:33–16:34:33 UTC**, original swap baseline
**1,429,408,317.44 bytes**, and all guard limits remain. Original failure/receipt
remain immutable, with a separate future-failure latch. Exclusive ownership,
zero prior science, old supervisor exit, all old guard observations and301
fresh samples against the original baseline gate readmission. **370.914s idle
monitoring gap** is disclosed and charged to the same clock; no continuous
sampling claim covers that gap. New monitored source is
**1d862d6f9ea2e5e56b23c84cac94561103b5da11**. No model/training/hand retry,
clock reset, baseline rebase or guard waiver occurred.
[Initial input PR states](hu100-independent-stages-artifacts/input-pr-status.json),
[source review](hu100-independent-stages-artifacts/source-review.json),
[repair review](hu100-independent-stages-artifacts/preparation-repair-review.json),
[readmission](hu100-independent-stages-artifacts/preparation-readmission.json).

The actual indexed #211 partial passes size/SHA256/audit identity and exact
resume compatibility: resumed and fresh direct seed2026100901 training to2M
both produce checkpoint SHA256
`f3d7a74076e41885477e4413be3606753eddc25bcc683f4042cfc56fe3531b41`.
Main training resumes the original1Mpartial, not this fixture. Existing old
averages are size/SHA256-audit pinned and physically linked into the nonsynced
root; no old cloud archive hydration or original mutation. Native binary remains
`7650ad60bbf2437622ea3c39d37c7d56686d00bac11680e44a6e47dab509a262`.

Seed2026100901's separate admission is **99.11min upper training/save/tools**
plus the shared **30min closeout reserve**. Free disk97.17GiB exceeds required
33.09GiB including prior-stage archive bytes, new models/originals/archive,
transient space and the fixed15.5GiB floor. Evaluation is not included in this
training gate; its later quote cannot retrospectively refuse useful training.
[Admission](hu100-independent-stages-artifacts/training-admission-2026100901.json).
The protocol requires retaining early39,438,279 and terminal1B total-node
checkpoints for both fresh seeds, allowing complete-iteration overshoots, with
full exports/audits. Seed2026100901 has reached1,000,000,506 nodes; its terminal
audit is running. Its early endpoint is fully audited. Seed2026100902 has not
started. These are progress observations, not completed qualification results.

## Remaining work

Training and final evaluation admission/results are pending. Recipe qualification
remains incomplete. No final sample/final hands/effect estimates exist yet.
Research evidence, failures and originals stay in the M4 ignored working root;
accepted Research-Cloud archive/alias restoration receipts will be indexed at
closeout. No remote-byte/archive-upload claim, cleanup, release, paid compute,
changed recipe, automatic larger run or merge. Source qualification:38 focused
tests pass; archive restoration regression passes. Final review/CI remain pending.
