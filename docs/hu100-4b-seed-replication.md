# HU100 4B seed replication: prospective protocol

## Scope and current admission

The owner authorizes this evidence campaign on the free 10-core, 16 GB M4,
using `feature/hu100-4b-seed-replication` in the fresh isolated checkout
`~/Local/hu100-4b-seed-replication-20261011`. Initial main is
`c3869c9d8884cf7eaab66790b17b3e36398a3ac3`.
There is no release, tag, publication, paid compute, recipe/menu/abstraction
change, other-PR cleanup or unattended schedule.

Initial admission is preventive: disk cannot support the required outputs and
archive while preserving the 16 GiB floor. GitHub CLI authentication also returns
HTTP 401. At the initial checks no training, timing pilot or final hand had started. The owner is
addressing the prerequisites. GitHub authentication subsequently succeeds;
free disk increased to approximately48GiB, still below the provisional77.6GiB
full-campaign requirement. The [recheck](reports/hu100-4b-seed-replication-artifacts/admission-recheck-20261011.json)
preserves this later observation without overwriting initial evidence.
The dated [admission receipt](reports/hu100-4b-seed-replication-artifacts/storage-admission.json)
is a storage forecast, not a timing quote or a guard breach. The owner subsequently authorized starting while making additional disk
available. Admit each training/save/export/audit stage independently above the
16GiB floor; this permits the first seed without admitting unstarted later
stages, final evaluation or archival. Pause before an unadmitted stage; preserve
its admission receipt. Remeasure before each stage. Final sample counts, elapsed-time budgets and physical schedule
disjointness are not yet frozen or verified.

Required reads: AGENTS.md, ROADMAP.md (How we work), artifact-storage,
development, rules, #226 report/protocol/model index, #223 report, #215 report,
compact-table and streamed-export reports, and v0.5.1's model card.
Do not modify `native/hu20-trainer/`. Build with an external Cargo target
directory under this checkout's ignored `build/`; pin source and binary.
A trainer defect stops the campaign and is reported on its PR.

## Sequential training and exactness

Train seeds **2026100901, then 2026100902**, fresh from each seed. One training
process at a time. Unchanged recipe:
`train --stack-bb 100 --seed S --roots-per-seat 1 --average-rule opponent-sampled`.
Save milestones at the first complete iteration at/past 1B, 2B and 4B nodes,
with telemetry. Between milestones continue only this campaign's newly produced
checkpoint, hash verified; never resume a historical checkpoint.

All checkpoint/current/average pins come from
[the #226 model index](reports/hu100-4b-seed-ladder-artifacts/model-index.json).

| Seed | Milestone | Checkpoint SHA256 | Reference max_entries |
| --- | ---: | --- | ---: |
| 2026100901 | 1B | 5f4898d839011d62fda20e3df7e1546150ca1d72dd666654a6d4bfcaaf63a6e7 | 57,658,644 |
| 2026100901 | 2B | a8d97ae543ff44673312916a4150070f7ecb753c315fa6ffa9b35d093395a4c0 | 72,437,552 |
| 2026100902 | 1B | c21e6dd51694b379b28cb5dcc2ab595fa154092f1b69a4cfd1a9149fc85b23e4 | 57,658,644 |
| 2026100902 | 2B | 361032d94f109f5251a1680106a5bd09c24c0f93a2a02ab14c22e4b73145bc0e | 72,437,552 |

The checkpoint header declares `config.max_entries = max(cli_cap, entries)`.
Reuse the reference cap for byte equality at the gates: the measured endpoint
entry counts (approximately 41M and 54.6M) are below the memory ceiling.
If a safe cap must differ, require identical decompressed rows and a header
equal except for this declared field, compared with hash-verified reference
bytes. Stop on any other mismatch. Preserve explicit gate receipts.
Stream-export current and average (uniform zero mass) and fully audit every
evaluated checkpoint; check indexed export identities at the reference gates.

For growth, derive the cap from #226's measured 4B **whole-family training/save**
peak: 6,826,049,536 bytes / 70,466,972 entries. Forecast at 1.10 times those
measured bytes per entry (10% headroom), covering the existing controller,
monitor and save workspace as part of that family measurement. Reserve an
additional 1 MiB for complete-iteration overshoot. The provisional cap is
floor((7 GiB - 1 MiB) / (1.10 * peak / entries)).
Before launch measure the new controller/monitor overhead; reserve any increase
above #226's measured fixed overhead rather than silently using this bound.
Freeze the resulting cap and do not raise it to obtain a target.
A clean entry-capacity stop at a complete iteration is a valid terminal:
fully export/audit and evaluate it, identifying actual nodes. If a required
reference milestone cannot be reproduced safely, stop and report the missing
gate. No scientific retries.

## Candidate restoration

The candidate is **#226's seed 2026100601 4B average**, not a newly trained seed:

- Checkpoint SHA256:
  `65fd61d6999d7a06f36fc1c718aea20e7c23856aaa491df7d517e312c1c33368`.
- Average: **2,121,415,396 bytes**, SHA256
  `0ed7f9512996bb3e9535a7258a5d0e4bc8c00f22734d511c642c0e5b95cbd3c0`.
- Its 2B comparator: **1,603,422,956 bytes**, SHA256
  `e6f79ccac39a651352424e05382681f4d1994bb570b025f692556d170f87e9ae`;
  checkpoint `e84039c7a934a966a2c01f237748a23951124a8b665edc2585ef4809e5681676`.

Check the owning PR's live status before restoration. Restore from retained
M4 originals or accepted archives, verifying sizes/SHA256 before use.
Record exact source path or Drive ID, member, expected/actual hash and retrieval
command in RESULTS_INDEX and a restoration receipt. Reference #207/#215/#223/#226
indexed models in this campaign's archive; do not duplicate them. No hashes
have yet been claimed freshly verified by this campaign.

## Guards and storage

Whole-family **7 GiB soft / 9 GiB hard**, **3,000,000,000 bytes total swap**
(not swap growth), normal memory pressure, at least 15% reported system free,
**16 GiB free disk** on every output/archive volume, and AC power.
Use the existing reviewed creation-verified ancestry monitoring and child
cleanup; missing resource observations fail closed. Include controller and
workers in the monitored family. Verify stop-file handling before relying on it.
Any actual guard breach, numerical failure, information leak, invalid action,
accounting error or exactness mismatch stops the campaign and is reported on
the PR, with one immediate SOMA. Do not retry failed science.

Admission counts all retained milestone files, new outputs/staging, replay and
reproduction traces, telemetry/guards and immutable ZIP copies; count hardlinks
once. Strictly sorted checkpoints use streamed exports without sort workspace.
No deletion or eviction inside synced folders, no force offloading and no
cleanup of other PRs. Remeasure disk before each operation.

## Direct primary and descriptive evaluations

Use unchanged #226 direct duplicate harness `scripts.evaluate_hu100_direct`:
100 BB reset each hand, blinds 50/100, no rake, each deal played with seats
swapped, private action streams, translation off, native-menu actions only.
Policies receive legal observations, without evaluator seeds or hidden cards.

Two primary contrasts: each new seed's fixed 4B (or clean capacity terminal)
versus its own 2B average. Paired Student-t intervals over independent duplicate
blocks; Bonferroni familywise alpha .05 over **two** two-sided contrasts:
each confidence **97.5%**, critical quantile t(1-.05/(2*2), blocks-1).
Each contrast improves only if its own lower bound exceeds zero. Both must
pass to claim repetition for these two fixed seeds. The harness's historical
three-contrast field is not this campaign's decision interval; derive the
two-contrast intervals from retained paired block values in the readout.

Aim at approximately 3 BB/100 half-width, using #226's 524,288-block primary
as reference. This is a provisional planning count, not a frozen sample.
Timing-only pilots use 32 and 512 blocks per contrast, full model loading,
play, replay and deterministic reproduction. Inspect time, memory and bytes
only; no pilot winnings or variances. Freeze sizes once from measured costs,
historical precision, memory and disk before any final hand.
No extensions, pooling or checkpoint selection.

Descriptive ordinary 95%: all three pairwise 4B/terminal cross-seed contrasts
among 2026100601, 2026100901 and 2026100902. Put each seed's ladder beside #226:
historical 2B-over-1B +29.51 (seed0601, #223 nominal95%), +28.14/+27.90
(seeds0901/0902, #226 adjusted98.3333%), and original 4B-over-2B
+17.90 [14.98,20.83] (#226 adjusted98.3333%). Do not rerun or pool historical
hands or label clean terminals as 4B when they stop earlier.

Reserved roots: direct timing **202610114501**, direct final **202610114502**;
scripted timing **202610114503**, scripted final **202610114504**.
Use distinct direct scenario names for every contrast. Verify physical deal
seeds against every earlier HU100 schedule, including #226 and newer relevant
merged/open campaigns, between pilot sizes (separate scenario suffixes) and
between new contrasts. Root reservation alone proves no freshness.
Candidate/comparator scripted arms deliberately share deals and opponent
streams; their policy streams remain private. Preserve the full freshness
inventory and coordinate counts before dependent play.

## Scripted safeguard: required

Candidate4B and v0.5.1's2B, both seed2026100601, each play all five opponents:
check_call, random, tight_aggressive, loose_aggressive and pot_pressure.
Translation on with #215's exact **512 states / 128 events** settings;
**4,096 duplicate blocks per opponent**, shared deals between checkpoints.
This panel is required and cannot be omitted to save time after outcomes.

Report each model's absolute BB/100 ordinary paired-block95% interval and each
paired 4B-minus-2B difference with a two-sided95% interval. **Severe regression**
means any opponent's difference95% **upper bound is below -20 BB/100**.
Retain all five results regardless of direction; absence of that flag is not
an equivalence or general-strength claim.

## Verification, quote and closeout

Independently replay and deterministically reproduce every final hand,
decision/probability/key/menu/translation witness and settlement.
Reconcile exact coordinates, counts and sum-zero accounting before readout.
Report coverage by policy/street/opponent and traverser visit bands
missing,0,1-9,10-99,100+, including exact and translated selected keys.
Conditional deal intervals do not estimate training-seed population uncertainty.

Post the **measured pilot-based time quote in the PR body before final play**,
with a SHA-bound posting receipt. Include measured completed training/load/
save/export/audit costs, fixed loading and variable play/replay/reproduction
costs, descriptives, required scripted panel, reporting and archival.
Use measured budgets with headroom, no arbitrary campaign cap.
Required primaries and scripted safeguard take priority if descriptive costs
require smaller prospectively frozen samples; no outcome-based scope changes.

Archive calibration must exercise source hashing, ZIP write, complete member
size/SHA256 readback and whole-archive hashing on **multi-GB actual campaign
bytes**, not extrapolate a tiny checkpoint. Include members of representative
size, and exclude calibration ZIP bytes from recursive packing.
Immediately before archive launch count actual bytes and members, refresh
throughput measurement if the workload differs, and freeze an allowance with
at least **3x the measured matching-workload projection**, plus measured fixed
cost. Account for disk and upload concurrency. #226's86.6s allowance failed;
its105.1s replacement completion is historical context, not a new quote.

One independent source review before final play and one end evidence review
are enough; resolve findings and run relevant checks. ZIP member-hashed evidence
into `~/Local/Research-Cloud/PR-<n>-hu100-4b-seed-replication/`, preserving
failures/partials, source/binary/environment and restoration provenance.
Record local readback, native uploaded/no-pending/no-conflicts and independent
Drive ID/name/size/parent acceptance separately under artifact-storage rules.
State whether remote bytes were downloaded. Retain originals and dependencies.

Stage intended files and run `python -m scripts.check_repository_artifacts`
before every commit. Commit/push completed subtasks. PR comments are only for
issues, each with one immediate SOMA; ordinary quote/status goes in the body.
Merge only when exact final-head checks are green and no findings remain.
Add one short Roadmap Current position entry when the completed work lands.
