# HU100 independent seeds: separately admitted stages

Prepared October 9, 2026 from current main `6e18043317817080fd38f400c5366fbf18fc6b53`.
The owner authorizes this new campaign, maximum six hours of M4 computation
including calibration and local closeout. Leave its new PR unmerged.
Merged #211 retained a 1M-node pilot; it did not qualify two new 1B seeds.

## Fixed experiment and interpretation

The unchanged #207 recipe and #211 statistical protocol apply. Heads-up 100 BB,
50/100 blinds, reset 10,000-chip stacks, no rake, one root per seat, linear CFR,
opponent-sampled iteration-weighted averaging, uncapped existing action menu,
HU100 v1 abstraction and uniform zero/missing fallback. Native trees and binary
must match #207's frozen source and `7650ad60bbf2437622ea3c39d37c7d56686d00bac11680e44a6e47dab509a262`.
No algorithm, rules, observation, default-policy or release change.

Three fixed lineages: existing seed 2026100601 (39,438,279 and 1,000,002,065
nodes); fresh 2026100901 and 2026100902, sequentially, first complete iteration
at/past 39,438,279 and 1,000,000,000 total nodes. Save exact recovery at both
endpoints, streaming current/average exports and full audits. Resume seed0901's
#211 partial only after indexed size/SHA256/full-audit validation and byte-exact
actual-partial-resume versus fresh-direct training to 2M total nodes. This
fixture is qualification evidence, not an extra campaign endpoint. Main seed0901
continues the original 1,000,373-node partial, not the qualification fixture.
Check #207 and #211 live merged status before model retrieval.

If evaluation fits, all three lineages use fresh identical paired deals and
private action streams. Five opponents: random, check_call, tight_aggressive,
loose_aggressive, pot_pressure. Early/off, terminal/off, terminal/on; one uniform
reference reused exactly. Full independent engine replay and deterministic
reproduction of hands, probabilities and classifications, excluding latency.
Translation limits remain 512 states/128 events and one private draw.

Six growth contrasts: terminal/off minus early/off against tight and loose for
each seed, Bonferroni FWER .05, paired two-sided Student-t 99.1666667% intervals.
Separate three-contrast translation family: terminal/on minus off against pot,
FWER .05, 98.3333333% intervals. No union-FWER claim. Lower >0 means improvement,
upper <0 decline, otherwise inconclusive; practical support requires lower
>10 BB/100. Also report descriptive 95% intervals and absolute play by seed.
Repetition requires all three seeds to improve for the stated contrast; overall
qualification requires all nine formal contrasts, all endpoints and integrity,
controls/storage/evidence checks. Training alone never establishes qualification.
Check_call/tight/loose terminal on/off actions/events/settlements must match
exactly. Random's ordinary paired 95% upper <-20 BB/100 flags severe regression;
absence is not noninferiority proof. Other comparisons are descriptive. Seed
variation is the three estimates/range/sample SD, with no pooled-hand/population
interval; common deals correlate estimates. Remaining absolute losses stay visible.

## Calibration, sample freeze and stage admission

Before main training, measure existing policies at **32 and 512 blocks** (16x
sample ratio), early/off and terminal/off/on, full replay/reproduction. Cost
receipts alone are inspected for admission; no winnings or variance-based sample
selection. Pilot roots **2026100911011/2026100911012**; final **2026100911013**.
Check physical deal disjointness against all prior schedules including #211.

Load one validated model per checkpoint into one worker, create fresh seeded
policy instances for every hand, explicitly set/reset translation per arm.
Validate full model hash/spec/training identity and game at every execution.
Use hardlinks to nonsynced canonical bytes for snapshots, record inode/path/hash
provenance and verify hashes at each boundary. Never mutate model bytes through
any alias. The canonical model is archived once; manifest records each alias,
size/hash and canonical archive member, with explicit reconstruction commands.
No copying/hydration of old cloud archives or modifications of old evidence.

Measure initial loading separately from validation, snapshot/model hashing,
fixed panel setup, block-scaled play/report/replay/reproduction and raw hashing.
Include final output inventory hashing (previous wall_seconds omitted it).
Use maximum measured variable seconds/block across both counts and arms, 3x
headroom; fixed setup/model costs are charged per arm, never per block. Charge
one load per each of six checkpoints, entries-scaled for new models, at 3x.
Strict final reporting reserves 3x #207 measured report time scaled by entries
or panel count/blocks; terminal visit data is scanned once for the union of
both options. Fixed and per-hand timings and measured sample slopes are reported.

**Training is admitted independently.** Before each new lineage require its
training/save/export/full-audit time plus 1,800-second closeout reserve and
its originals/archive/transient disk quote. Use 2x #207 measured nonsave train
cost and maximum save/export/audit seconds per entry at the unchanged entry
ceiling, 10% early-entry headroom. Do not refuse training because evaluation
cannot fit. At each endpoint preserve tools and closeout time before launching
training; stop file requests a complete-iteration save before that reserve.
Keep audited partials if an endpoint cannot finish; never replace a seed.

After training, admit evaluation against actual remaining clock/disk. Choose
**8,192 blocks**, otherwise **4,096**, only by measured cost; fewer refuses full
qualification. Freeze sample, models, schedule and comparison families with
hashes before any final hand. No shrinking/extending after outcomes. If neither
fits, retain all audited training endpoints for later independently authorized
evaluation and label qualification incomplete. Raw disk uses 2x measured larger
calibration bytes/block; originals plus one uncompressed archive and 3 GiB
transient reserve are counted, above the disk floor. Packing uses the existing
measured #207 costs and a fixed 1,800-second reserve. No extra computation after
expiry. Administrative M1 review/CI/cloud metadata are separately disclosed.

## Ownership, unchanged guards and preservation

Read-only October9 10:22 UTC inventory: M4 idle, no research lock holder,
10 cores/16 GiB/AC, about98 GiB free. This is reconnaissance, not admission.
Require exclusive `~/Local/.hu100-m4-research.lock` and fresh 301-sample/60-second
stable admission at 200ms target cadence, >=8GiB headroom. The single absolute
21,600-second clock starts before any model retrieval/hashing/fixture/calibration.
Baseline never rebases after a breach. Existing guards: **6GiB soft /8GiB hard**
whole-family RSS, **57,658,644 entries**, normal pressure and **>=15% system-free**,
**3,000,000,000 bytes swap growth** from fixed fresh baseline, **15.5GiB disk
floor**, AC, continuous timestamps including packing/readback. Hard/correctness
failure stops without retry; no silent relaxation. Process-group cleanup checked.
Source/protocol independently reviewed before dependent science and evidence
reviewed at closeout; relevant tests and exact-head CI required.

Store source/binary/environment, models/recovery, failures/partials, full raw
traces and resource logs in the owning PR's Research-Cloud folder. New archive
uses exclusive creation and full member readback, whole/manifest SHA256; record
alias restoration, current native upload/no-pending/no-conflicts and independent
cloud ID/name/size/parent separately. No remote-byte claim without download.
Keep originals, all #207/#211 failures and active dependencies, shared Git and
other open PR work intact. No cleanup, paid compute, changed recipe, automatic
larger run, schedule, release or merge. Commit source and compact evidence only;
staged repository artifact check before every commit.
