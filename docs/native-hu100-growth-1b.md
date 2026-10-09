# HU100 growth to 1B: prospective campaign protocol

Owner authorization: the October 8 task authorizes this campaign on the free M4,
including normal green-check merge. Base main: 9da625f9b376dcf02420680a2d4cb68bf4bd1b2c.
Isolated checkout: ~/Local/hu100-1b-growth-20261008; feature/hu100-1b-growth.
Read AGENTS.md, ROADMAP.md, rules, compact-table, #203/#204 growth, diagnosis and
#205 translation reports before execution. No native trainer source changes.

## Fixed science

Train fresh with seed 2026100601, HU100, one root/seat, opponent-sampled averaging,
linear CFR, existing uncapped menu, abstraction and uniform zero/missing fallback.
First reproduce the 7,642,767-entry capacity stop: checkpoint SHA256
792a675ce6d45d4de8d1b7f3fc6d976548610810f11da8f336c3926f37f8d416,
average SHA256 ba62d13536120a9d549f2f3ff84bcb2a96fbd8143ac2fc8477addab368dee0c4.
Any difference stops the campaign. This recreates the 39,438,279-node parent;
no previous research archive or active work root is copied.

Main training is a separate fresh run to 1,000,000,000 requested nodes,
with saves at 100,000,000, 250,000,000 and 500,000,000 requested nodes.
Each save is the first complete iteration at/past its target, or a valid
complete-iteration capacity stop. Keep actual nodes and entries, throughput,
write times, sampled family peak and kernel process peak. Forecast memory with
110 bytes/entry + 100,000,000 bytes; no hash-map doubling allowance.
Streaming current/average export and full checkpoint audit cover every save.
Do not chain training through gate/pilot/old checkpoints.

The experiment compares the parent with every actual saved milestone and
the terminal (including a clean capacity endpoint if 1B is not reached).
Translation is disabled in that full curve. The primary family contains only
terminal minus parent against tight_aggressive and loose_aggressive:
two-sided paired Student-t 97.5% intervals, Bonferroni FWER .05 for two tests.
An improvement requires lower bound >0; report effect size and both limits.

Separately predeclared secondary: terminal translation on minus off against
pot_pressure, ordinary paired 95% Student-t interval. TranslationOptions uses
#205 defaults (512 states, 128 events), unchanged weights and exactly one
private policy draw. No pooling of this secondary with the primary family.
Other opponents, intermediate gains, coverage and visit bands by street are
descriptive; no checkpoint selection or sample extension.

Use unchanged #203/#204 arena, independent engine replay and strict reporting.
Five scripted opponents: random, check_call, tight_aggressive, loose_aggressive,
pot_pressure. Heads-up stacks reset to 10,000 chips, blinds 50/100, no rake.
Every independent block has both swapped seats. Candidate private action
streams are paired across all policies; evaluator roots/decks/other hole cards
are never passed to policies. One uniform reference is reused exactly.
Fully replay and deterministically reproduce every final hand and decision;
latency is the only excluded reproduction field.

Pilot root 2026100810011; proposed final root 2026100810012.
Verify physical deal disjointness against #197 baseline, #200 curves,
#203 growth, #204 growth and all #205 pilot/final roots, and current pilot.
The cost-only pilot uses 16 blocks/opponent, excluded from final play. Its
outcomes are not inspected. Freeze the final sample and measured cost quote
before seeing final outcomes; 2,048 blocks/opponent/policy is the reference.
Never extend, pool, rerun failed science or select a checkpoint from outcomes.

## Resource and cost admission

Only Apple M4, 10 cores, 16 GiB, AC; one scientific worker family.
Fresh admission reserves at least 8 GiB system-free memory. Whole-family
soft ceiling 6 GiB (entry capacity floor((6 GiB - 100 MB)/110)); reaching
that soft training ceiling requests the native stop file and a valid save.
Hard family ceiling 8 GiB, pressure must remain normal with >=15% free,
swap growth <=512 MiB from this campaign's original baseline,
disk floor 15.5 GiB and AC throughout. Sample all guards at 200 ms target
cadence; retain actual timestamps. A hard breach stops and is reported on PR
with its single immediate next step (SOMA). No trainer fix in this PR:
a demonstrated trainer defect stops and is reported.

Measured gate training/save/export/audit and timing-only play/replay/reproduction
costs determine the quote. Extrapolate costs by nodes/entries with explicit
headroom, including larger-model loads and slow late training. Post the quote
in the PR body before final play. Main training additionally requires the
owner's requested readiness confirmation immediately before launch. No inherited 30-minute cap or arbitrary
time cutoff. Final execution uses the pre-play measured frozen budget.
Admission forecasts both retained originals and ZIP archives, with measured
pilot raw-hand storage and a fixed disk floor. A storage shortfall refuses
dependent work before crossing a guard. Do not clean other PRs or synced files.

One independent source review before final play; one evidence review at end.
Resolve findings, run relevant tests and repository artifact guard after staging,
push completed subtasks, merge only green exact-head checks/no open findings.
No paid compute, recipe/menu/abstraction change, release, tag or unattended work.

## Storage and reporting

All evidence, partials, environment/build/runtime, source snapshots, checkpoints,
exports, raw hands/reproduction, guard receipts and reviews belong in member-hashed
ZIP archives with embedded manifests under ~/Local/Research-Cloud/PR-<n>-hu100-1b/.
Verify archive/member readback and native upload status plus independent Drive
metadata; index exact model member paths, SHA256s and restoration commands in
RESULTS_INDEX.md. Keep large research bytes outside Git and all originals.
No extra metadata seals; compact later acceptance/review receipts remain in Git.
PR/report explain problem, design, evidence and limits in plain language.
Update Current position with one short entry when the work lands.


## Measured preparation, before training

Both exact gates and full audit passed. The 16-block/opponent pilot at
26a8ed429ea186d45d4ad70d25c0ac381b643d49 fully replayed/reproduced; no
winnings inspected. Freeze 2,048 blocks/opponent/policy. Measurements:
train/save 40.01s (native nonsave 16.76s, save 22.35s); streaming export
75.80s; full audit 171.41s; four policy loads 128.59s and scalable
play/replay/reproduction 3.437s. Main projected upper budget: training
42.50min (6x early gate node cost for later cache slowdown), saves 10.86min
(2x entry scaling), exports/audits 120.08min (2x), play/replay/reproduction
135.05min (3x loads/blocks scaling), closeout reserve 10min: **318.49min**.
This is an upper admission quote with headroom, not a 30-minute cap.
Nominal save/tool/play projections total about 2h20 plus training and closeout.

Memory entry ceiling **57,658,644**, leaving 2 GiB below the hard family guard.
The 1B entry extrapolation is 45,156,725, about 4.73 GiB forecast family RSS.
Storage requires **45.65 GB additional**, including retained original model
sets, arena model snapshots, final raw/reproduction, a complete ZIP copy and
512 MiB fixed reserve. Require **62.30 GB /58.02 GiB free** immediately before
training to preserve the 15.5 GiB floor. At the corrected preparation snapshot
50.09 GiB was free, 8.52 GB short; the owner is making space. The original
incorrect raw forecast scaled fixed model snapshots by blocks and is preserved
alongside the corrected receipt; it admitted no training. A duplicate prior-root
setup refusal also preceded every pilot hand.

## Owner-authorized archive-only retry — October 9

After all scientific work passed, the original ZIP packer stopped on swap growth
829.38 MiB above the fixed 563.56 MiB baseline, exceeding 512 MiB. Cleanup raised
PermissionError after termination and suppressed its normal receipt. The original
failure latch, raw guard samples, traceback and 3,354,661,317-byte synced partial
ZIP remain. No scientific operation is retried and no synced file is replaced.

The owner then authorized: “You can use up 3 gb swap. You can try again”.
The separate archive-only retry conservatively caps **total swap at 3,000,000,000
bytes**, which is stricter than a 3 GiB growth allowance, while retaining the
original baseline. All RSS, pressure, disk and AC guards remain unchanged.
Readmission binds the original failure, baseline and verified scientific completion
hashes to one exact archive command; training and final-play commands remain blocked.
The cleanup path now writes a failed receipt even when termination raises.

Measured uncompressed contents are 21.66 GB, including frozen failed-operation
logs. Exclusive creation of a new ZIP in the same native PR207 folder leaves
8.03 GB above the disk floor even without compression, preserving the old partial
separately with its hash in the new manifest. The partial SHA256 took 1.29s
(2.60 GB/s); the first attempt wrote 3.35 GB within 14.21s including manifest
hashing. Charging three full-size reads plus two conservative writes estimates
**209s local closeout**, or **418s with 2x headroom**. Upload completion is checked
separately; no fixed timeout substitutes for the resource guards. Mutable current
archive lifecycle, cloud acceptance and the final evidence review remain compact
Git receipts. The original scientific source stays bd0e7a417064f736091dc2b667954b50becb4b69.

## Owner upload handoff — October 9

After the retry ZIP passed all 948 member checks, the owner explicitly instructed:
“You just have to start the drive upload, don’t wait for it to finish I will delete
the originals as soon as it’s uploaded not before making sure the hashes match”.
The task therefore closes with native upload started and final cloud acceptance
pending for the owner. Record local ZIP/member hashes and a single guarded native
status snapshot; retain all originals and partials. No agent deletion or background
campaign waiter follows. The existing storage contract's upload-acceptance wait is
superseded only for this owner-directed handoff, not represented as satisfied.
