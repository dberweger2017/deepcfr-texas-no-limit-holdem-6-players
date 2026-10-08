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
in the PR body before final play. No inherited 30-minute cap or arbitrary
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

