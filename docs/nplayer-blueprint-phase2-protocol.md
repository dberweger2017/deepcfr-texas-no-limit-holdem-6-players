# First three-player blueprint: Phase 2 protocol

Status: harness/admission and all three sequential training seeds complete.
Final regression closeout is running; final evaluation has not started.
The owner authorized this campaign on October 15, 2026. No search, release or
merge is authorized. PR289 and PR291 are already merged; the feature branch
starts at main `623e2e8741dd8c267f062ded28b9bcc5dde36a0f`.

## Fixed training and decision

Three sequential seeds **2026101541, 2026101542, 2026101543**. The owner chose
these fresh numbers after discovering the originally requested 1511–1513 were
used for Leduc. Before reservation, repository search and 19,760 small text
records in bb128's qualification and three Phase 1 roots found no uses of
1541–1543 (excluding build/environment/input directories and files >10 MB).

Each seed: three players, equal 2,000-chip stacks, 50/100 blinds, unchanged
K50 abstraction and `min-pot-conditional-jam-v1` menu, external-sampling linear
CFR, **all-street averages (scope 3)**, 1B traversal nodes. Save/export/audit at
200M, 500M and 1B, at complete-iteration boundaries. A 150M-entry capacity stop
is terminal; no frozen table. Record entries, visits/key and decision/missing
counts by street at every save. Every export row must pass the independent
probability audit. Resume may implement the scheduled milestones, never an
outcome-driven extension. All seeds run one after another.

**Learning is established only if, on all three seeds, the 1B model's two-sided
95% lower bound against two copies of its own 200M model exceeds zero.** All
other comparisons are descriptive, including 500M, scripts and cross-seed play.
These are conditional playing comparisons, not exploitability or a multiplayer
Nash-equilibrium guarantee. All final hands must independently replay and
exactly reproduce before readout. Failures and partials remain recorded; no
reruns or outcome-driven extensions.

## Hardware admission and stops

Only cygnus-x1 (`bb128`) computes; M1 is for light editing and small tests. No M4.
Check other campaigns before each stage. Until the owner confirms the VRM
heatsink is reinstalled: one job at a time, affinity CPUs 0–3 and all thread
limits <=4. The on-host supervisor owns the process group and an exclusive lock.
Every resource row includes nct6687 **Thermistor 15** and **fan1 RPM**. Missing
sensors refuse admission or stop the job. Start only below CPU 70°C and VRM 85°C.

Retain the existing conservative CPU **85°C immediate stop** (stricter than the
90°C ceiling). VRM >=95°C for 30 continuous seconds writes the child's stop
file; training finishes its iteration and saves, evaluation finishes its block
and retains its partial. Continue monitoring during a bounded 180-second clean
stop grace, then terminate the group if needed. CPU/resource/deadline breaches
still terminate immediately. A thermal clean stop is a failed campaign attempt,
not an automatic restart. Family RSS <=80 GiB, available RAM >=6 GiB, swap growth
<=256 MiB, free disk >=32 GiB. Per-stage wall caps and the total cost quote will
be fixed after bounded cost qualification and before campaign admission.

## Evaluation design

Each independent block uses a fresh domain-separated deck stream and all **six
permutations of three labeled players** into fixed physical seats. Identity 0
is the singleton A, identities 1 and 2 are independent policy instances of B.
The button is block modulo three. Deck and board remain fixed within the block;
each identity occupies each seat twice. Action RNGs are domain-separated from
deals and keyed by root/block/identity, resetting for each permutation. Policies
receive only acting-seat observations, never the root, deck or opponent labels.

Report overall, per-physical-seat and button-relative position (BTN/SB/BB)
BB/100 for the singleton and per-player pair. Compute one mean per independent six-hand block, then a two-sided 95%
Student-t interval over those block means. Never count the six correlated
hands as independent samples. Report role exposure and missing decisions by
street. The conjunction of the three declared primary lower-bound checks is
the learning rule; descriptive intervals are nominal and not simultaneous.

Controls: random uniform over the training menu; check/call; tight-aggressive;
pot-pressure. The latter two port the public hand-selection scores and
frequencies from `src/arena/heuristics.py`. Both raise one pot after calling,
clamped to legal bounds; pot-pressure uses the existing matched-pot variant,
not the off-menu 1.5-pot stressor. Card scores are heuristics, not equities.

Final matrix: for each seed, 1B singleton versus two own-200M and two own-500M,
plus each reverse lineup; each 1B versus each of the four bots in both lineups;
all six directed cross-seed 1B-singleton versus two other-1B comparisons. Fixed
roots, block counts, model hashes, one-off loading costs, repeatable per-hand
costs and total runtime caps will be committed **before any final hands**.
The PR291 scoped-average smoke model is only a harness proof/cost input, not a
Phase 2 model or a strength result. Its original provenance is in
[the Phase 1 report](reports/nplayer-size-smoke.md).

## Fixed harness qualification (before final hands)

After unit/integration tests, use the exact PR291 smoke export
`fcbe8bd5733572fe9f9046840d6e277a0b11ce0ac96936546df0815be218485e`.
Nine fixed cases: self-play, then random, check/call, tight-aggressive and
pot-pressure, singleton-model then paired-model for each bot. Roots
2026101551–2026101559 in that order; **64 blocks/384 hands per case**, twice
from scratch. Every first trace replays in the pinned engine and its complete
bytes must equal the second trace. This is 6,912 executed hands, 3,456 separately
replayed; no outcome-based choice or extension. Cap the combined qualification
at **900 seconds**, 80 GiB family RSS and the same thermal guards. Record load,
play and replay time separately; do not use poker outcomes to select final
counts. Synthetic tests and tiny model fixtures are separate correctness work.

## Final block reservation

Fix **8,192 independent blocks (49,152 hands) per contrast**, all six
permutations, then one complete fresh-process reproduction. The 42 contrasts
are enumerated by `scripts/nplayer_phase2_matrix.py`: first each seed's 200M
and reverse, 500M and reverse; then each seed's four bots in the stated order,
singleton-model and paired-model; then all six ordered cross-seed pairs in
ascending seed order. Roots **202610150801–202610150842** map to that order.
Repository search found no pre-existing uses before reservation.

Total: **2,064,384 final hands**, independently replayed, plus **2,064,384
reproduction hands**. Counts and roots are fixed before any Phase 2 model is
trained or final hands played. The smoke-model qualification measures cost
only. A separately committed admission manifest will pin the nine model
hashes and the measured/scaled cost quote before final hands. A missing audit,
failed replay, non-identical reproduction or any resource stop prevents a
settled readout. Keep all partial data and do not replace roots.

## Training admission quote

The nine fixed smoke-model cases passed all-hand replay and exact reproduction
in **263.62 seconds**, peak **2.339 GiB family RSS**, CPU **65°C**, VRM **66.5°C**,
fan1 **337–611 RPM**. The immutable 8,246,876-row table loads in about 12.5 seconds;
384 hands take roughly 0.03 seconds to play and 0.08 seconds to replay. The
[complete cost receipt](reports/nplayer-blueprint-phase2-artifacts/harness-costs.json)
separates these costs. Each case has 768 executed hands and 384 independent
engine replays, with no mismatch. No poker outcome was used to set final counts.

Clarification: Phase 1's **39.1 GiB** figure describes 100-BB **preflop** averages.
Its uncapped 20-BB **full-average** origin-linear scenario is **79.1 GiB** at 1B
nodes, with 245.4M entries. The owner's 150M-entry terminal cap limits that
scenario: the same measured layout and allocation allowances at 150M entries
project **43.41 GiB** including hash-growth overlap, vectors, sorting and K50.
This is a planning scenario, not an RSS guarantee. Keep the independent 80-GiB
guard; a capacity/resource stop is retained and does not authorize freezing,
changing averages, increasing limits or continuing the campaign.

At Phase 1's roughly 0.9M full-average nodes/s, 1B nodes costs about 19 minutes
of traversal per seed. Checkpoint writing/loading, three complete exports and
three every-row audits are additional, material one-off costs. With the local
power-fit endpoint of roughly 64M keys and the measured smoke export/audit
rates, plan approximately **45–100 minutes per seed including those costs**;
allow a fixed **3-hour whole-seed cap**, at most **9 hours across three seeds**.
Each scheduled training segment has its own **1,800-second traversal clock**;
checkpoint I/O is included in the whole-seed supervisor clock. Stop the campaign
on any failed seed or audit. No concurrent seeds or evaluation. Final evaluation
has a separately committed quote using actual row counts before final hands.

## Telemetry interpretation and execution boundary

Entry `visits` counts traverser regret-update visits, not opponent-average
updates and not iteration-weighted mass. All-street averaging can retain
opponent-only keys with zero regret visits; report those zero-visit shares as
well as the mean. A training missing decision is a lookup absent from the
iteration-start table, before that iteration's batched inserts; it is not the
same as the final policy's played missing-key rate. Each milestone audit gives
cumulative and since-previous-save missing/decision counts by street. Evaluation
reports actual played lookup misses separately, by identity and street.

All 42 final contrasts run sequentially inside one exclusive on-host supervisor
whose wall cap is the committed manifest's budget. Each contrast still loads
from scratch for its second pass. Any failed contrast or thermal/resource stop
ends the matrix; all-case readout also requires that outer guard to pass.
Archives retain source/model/raw-trace originals on bb128. ZIP member SHA256/size
and CRC readback precedes transport splitting; each 128-MiB byte part is hashed
and their concatenated hash must equal the ZIP. Small M1 transport buffers avoid
staging the entire campaign on the owner's nearly full laptop. Split archives
must be concatenated in indexed order before extraction.
