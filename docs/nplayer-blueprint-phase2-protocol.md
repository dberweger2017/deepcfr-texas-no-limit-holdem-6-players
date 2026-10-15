# First three-player blueprint: Phase 2 protocol

Status: implementation and cost qualification; no training or final evaluation yet.
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

Report overall and per-physical-seat BB/100 for the singleton and per-player
pair. Compute one mean per independent six-hand block, then a two-sided 95%
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
