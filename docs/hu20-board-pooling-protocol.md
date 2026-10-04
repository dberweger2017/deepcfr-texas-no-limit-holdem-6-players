# HU20 cross-board pooling protocol

Revision 3 records the October 4 owner-authorized M4 execution before any main
outcomes. Revision 2's corpus, 20/20 halves, cross-fitting, thresholds, replay
sample, bootstrap and classification point rule remain frozen. Only host,
runtime admission and worker shape change, plus the owner-requested prospective
coverage restriction and covered-context sensitivity. The machine-readable
[plan](../configs/diagnostics/hu20-board-pooling.json) and
[M4 budget](../configs/diagnostics/hu20-board-pooling-m4.json) record this revision.
#148 merged and released the M4. No pod was ever rented; the RunPod quote and
rental watchdog are superseded historical records. The $5 budget is unused.
No training, promotion or automatic merge. M1 is for development, retrieval and
reporting only; no solves. M4 owns all new qualification and production work.

## Question and prior evidence

The final [#145 report](reports/hu20-exact-turn-check.md) measures a conditional
turn-root game. Its pooled current/average means are 3.1958 BB blueprint and
0.6951 BB full-v1 projection. The stored-average subset alone is 2.0681 versus
0.6807 BB, R=0.3291: the pooled H2 label is not an average-only baseline.
The PR thread, including the [first Claude analysis](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/145#issuecomment-5959424711)
and [second Claude analysis](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/145#issuecomment-5962513514),
correctly identifies cross-board feasibility as unresolved. Its partial-run
numbers do not replace the final common-root report or set this protocol's
thresholds. Nested strategy classes constrain their optimal losses; they do
not force monotonic losses for these independently constructed projections.

## Outcome-blind sample

The census uses seed-1 B500M stored average, 3,000 fresh 20-BB deals, fixed
button=0, and seeds 202610030301 (deals), 202610030302 (actions), 202610030303
(board sampling). No turn action or payoff is sampled. A line retains exact
native action amounts and relative seats; only physical board identities are
removed. Choose the highest-occupancy line with at least 40 unique turn roots;
ties break lexically. Sample 40 without replacement from sorted root identities.
Never extend the census or replace a board after a solver value.

The selected line is button limp, big-blind check, flop check-check: 135 reaches
and 135 unique boards out of 1,075 live turns (1,925 earlier terminals). All
10,814 policy decisions hit retained keys. The root pot is 200 chips (2 BB),
stacks 1,900 each. The min-raise/call/check-check line had only 39 qualifying
roots. The frozen [corpus](reports/hu20-board-pooling-artifacts/corpus.json)
contains all occupancy counts, replayable events, weights and seeds. Each root
has weight multiplicity / inclusion probability, here 135/40=3.375.
This estimates a conditional census distribution from seed 1; it does not
represent every public line or unbiased occupancy of the other two lineages.

Use the three pinned B500M lifetime CFR-average exports from #141, separately.
For every root and lineage, derive private ranges with `public_ranges` from that
export's preflop policy, exactly as #145. Normalize as that function does; do not
substitute empirical ranges. Evaluate both target seats, average seats equally,
and average the three lineages equally within each board for the pooled readout.

## Primary and secondary losses

Every loss is responder best-response value minus responder equilibrium value,
in BB=100 chips and percent root pot. Report achieved exploitability and each
seat separately. The native `choices()` reopening tree is mandatory; no raise
cap, menu approximation or off-menu action translation is admitted.

1. **e_bp:** target locked to the actual immutable average export.
2. **e_root_v1:** own-range/own-equilibrium-action-reach projection onto actual
   full v1 keys within one root, pooling aliased public lines and river runouts.
3. **e_cross_v1 (primary P):** split the 40 boards into two seeded, frozen
   halves of 20. For each evaluation half, pool the other half's sufficient
   statistics **per lineage and actual full v1 hash**, multiplying board weight
   once. No evaluation-half action masses enter that policy. Apply it to the
   held-out roots and compute responder best responses. Fit the opposite
   direction and swap. The [split and replay sample](reports/hu20-board-pooling-artifacts/crossfit.json)
   records seeds 202610030307–309 and its hash is pinned in the plan.
   Positive/zero-reach contexts share the fitted policy. A key absent from the
   training half, or with globally zero training mass, uses uniform
   probabilities in its actual menu. Record missing-key coverage and target
   reach mass; never fill it from the evaluation half or blueprint.
4. **e_cross_eq50:** the same held-out policy procedure with a K=50 codebook
   fitted on the training half only: exact uniform-opponent turn histograms
   (20 bins), mean-centroid cumulative-L1/EMD clustering, and shared river
   equity-quantile edges. Assign held-out features using those frozen centers
   and edges. Equal features receive equal labels; independent per-root bucket
   numbering is forbidden. Seeds are 202610030305 plus training-fold index;
   codebooks are shared across lineages. No held-out features fit centers/edges.
5. **e_board_v1 / e_board_eq50 (secondary):** fit and evaluate on all common
   eligible boards, preserving the original optimistic in-sample comparator.
   Use a separate all-board codebook with seed 202610030305; never mix these
   estimates into the held-out primary D.

The template retains street, relative actor, complete v1 history and menu names.
Pool action masses by name, never by an unchecked solver index. Chance constants
cancel within a street because every root begins on the same turn line/deck
size. Preserve blockers and every own-range normalization. No board cards may
enter v1's pooled policy identity.

## Execution and gates

The AGPL upstream and new harness live outside this MIT repo. Pin upstream
9d1509fe5077d019825f833eed04b16d342dfda1 and native engine
5db20e3d5d6862b32a7402035c1340b622d3b005. Preserve #145's original external tool
and macOS binary cdc46b10d985d64747982ed1e3d40a1697d533cfbc444c0d16ed284cc4148952.
Copy and hash-verify the retained Mac v2 harness and original reference binary
into a fresh external directory. Preserve source/lockfile/toolchain fingerprints.
No AGPL source or binary enters this MIT repository. Linux parity is **not run**;
it is no longer an applicable admission gate.

Run #145 K/V1–V5 fixtures on M4 against the retained Mac references. V1 requires zero tree/chip
mismatches, V2 exact terminal settlements, V3 river toy MES/native BR agreement,
K at least 100,000 full-key/factored-key comparisons with zero mismatches. V4
must additionally use the real exports (20,000 native range deals per admitted
pilot), solver EV within the Monte Carlo 95% interval. Fixed pilots are the
first, middle and last corpus roots in frozen order, seed-1 average. Target
convergence <=0.2% root pot; reject above target, including 0.2–0.5% cases.
All #145 payoff and root-centering conventions remain unchanged.

Cross-board gates add a two-board contradictory-strategy oracle, singleton
reproduction of full per-root v1 projection, exact menu/actual-key coverage,
shared equity label identity, and a same-host reference comparison of sufficient
statistics and relocked losses. Compare exact trees/keys/hashes discretely;
floating probability/EV/statistic parity tolerance is 1e-5 of root pot (EV)
and 2e-5 absolute (probabilities / normalized masses).

Avoid enormous per-hand profile dumps or game snapshots. Phase 1 solves each
root/lineage once and atomically writes sufficient statistics, local losses,
equilibrium EV and convergence evidence. After all 120 outcomes, freeze the
common eligibility mask using only support, gates and convergence. Fit both
training-half policies and the secondary all-board policy on that common mask.

Phase 2 builds each native tree anew, allocates under the same budget, locks
the target policy at every target node and calls `compute_mes_ev` with **zero
CFR iterations**. Subtract the responder's hash-linked phase-1 equilibrium EV;
no fresh uniform-profile EV is an equilibrium reference. BR against the fully
locked target must not depend on the responder's strategy. Validate this
against solved-tree BR on fixtures, including both responder seats.

The frozen replay sample has three boards per half, six boards × three
lineages = 18/120 jobs (15%). Deterministically re-solve only these jobs with
the original iteration count, binary, compression and thread count; compare
EV, residual, sufficient statistics and all lock-only BR measurements. Preserve
both responses. A mismatch stops the campaign without relaxing tolerances.
No replacement replay root after outcomes; a support-excluded sampled root is
recorded as excluded. Lock-only completion is labelled evaluated, never solved;
its convergence qualification is inherited explicitly from phase 1.

Use one worker with six native Rayon threads, nice 10; freeze this count before
pilots and keep it for the 18 replay jobs. Each worker has a 5-GiB owned RSS
ceiling and a 4-GiB native arena ceiling checked before allocation. The family
cap is #148's `macos_memory_admission`: min(8 GiB, 0.8 * (free + inactive +
speculative + file-backed) - 0.5 GiB sidecar). Reuse `validate_admission`,
`RunBudget`, `machine_snapshot`, `rss_for_tree` and the existing swap guard;
do not introduce another memory law. Two workers are not admitted in revision 3.
Record other processes, pressure and disk before admission; leave them alone.

Approved cumulative allowance is 24 M4-hours including preparation, tests,
failures and retrieval. Record one append-preserving clock and an unreset
24-hour absolute deadline (a conservative bound including idle time). Reserve
one hour for retrieval/reporting. Keep >=20 GiB free disk, stop on swap growth
>1 GiB, any RSS/clock/disk/nonfinite/gate failure, and preserve all partials.
No deletion, automatic restart, replacement root, raise cap or science change.
Compression and nonzero-range trimming only. If the current disk does not fit,
wait or ask the owner; increasing future capacity is not present admission.

The first/middle/last frozen seed-1 corpus roots are mandatory real-export V4
and memory/convergence pilots (20,000 native deals each). They are limped
2-BB pots with 19-BB stacks: record both memory estimates, achieved residual,
solve/lock time and peak RSS. Before production post the conservative forecast:
1.5 * (slowest solve * 138 + slowest locked pass * 138), one worker. This must
fit the remaining allowance including reserve. If pilots fail or exclusions
would violate >=32 common boards / >=16 each half, report before production.

Interleave boards and lineages with unchanged seed 202610030306; retain atomic
results, append-only progress, failure records, estimates and resource history.

## Frozen analysis and thresholds

Only boards eligible and completed across all three exports enter primary
inference. Require the complete frozen campaign, at least 32/40 common boards
and >=80% frozen board weight, including at least 16 common boards in each
frozen half. Otherwise report descriptive values and no
hypothesis decision. Record all missing/excluded/oversize roots without
replacement, and the reason and retained weight. This mask is common to all loss estimates; each held-out policy fit uses
only the opposite half of that mask.

Let B, L, P be weighted means of e_bp, e_root_v1 and **e_cross_v1** (held-out). The placement
is D=(P-L)/(B-L), defined only when B-L>=0.1 BB. D>=0.7 is board-pooling
consistent; D<=0.3 is trainer/coverage consistent; otherwise mixed. Below the
gap floor no attribution is made. Report signed P-L, B-P, all BB/pot means,
P/B, L/B, equity/v1 headroom and D. Do not clip negative differences, D<0,
D>1 or residual-sized losses. Do not present the differences as an identified
causal decomposition or high P as proof that the abstraction cannot improve.

Bootstrap independent boards 2,000 times with seed 202610030304, preserving
lineage/seat pairing and board weights; use ratios of weighted means. Report
pooled, lineage and seat results. Intervals are **conditional on the fitted
two training-half policies/codebooks**, which are not refit and re-solved in every bootstrap
draw. They do not include policy-fitting uncertainty. Classification uses the
frozen point thresholds; show intervals and whether they cross thresholds.

### Missing-key coverage restriction (prospective revision 3)

Report own-target-policy decision reach and missing/zero-training-mass reach
per evaluation fold, lineage and street. Aggregate using frozen board weights
and both target seats. If the missing fraction exceeds 5% on either street in
any fold/lineage, or coverage is unavailable, primary D is **descriptive only**.
Also report totals across streets, whose scale differs by chance-node count;
the stricter per-street restriction prevents river context count masking turn
coverage. Coverage uses the harness's retained own-policy reach convention,
not joint reach under the best response.

Report `D_covered` with the same B/L denominator and paired-board bootstrap:
lock the held-out policy on covered actual keys and the per-root v1 witness on
absent/zero-mass keys. Its loss `e_cross_v1_covered` is a full-game hybrid
sensitivity isolating changes at covered contexts; it is not a conditional EV,
a causal decomposition, a new primary fit or a coverage-gate bypass.
The main policy retains its frozen uniform fallback unchanged.

## Solver-free companion

Select the top 20 excess-fold keys in each #145 Set A/B final supplement,
deduplicate their union, and record selection before reading retained node
statistics. Hash-verify the original #141 500M checkpoints and stream selected
rows only: visits, signed regrets, normalized retained average, zero-average
mass, current regret-match strategy and pairwise lineage TV by named action.
Do not resume or train. Stream retained #145 overfold contexts to count distinct
root boards feeding each selected key. This is observed diagnostic diversity,
**not historical training board occupancy**, which the checkpoint does not
retain. Keep missing keys/inputs explicit; no lower-milestone substitution.

## Limits and closeout

Ranges are taken as given; preflop mistakes are excluded. Turn/river conditional
values cannot answer the flop question, inherited aliases across other root
lines, full-game strength or which trainer mechanism is faulty. Both projections
are feasible witnesses on the sampled corpus, not abstraction equilibria or
lower bounds. Corpus-fitted equity buckets are more favorable than fixed global
blueprint buckets. Primary policies are fitted on one half and evaluated on the other. This
removes direct action-value fitting leakage for the 40-board sample, but does
not establish coverage of all boards or independent uncertainty across fitted
policies. Report the optimistic all-board comparator separately. Absent-key
fallback can increase held-out loss; show its coverage rather than treating
that increase as an abstraction lower bound.

Retrieve all raw attempts, results, fingerprints and resource journals to M1;
verify every member against the M4 manifest. Retain raw evidence in the dated
M4 folder for Drive archiving under RESULTS_INDEX. Do not delete anything or
touch CloudStorage/GoogleDrive. Cost is zero paid compute; $5 unused. Report
every failure, exclusion, interval and cannot-classify result. Update ROADMAP,
report and PR, keep draft. State the next 0.4.x implication explicitly, limited
to this seed-1-selected limp/check-through line, not raised pots or all boards.
