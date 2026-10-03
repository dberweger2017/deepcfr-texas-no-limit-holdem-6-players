# HU20 cross-board pooling protocol

Frozen before any new solver outcomes. The machine-readable plan is
[hu20-board-pooling.json](../configs/diagnostics/hu20-board-pooling.json).
Paid execution remains unapproved; the separate quote must be approved first.
No training, promotion or automatic merge. M4 belongs to #148 and is not used.
M1 is limited to development, short tests and the solver-free companion.

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

## Four losses

Every loss is responder best-response value minus responder equilibrium value,
in BB=100 chips and percent root pot. Report achieved exploitability and each
seat separately. The native `choices()` reopening tree is mandatory; no raise
cap, menu approximation or off-menu action translation is admitted.

1. **e_bp:** target locked to the actual immutable average export.
2. **e_root_v1:** own-range/own-equilibrium-action-reach projection onto actual
   full v1 keys within one root, pooling aliased public lines and river runouts.
3. **e_board_v1:** the same sufficient statistics, multiplied by board weight
   once, pooled across common eligible boards **per lineage and actual full v1
   hash**. Expand that one resulting policy back onto every root, then compute
   the best response there. Positive and zero reach contexts share the same key
   policy; globally zero mass uses uniform probabilities in the actual menu.
4. **e_board_eq50:** analogous pooling onto one shared equity codebook. Compute
   exact uniform-opponent turn histograms (20 bins), fit one K=50 mean-centroid
   clustering with cumulative-L1/EMD assignment across all 40 boards' compatible
   holdings (seed 202610030305), and one shared river equity-quantile edge set.
   Equal histograms/equities get equal labels. Independent per-root bucket
   numbering is forbidden. These features are outcome-free and frozen before
   equilibrium values; the codebook is shared across lineages.

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
Fingerprint new sources, lockfile, toolchains and macOS/Linux binaries. Build
x86_64 Linux natively on the quoted pod; no emulation. Publish an external-tool
manifest reusable by #148 without placing AGPL code in this repository.

Run #145 K/V1–V5 fixtures on both platforms. V1 requires zero tree/chip
mismatches, V2 exact terminal settlements, V3 river toy MES/native BR agreement,
K at least 100,000 full-key/factored-key comparisons with zero mismatches. V4
must additionally use the real exports (20,000 native range deals per admitted
pilot), solver EV within the Monte Carlo 95% interval. Fixed pilots are the
first, middle and last corpus roots in frozen order, seed-1 average. Target
convergence <=0.2% root pot; reject above target, including 0.2–0.5% cases.
All #145 payoff and root-centering conventions remain unchanged.

Cross-board gates add a two-board contradictory-strategy oracle, singleton
reproduction of full per-root v1 projection, exact menu/actual-key coverage,
shared equity label identity, and a cross-platform comparison of sufficient
statistics and relocked losses. Compare exact trees/keys/hashes discretely;
floating probability/EV/statistic parity tolerance is 1e-5 of root pot (EV)
and 2e-5 absolute (probabilities / normalized masses).

Avoid enormous per-hand profile dumps or game snapshots. Phase 1 solves each
root/lineage and atomically writes sufficient statistics and local losses.
After all 120 outcomes, freeze the common eligibility mask using only support,
gates and convergence. Build pooled policies on that mask. Phase 2 deterministically
replays each eligible solve to its phase-1 iteration count with identical
binary, compression and thread count; verify equilibrium EV, residual and
sufficient statistics against phase 1 before any pooled loss counts. Keep both
raw responses. Replay mismatch is a mandatory stop, not a relaxed tolerance.

Parallel independent workers only after measured cgroup/RSS admission. Inspect
memory estimates before allocation; compression then nonzero range trimming
are allowed, no cap. A guard failure, nonfinite value, parity/gate failure,
quote clock/cost exhaustion or oversize root stops owned production and
preserves partial evidence. No automatic restart. OS and aggregate worker
limits, worker count, disk, hard clock and retrieval/shutdown reserves belong
in the owner-approved resource quote. Never bypass guards to fill the corpus.
Interleave boards and lineages with seed 202610030306. Keep atomic results,
append-only progress, failures, RSS/memory/CPU and the unreset rental clock.

## Frozen analysis and thresholds

Only boards eligible and completed across all three exports enter primary
inference. Require the complete frozen campaign, at least 32/40 common boards
and >=80% frozen board weight. Otherwise report descriptive values and no
hypothesis decision. Record all missing/excluded/oversize roots without
replacement, and the reason and retained weight. This mask is common to all
four loss estimates and to the pooled policy fit.

Let B, L, P be weighted means of e_bp, e_root_v1 and e_board_v1. The placement
is D=(P-L)/(B-L), defined only when B-L>=0.1 BB. D>=0.7 is board-pooling
consistent; D<=0.3 is trainer/coverage consistent; otherwise mixed. Below the
gap floor no attribution is made. Report signed P-L, B-P, all BB/pot means,
P/B, L/B, equity/v1 headroom and D. Do not clip negative differences, D<0,
D>1 or residual-sized losses. Do not present the differences as an identified
causal decomposition or high P as proof that the abstraction cannot improve.

Bootstrap independent boards 2,000 times with seed 202610030304, preserving
lineage/seat pairing and board weights; use ratios of weighted means. Report
pooled, lineage and seat results. Intervals are **conditional on the fitted
pooled policy/codebook**, which are not refit and re-solved in every bootstrap
draw. They do not include policy-fitting uncertainty. Classification uses the
frozen point thresholds; show intervals and whether they cross thresholds.

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
blueprint buckets. The same boards fit and evaluate the pooled policy; report
this in-sample scope rather than claiming held-out generalization.

Retrieve all raw attempts, results, fingerprints and resources, verify member
hashes against the remote manifest, then terminate the owned pod and its
ephemeral storage. No network volume. Report actual billed/estimated cost and
any distinction, every failure and exclusion, source/input/output hashes,
companion limits and Linux qualification. Keep the PR draft until owner review.
