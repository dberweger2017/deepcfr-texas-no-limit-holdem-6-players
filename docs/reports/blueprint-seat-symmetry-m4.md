# Blueprint seat symmetry and learned-value audit

Draft PR #109 repairs the checkpoint lookup boundary and measures the saved
5.83M- and 12M-entry blueprints. The [frozen protocol](../blueprint-seat-audit.md)
was committed before the outcome runs. The [artifact index](blueprint-seat-symmetry-m4-artifacts/artifact-index.json)
contains full checkpoint, source, plan, archive and run-file hashes. The
[5.83M analysis](blueprint-seat-symmetry-m4-artifacts/analysis-5m.json) and
[12M analysis](blueprint-seat-symmetry-m4-artifacts/analysis-12m.json) contain
all block rates, intervals and reached-decision counters.

## What the repair establishes

The old information key mixed button-relative actor/history coordinates with
an absolute-seat folded/all-in vector. Native-engine tests couple the deal,
stacks and actions across all six rotations, including folds, all-ins,
off-menu raises and all streets. The versioned `button-zero-compatible-v1`
lookup produces equal keys and table probabilities for those rotations while
preserving the original button-zero hash and probabilities. It requires a
hash-pinned, button-zero-trained checkpoint. Legacy lookup remains the default
for historical reproduction. Tests also swap unseen opponent cards and check
that a hero's key and distribution do not change.

Coverage improves on **the same observed decisions**. On the unchanged
`U_safe` trajectories, the counts over all streets and seats were:

| Checkpoint | Hero decisions | Legacy trained hits | Compatible trained hits | Compatible-only hits | Legacy-only hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5.83M | 15,425 | 2,646 (17.2%) | 7,175 (46.5%) | 4,529 | 0 |
| 12M | 15,425 | 2,900 (18.8%) | 8,328 (54.0%) | 5,428 | 0 |

At button zero, both keys agree. Outside button zero, some keys happen to
agree when the status vector is rotationally symmetric. The extra trained
hits do not themselves establish a playing-strength gain.

## Paired play

Each checkpoint ran 1,024 independent six-rotation blocks against the frozen
scripted pool and 256 blocks against random opponents, with seven arms on the
same deal and opponent assignments per rotation. Each run retained all 53,760
hand attempts; all completed legally with conserved chips. Results below use
**BB/100**, with the six rotations clustered into one block. The primary
scripted contrast uses a two-sided 97.5% Student-t interval for each
checkpoint, giving at least 95% family coverage across the two primary
claims. Other intervals are exploratory, unadjusted 95% intervals. The
random suite is secondary.

### Scripted pool: paired differences

| Contrast | 5.83M, estimate [interval] | 12M, estimate [interval] |
| --- | ---: | ---: |
| **Canonical safe − uniform safe** (primary) | **+2.75 [−77.87, +83.37]** | **+36.58 [−51.87, +125.04]** |
| Canonical − legacy | −58.9 [−111.6, −6.2] | −27.3 [−83.0, +28.4] |
| Canonical safe − legacy safe | −30.4 [−85.7, +24.9] | −14.1 [−71.8, +43.7] |
| Uniform safe − uniform | −25.8 [−76.7, +25.1] | −25.8 [−76.7, +25.1] |
| Canonical safe − TAG | −563.8 [−670.1, −457.4] | −529.9 [−635.1, −424.7] |

Neither primary interval excludes zero. Thus this audit does **not detect an
additional learned-policy advantage** after controlling free folds; it does
not prove that the checkpoints learned nothing. The canonical-versus-legacy
contrast is not a demonstrated improvement. The unsafed 5.83M contrast is
negative under its exploratory, unadjusted interval, while the safe contrasts
remain inconclusive. The no-free-fold wrapper changes distributions often,
but `U_safe − U` is also inconclusive. Both blueprint-safe arms remain far
below the existing `tight_aggressive` hero on this scripted pool.

### Scripted pool: absolute returns

| Arm | 5.83M, estimate [95% interval] | 12M, estimate [95% interval] |
| --- | ---: | ---: |
| `U` | −541.1 [−634.7, −447.4] | −541.1 [−634.7, −447.4] |
| `U_safe` | −566.8 [−668.0, −465.7] | −566.8 [−668.0, −465.7] |
| `B_legacy` | −527.7 [−618.8, −436.7] | −508.1 [−598.1, −418.0] |
| `B_legacy_safe` | −533.7 [−630.5, −437.0] | −516.2 [−612.8, −419.6] |
| `B_canonical` | −586.6 [−680.4, −492.8] | −535.4 [−627.4, −443.4] |
| `B_canonical_safe` | −564.1 [−664.2, −464.0] | −530.3 [−628.4, −432.1] |
| `TAG` | −0.3 [−42.2, +41.5] | −0.3 [−42.2, +41.5] |

### Random opponents

| Arm | 5.83M, estimate [95% interval] | 12M, estimate [95% interval] |
| --- | ---: | ---: |
| `U` | +160.4 [−113.1, +434.0] | +160.4 [−113.1, +434.0] |
| `U_safe` | +169.6 [−107.9, +447.1] | +169.6 [−107.9, +447.1] |
| `B_legacy` | +192.9 [−120.5, +506.3] | +105.2 [−200.6, +411.0] |
| `B_legacy_safe` | +198.4 [−117.6, +514.4] | +105.3 [−204.4, +415.0] |
| `B_canonical` | +250.0 [−71.0, +571.0] | +176.4 [−131.1, +484.0] |
| `B_canonical_safe` | +269.0 [−52.7, +590.8] | +185.8 [−126.7, +498.2] |
| `TAG` | +192.1 [−22.1, +406.3] | +192.1 [−22.1, +406.3] |

The random-suite primary-style paired effects are +99.5 [−85.3, +284.3]
and +16.2 [−174.5, +206.8] for 5.83M and 12M respectively, with
unadjusted 95% intervals. Every prespecified secondary contrast is in the
machine-readable analyses. None establishes a reliable random-opponent gain.

## Work actually reached during play

The table below uses the same `U_safe` observations for both lookup modes.
Each cell gives action-frequency-weighted trained-hit rate; distinct-key hit
rates within street/button groups are retained in the analysis JSON.

| Street | 5.83M legacy | 5.83M compatible | 12M legacy | 12M compatible |
| --- | ---: | ---: | ---: | ---: |
| Preflop | 24.2% | 58.9% | 25.6% | 65.2% |
| Flop | 8.3% | 39.9% | 10.7% | 51.4% |
| Turn | 2.1% | 9.5% | 4.3% | 19.0% |
| River | 1.0% | 4.9% | 1.8% | 8.0% |

For the canonical-safe player's own trajectories, its 12M checkpoint found
6,265/9,594 preflop decisions, 1,644/3,237 flop, 380/1,727 turn and
107/1,006 river decisions. Among **found** canonical-safe decisions, one
recorded update accounted for 10.6%, 20.1%, 52.1% and 57.0% by street;
at most two updates accounted for 16.6%, 32.6%, 69.7% and 76.6%.
Missing keys are separate from those thinly updated hits. On the same
`U_safe` observations, 12M compatible distinct-key hit rates within
street/button groups were 58.8% preflop, 48.0% flop, 18.6% turn and 8.0%
river. The [analysis files](blueprint-seat-symmetry-m4-artifacts/) retain
every button, street and lookup-mode histogram.

At check-legal decisions, the wrapper removed a mean FOLD probability of
24.78% for `U_safe` (4,213 eligible decisions), 21.19%/20.87% for
5.83M/12M `B_canonical_safe` (4,397/4,324 eligible), and
23.85%/23.92% for `B_legacy_safe` (4,171/4,152 eligible). These are
**distribution mass removed per eligible decision**, not observed free-fold
action rates. The wrapper selected no free fold by construction and by test;
it retained FOLD when checking was unavailable. It encountered 18/41
all-mass-on-FOLD canonical cases and selected CHECK deterministically.
The final replay directly counted selected free folds:

| Arm | 5.83M, selected / check-legal | 12M, selected / check-legal |
| --- | ---: | ---: |
| `U` | 923 / 3,534 (26.1%) | 923 / 3,534 (26.1%) |
| `U_safe` | 0 / 4,213 | 0 / 4,213 |
| `B_legacy` | 884 / 3,534 (25.0%) | 884 / 3,520 (25.1%) |
| `B_legacy_safe` | 0 / 4,171 | 0 / 4,152 |
| `B_canonical` | 834 / 3,783 (22.0%) | 816 / 3,705 (22.0%) |
| `B_canonical_safe` | 0 / 4,397 | 0 / 4,324 |
| `TAG` | 0 / 1,256 | 0 / 1,256 |

Policy actions alter later histories, so these denominators differ across
arms. The action-count files also retain all selected action types; an
overall fold count would include folds facing a bet.

## Whole-checkpoint density and resources

The [streaming density report](blueprint-seat-density-m4.md) measured every
row of all three immutable checkpoints. Median visits remained **one** in
5.83M, 12M and 58.02M; exactly-one-visit shares were 93.37%, 90.98% and
83.17%. The 58.02M checkpoint had **58,015,659 entries and 92,086,645
recorded visits**, mean 1.587, p99 11, with 224,425 entries (0.387%) at
20 or more visits. Its 59.85% near-pure current-policy share is descriptive,
not a strategy-quality score. These are whole-table counts; no street label
was inferred from hashed keys. The 58.02M file was streamed, not loaded for
play under the M4 limit.

The final replay took 98.84 seconds at 4.99 GiB peak process RSS for
5.83M and 135.14 seconds at 7.95 GiB for 12M, including checkpoint load.
Both were below the 10.5-GiB process cap and the ten-hour overall limit;
system swap remained 769.38 MiB before and after both replays. Memory
pressure was recorded independently in each manifest and result. No paid
host or new training was used.

## Replay correction and limits

The first complete outcome runs used source `5942078`, but a list/tuple
comparison in the coverage summarizer left decision-weighted visit histograms
empty. Their 53,760 hand rows per checkpoint were retained in the
`initial-*` archives. Source `1fdc881` fixes the telemetry and checks both
histogram totals; the same frozen plans and seeds were replayed into
`replay-*` archives. Source `3f234d1` adds direct selected-free-fold counters
and ran the unchanged plans again into `final-*` archives. After excluding
elapsed seconds and each row hash that includes timing, **all 53,760 stable
hand rows matched across all three runs for each checkpoint**, with zero
mismatches. Every archived run file passed its own SHA-256 check, and the
final analyses validated the complete
arm/rotation/block pairing, chip conservation and legal completion. There
were no failed or partial hand attempts. The [artifact index](blueprint-seat-symmetry-m4-artifacts/artifact-index.json)
identifies all three source revisions and both stable-outcome digests. To reproduce
an analysis, extract a `final-*.tar.gz` archive, then run
`python -m scripts.report_blueprint_symmetry` with its matching frozen plan,
extracted run directory and an output path.

The fixed-button repair is verified; its measured coverage gain is large.
The strength comparison remains conditional on these two checkpoints, the
scripted pool, the random suite and this six-player abstraction. It does not
establish v0.5 strength, a model promotion, or a reason to discard the exact
river solver. The independent 3,000-hand diagnostic cited in the task used
a different method and is not treated as confirmed here.

**Recommended single next intervention:** design a small, correctness-gated
training-sampling change that concentrates *repeat updates on actually
reached postflop information sets* at a fixed traversal budget, then evaluate
it on fresh paired hands with this compatible lookup and safe uniform
control. The low late-street hit rates and one/two-update shares support
prioritizing useful revisits over simply growing the table or renting more
compute. This recommendation is not authorization for a new training run;
its unbiasedness and resource budget need a separate protocol. PR #108's
unstarted blueprint-dependent river confirmation remains deferred, while its
solver, independent evaluator and historical results remain intact.
