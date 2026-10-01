# B500M stored CFR-average extraction

## Result

All **38,400 frozen hands** completed and passed the model-free native audit,
including **85,370 target decision observations** and all paired report arithmetic.
No three-lineage panel change excludes zero, even before multiple-comparison
adjustment. The stored average is **not demonstrated better or worse** by this
bounded experiment. All negative and inconclusive results remain in the raw data.

| Panel | Current BB/100 [95%] | Average BB/100 [95%] | Paired average−current [95%] |
| --- | --- | --- | --- |
| Native pressure | +79.85 [20.36, 139.34] | +50.00 [−6.20, 106.20] | −29.85 [−72.42, 12.72] |
| Selective-stackoff | +50.65 [33.81, 67.49] | +47.56 [31.99, 63.12] | −3.09 [−13.95, 7.76] |
| Bounded LBR | −141.86 [−210.31, −73.42] | −92.32 [−158.35, −26.28] | +49.54 [−6.04, 105.13] |

Both extractions still lose to this bounded LBR. The favorable average LBR point
is a lead for a more precise future comparison, not an improvement certificate.
All **13 panels** and per-seed/position results are in the
[tables](hu20-cfr-average-artifacts/derived/tables.md),
[panel CSV](hu20-cfr-average-artifacts/derived/panels.csv) and
[paired-change CSV](hu20-cfr-average-artifacts/derived/paired-changes.csv).

Native-pressure seed changes are +36.72, −53.32, −72.95 BB/100; LBR changes are
−0.39, +100.98, +48.05. Every corresponding overall per-seed interval includes
zero. One adverse native-pressure seed-3 button contrast is −129.49
[−247.86, −11.13] BB/100; it is an exploratory position result among many
comparisons, not proof of a general positional defect. Do not cherry-pick that
position or the favorable LBR lineage.

## Scope and interpretation

I compare all three original B500M current policies against a diagnostic export
of the trainer's stored lifetime reach/iteration-weighted CFR average. This is
an extraction experiment, not a model promotion or new training. The production
short-stack current-only export guard, inference loader, engine, abstraction and
training code are **unchanged**. The implementation is separate in
[draft PR #141](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/141).

The [protocol](../hu20-cfr-average-protocol.md) preceded outcomes. Native pressure,
scripted styles, restricted reactive controls and LBR remain separate panels;
no comparison is pooled with #136 or earlier evaluation schedules. All inference
is observation-only. The new format is
`holdem-hu20-stored-cfr-average-diagnostic-v1`, extraction
`normalize-lifetime-iteration-own-reach-accumulator-v1`; production loading rejects
these JSONL artifacts and the production `strategy="average"` guard still rejects
short-stack export requests.

## What the retained checkpoints establish

At each traverser node, `solver._collect_root` adds
`iteration × own_reach × policy[action]` to the average accumulator. Own reach
multiplies only that player's earlier action probabilities; rival actions are
externally sampled. Complete iteration deltas add into `Node.average`. For
positive total mass, this diagnostic divides each action total by the sum.
It does not average saved checkpoint policies or reconstruct a recent window.
The result is the **stored CFR average for this trainer/abstraction**, not a proof
of exact equilibrium play or unbiasedness under every hypothetical sampling rule.

Before evaluation, all three original compressed checkpoints and current exports
passed length/hash checks. Every one of **8,026,069 nodes** passed accumulator,
normalization, current-regret extraction and coverage checks. Required invariants:
finite nonnegative average vectors, integer nonnegative visits, unique keys/labels,
matching game/schema/table/seed/iteration and total mass ≤ last iteration × visits
(relative floating tolerance 1e−9). The independent synthetic traversal test checks
exact own-reach/iteration factors across own and rival actions. All emitted
probabilities are checked against their retained accumulator vectors, and every
current vector matches regret matching on the retained regrets.

| Training seed | Iteration | Retained keys | Positive average mass | Zero mass | Unweighted current/average TV |
| --- | --- | --- | --- | --- | --- |
| 2026093001 | 1,098,929 | 2,701,878 | 622,174 | 2,079,704 | 0.47437 |
| 2026093002 | 1,110,069 | 2,670,044 | 608,189 | 2,061,855 | 0.47521 |
| 2026093003 | 1,096,178 | 2,654,147 | 605,553 | 2,048,594 | 0.47487 |

About **77% of retained keys have zero average mass**. Traversal enumerates own
actions even when their own reach is zero, so a visited node need not have average
mass. Zero totals have no defined normalized strategy: the declared diagnostic
uses uniform probabilities in their retained menu, and logs them separately from
missing-key uniform fallback. It does **not** substitute current strategy there.
“Trained” in existing tail counters means a retained key exists, including zero
mass; it is not a claim that the average has learned a distribution at that key.
The global zero-mass share and unweighted key TV are not played-state occupancy.

The final checkpoints do not retain every historic reach/policy increment; this
audit verifies the stored totals and bounds, not a replay of 500M training nodes.
No new training or retrospective reconstruction was attempted. All full audit
counts and artifact identities are in the
[extraction summary](hu20-cfr-average-artifacts/extraction-summary.json).

### Recorded technical interruption

The tool turn was interrupted while the last average-policy LBR panel was
running, terminating the original worker before its summary/peak-RSS publication.
**38,144 completed hand records** survived. Five gzip files were closed and
complete; the final average-policy stream retained 6,144 cheap-panel hands and
had no end marker. Its original truncated bytes are preserved separately.

The bounded completion script retains every completed record and runs only the
256 unrecorded frozen LBR coordinates. It verifies that gameplay/extraction code
is byte-identical to the original source, keeps the same deals/streams, and uses
the original source commit timestamp as a conservative earlier-than-launch
30-minute deadline. No completed hand or observed result is selected for rerun;
there is no new seed, budget extension or replacement of the interrupted evidence.
The recovery fixture checks unchanged original bytes, exact missing-coordinate
counts and complete native replay. The original worker's peak memory and load
times are unavailable; the recovery process is measured separately.

## Frozen schedule and evidence

The separate timing pilot used first-lineage current/average on 208 hands and all
13 panels: 32.08 seconds including loads, peak 2.670 GiB. Eight LBR hands took
1.992/3.615 seconds current/average. Slower measured LBR and cheap-panel rates,
six loads and 1.25× headroom projected about 18.3 minutes. Pilot payoffs are
excluded from final inference; only runtime selected the budget.

The [final plan](../../configs/diagnostics/b500-cfr-average-comparison.json) fixes
**38,400 hands**: three lineages × current/average × both positions on 256 fresh
deal blocks per cheap/style/native-pressure/stackoff panel, and 128 for LBR.
Each hand resets to 2,000 chips/seat (100 chips/BB), no rake/ante. Root
`202610050201` is disjoint from pilot `202610050101` and prior studies. Deals and
target/rival streams are identical across current/average and the three lineages;
seat rotations are paired within each block. Intervals use paired block means,
not individual correlated rotations, and condition on these fixed lineages.
They are exploratory unadjusted 95% Student-t intervals across many panels.

Uniform and selective-stackoff use the uncapped restricted native-reopening menu.
Passive, `minraise-cap2` and `pressure-cap2` retain the existing original-cap2
reactive control definitions; the target stays uncapped. Six existing styles use
their native legal sizing, as does native pressure (exact minimum raises without
the artificial cap). LBR uses the explicitly selected exact-ranker/shared-cache
implementation validated in #126/#138, four chance samples/holding and a
five-second soft batch budget. Each hand gets a fresh LBR/cache; the cache belongs
to its single immutable target source. It remains a bounded range/checkdown
opponent, not exploitability or an exact best response.

Positive paired changes mean average profit exceeds current profit. BB/100 equals
mean net chips/hand under this 100-chip BB; displayed stacks resetting per hand
do not reset the accumulated result. Larger wagers are raises demanding at least
800 additional rival chips (8BB); their denominators are target decisions with
such an action available. Full-stack wins/losses require exactly ±2,000 net chips.
Whole-hand response subgroups are **not individual-bet EV**.

### Coverage and large-pot tails

| Panel / extraction | Hands | Large actions / opportunities | Rival folds / continues | Full-stack wins / losses | Missing keys / decisions | Zero mass / decisions |
| --- | --- | --- | --- | --- | --- | --- |
| Native pressure / current | 1,536 | 465 / 2,516 | 138 / 327 | 205 / 145 | 15 / 5,840 | n/a |
| Native pressure / average | 1,536 | 475 / 2,443 | 135 / 340 | 198 / 144 | 19 / 5,810 | 97 / 5,810 |
| Selective-stackoff / current | 1,536 | 22 / 59 | 4 / 18 | 3 / 5 | 0 / 2,055 | n/a |
| Selective-stackoff / average | 1,536 | 13 / 58 | 4 / 9 | 3 / 4 | 0 / 2,106 | 1 / 2,106 |
| LBR / current | 768 | 146 / 625 | 68 / 78 | 45 / 81 | 0 / 1,909 | n/a |
| LBR / average | 768 | 162 / 693 | 78 / 84 | 60 / 73 | 1 / 1,980 | 9 / 1,980 |

The native audit recomputes these counters from exact menus/actions and chip
results. [Coverage](hu20-cfr-average-artifacts/derived/coverage.csv) reports every
panel's street and distinct-key counts. The
[large-action coverage CSV](hu20-cfr-average-artifacts/derived/large-action-coverage.csv)
separates positive-mass, zero-mass and missing-key opportunities/actions; the
[whole-hand partitions](hu20-cfr-average-artifacts/derived/whole-hand-partitions.csv)
retain exact subgroup denominators and chip totals.

Across the full scheduled inventory (not a pooled performance estimate), average
queries reach positive mass 42,074/42,849 times, zero mass 188/42,849 (**0.44%**)
and missing keys 587/42,849 (**1.37%**). Current missing-key exposure is
555/42,521 (**1.31%**). Thus the 77% global zero-mass key share is very different
from realized exposure in these panels. All LBR decisions complete their declared
chance-comparison batches; none is reported limited. This does not eliminate
the bounded LBR's modeling limitations.

### Resources and validation

| M1 stage | Wall seconds | Measured process peak |
| --- | --- | --- |
| Three extractions plus all-node/current audits | 195.98 | 3.143 GiB |
| Timing pilot | 32.08 | 2.670 GiB |
| Final schedule including interruption/recovery | 1,095.61 from conservative commit anchor | Original worker peak unavailable |
| Completion process (only 256 missing hands) | 171.21 | 2.625 GiB |
| Separate final model-free audit | 49.48 | 1.745 GiB |

The final elapsed window includes editing during the interruption and is not pure
algorithm throughput. It ends before the original conservative 1,800-second
deadline. The six original model loads plus one final-average reload for recovery
are explicit; all owned workers have exited and no service/model remains resident.
Do not infer the lost original peak from the separately measured processes.

Original gameplay source: `f2934d3a9fe5247ac126013c86b698f6b040176b`.
Completion source: `9501d42738513a2dab71e62720680ac58783489f`; original gameplay
files verified unchanged. Extraction source: `ee79b39` (full identity in the
extraction summary). The fixed plan and opponent definitions did not change after
the pilot. [Final audit](hu20-cfr-average-artifacts/final-audit.json) verifies
all records; [environment](hu20-cfr-average-artifacts/validation-environment.json)
records Python 3.11.15, NumPy 1.26.4, SciPy 1.17.1 and native engine
`5db20e3d5d6862b32a7402035c1340b622d3b005`. **28 focused tests pass**, including
the production guard, independent accumulator factors, information isolation,
native replay/tails/paired arithmetic, byte reproducibility, corruption rejection
and interruption/expired-deadline checks. CI status is on the draft PR.

The [evidence inventory](hu20-cfr-average-artifacts/evidence-manifest.json) pins
raw records, derived tables, audited extraction hashes and the retained truncated
stream. Average artifact hashes are `f83d250e27d45d5e12434b90fb270c0217a329bb76e19e7d73bfcefd701c0aaa`,
`8829d26430e6dc0c47d5e550ae1f8fcf6e99a28619c53b54097f51a9478f445d`,
and `c5fa910a1211c75996773cbf280650c94f47a13f83c0fd36d64be6647d100285`
for seeds 1–3. Current/checkpoint hashes and exact sizes are in the input plan.

### Unresolved questions

- Does the LBR point improvement persist on a larger independently frozen paired
  sample? This run is too uncertain to establish it.
- Why do native-pressure directions differ by lineage and position? Inspect
  matched decisions before attributing them to extraction or abstraction.
- Does another opponent drive materially more visits into zero-mass keys? These
  panels cannot establish good coverage against every off-tree history.
- Are stored averaging/sampling behavior and abstraction adequate for the intended
  game? This audit validates retained totals and extraction, not a new convergence
  theorem or a reason to weaken the current-export guard.

## Reproduce without M4 computation

Use the audited [source environment](../../readme.md#quick-start), measured on M1
macOS arm64/Python 3.11.15. Other platforms are not claimed tested by this report.
The [input plan](../../configs/diagnostics/b500-cfr-average-inputs.json) pins the
three full checkpoint/current pairs by size, SHA-256, lineage and iteration.
Coordinate retrieval of these research artifacts; this report does not claim
they are all public release assets. Keep compressed inputs unchanged and put them
under the plan's relative filenames in a private input directory. Inference
exports alone cannot supply the retained averages or resume training.

```sh
python -m pytest -q tests/diagnostics/test_cfr_average.py tests/diagnostics/test_cfr_average_evaluation.py tests/test_blueprint_hu20.py tests/test_blueprint_native_reopening.py
python -m scripts.extract_hu20_cfr_average --plan configs/diagnostics/b500-cfr-average-inputs.json --inputs results/cfr-average-inputs --out results/cfr-average-extraction-new
python -m scripts.evaluate_hu20_cfr_average --plan configs/diagnostics/b500-cfr-average-comparison.json --inputs results/cfr-average-inputs --averages results/cfr-average-extraction-new --out results/cfr-average-comparison-new
python -m scripts.audit_hu20_cfr_average --directory results/cfr-average-comparison-new --out results/cfr-average-audit-new.json
python -m scripts.report_hu20_cfr_average --directory results/cfr-average-comparison-new --out results/cfr-average-tables-new
```

Fresh output names are required; previous outputs are not silently overwritten.
The first command uses generated small fixtures and a synthetic traversal, not
new campaign training. Extraction checks SHA before parsing, audits all nodes and
returns a complete/incomplete status before any evaluation. The final plan uses
the resulting pinned diagnostic hashes. Frozen source/engine/dependency identity
is required for byte-for-byte reproduction; new model outputs must not silently
replace plan hashes. All three real pairs are required; do not cherry-pick a seed.

The committed hand records contain generated offline simulator observations,
deal seeds, actual actions and policy distributions for native replay; they are
not browser histories or private human sessions. No model binaries, access tokens
or private journals are committed. Doctor Research permitted a bounded read-only
transfer of 389,861,569 checkpoint bytes; all hashes were verified on M1 before
parsing, the shared M4 coordination note records release, and originals/jobs were
not changed. All computation is on M1. No paid services, model promotion or changes
to #136 follow from these results.
