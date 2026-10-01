# B500M stored CFR-average extraction

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
