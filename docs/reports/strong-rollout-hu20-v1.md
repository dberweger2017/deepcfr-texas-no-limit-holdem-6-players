# Range-aware HU20 heuristic: frozen v1 and decision diagnostics

## Result

I completed the bounded M1 study: **12,288 fresh, position-balanced saved-policy
hands**, **192 outcome-blind decision cases**, and **54,016 sampled continuation
branches**. Every actual hand passed native replay; a separate model-free audit
rechecked every hand, sample coordinate and held-out gap, plus all 844 first-world
alternative traces. The draft implementation is [PR #140](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/140).

**`strong-rollout-hu20-v1` did not meet its predeclared calibration quality goal.**
It is an inspectable heuristic baseline, with strength unvalidated. I made no
strategic revision after development and froze this result before loading the
saved-policy comparisons. There is no training, model promotion, action
translation or change to the active #136 campaign.

Positive numbers below are the **saved model's** profit against frozen v1.
Intervals use shared deal blocks, condition on these three original lineages,
and are exploratory unadjusted 95% Student-t intervals.

| Rival sizing | B100M BB/100 [95%] | B500M BB/100 [95%] | Paired B500M−B100M [95%] |
| --- | --- | --- | --- |
| Restricted | +22.92 [4.85, 40.98] | +18.75 [−1.20, 38.70] | −4.17 [−20.57, 12.24] |
| Native | +11.76 [−6.08, 29.60] | +5.25 [−14.84, 25.35] | −6.51 [−22.90, 9.88] |

**Neither sizing panel resolves improvement or regression from 100M to 500M.**
The point declines are not evidence of a demonstrated regression. No per-lineage
checkpoint-change interval excludes zero. This is one additional opponent, not
a replacement for the scripted/LBR panel or an independent confirmation of a
post-Luna mechanism hypothesis.

## Frozen behavior and calibration

The [protocol](../strong-rollout-hu20-v1-protocol.md) and
[configuration](../../configs/diagnostics/strong-rollout-hu20-v1.json) declare
preflop ranges, public-action likelihood coefficients, seeded mixing, menu rules,
sample sizes and acceptance criteria. The opponent receives immutable acting-seat
observations only. Its 128-particle assumed range uses public betting history and
card removal. Postflop exact ranks over 64 common worlds estimate equity; raise
scores use estimated folds and equity against the estimated continuing range.
Pot odds, effective stacks/SPR, draws, texture and nut-suit blockers are logged.

These are approximate checkdown scores and heuristic range likelihoods, not a
multistreet solve or a calibrated posterior for the saved blueprint. The name
does not establish strong play. The reusable diagnostic scorer separately runs
full observation-only continuations to settlement.

Development used 768 hands; fresh confirmation used 3,072 hands (128/control/mode,
64 paired blocks). Both sizes were fixed before results. The six random/call/
minraise confirmation checks use Bonferroni family-95% intervals. Only check/call
clears that positive lower-bound gate in both modes. Random and minraise remain
uncertain. The secondary nonnegative-point threshold passes at 6/9 other controls
in restricted mode and 7/9 in native mode; the combined quality goal remains
**failed**. Restricted jam/raise controls produce the same trajectories here;
they are not independent evidence. Aggressive controls include negative points.

All 24 confirmation panels, positions and full-stack counts are in the
[derived tables](strong-rollout-hu20-v1-artifacts/derived/tables.md#calibration-confirmation).
The [panel CSV](strong-rollout-hu20-v1-artifacts/derived/panels.csv) includes exact
intervals and counters. Restricted fixed-style controls project their preferred
raise to a concrete menu action, as declared before calibration; native style
controls keep exact legal sizes. The saved target's actions and lookup keys are
never translated. Random samples uniformly from each mode's small menu.

## Comparison schedule, positions and tails

The [final plan](../../configs/diagnostics/strong-rollout-comparison.json) fixes
512 paired deal blocks/model/mode, two seat rotations per block and six exports:
three original training seeds × B100M/B500M. Each hand resets to 2,000 chips/seat,
with 100 chips/BB, no rake/ante. Deal root `202610040301` is fresh and disjoint
from calibration and pilot roots; deal and action/opponent streams are separate.
All checkpoints share deals and stream identities. Do not count paired rotations
or repeated checkpoint deals as independent samples.

Restricted mode uses the uncapped min/pot/conditional-jam game. In native mode
the heuristic can use min, 2.5BB openings, 3× reraises, one-third/two-thirds and
conditional jam, with exact native bounds. The target uses its unchanged saved
restricted menu in **both** modes, including the existing missing-key fallback.
Native mode is a distinct opponent-sizing experiment, not a solved off-tree game.

| Mode / checkpoint | BB / button BB/100 | Large actions / opportunities | Rival folds / continues | Full-stack wins / losses | Fallback / target decisions |
| --- | --- | --- | --- | --- | --- |
| Restricted B100M | +15.23 / +30.60 | 171 / 748 | 119 / 52 | 43 / 60 | 4 / 4,150 |
| Restricted B500M | +24.93 / +12.57 | 150 / 807 | 100 / 50 | 39 / 65 | 0 / 4,313 |
| Native B100M | +21.86 / +1.66 | 155 / 683 | 104 / 51 | 36 / 61 | 138 / 4,508 |
| Native B500M | +20.05 / −9.54 | 126 / 700 | 84 / 42 | 42 / 72 | 143 / 4,567 |

Each row has 3,072 hands; counts sum the three lineages. A large action demands
at least 800 additional rival chips (8BB). Opportunity denominators are target
decisions with such a legal menu action, not hands. Responses partition large
actions; all have a recorded response here. Full-stack counts require exactly
±2,000 net chips. Fallback rates are about 3.06%/3.13% in native B100M/B500M,
versus 0.096%/0% restricted. Street and large-action lookup counters are preserved.

The [per-lineage table](strong-rollout-hu20-v1-artifacts/derived/tables.md#saved-checkpoint-panels)
and CSV report all positions and intervals: for example, native B500M button
points are −5.84, −30.11, +7.31 BB/100 across seeds 1–3. These differences deserve
inspection, but their uncertainties do not establish a seed or position defect.
Large-action frequency falls at B500M, while full-stack losses rise in both modes;
these correlated descriptive counts are not causal estimates of an individual bet.

[Whole-hand partitions](strong-rollout-hu20-v1-artifacts/derived/whole-hand-partitions.csv)
separate hands by the response to their first large raise, including no-large-raise
hands, and reconcile total chips exactly. Their returns are **whole-hand subgroup
returns, not bet EV**. Earlier wider opponents remain in the
[stackoff report](hu20-stackoff-v1-m1.md) and #136 campaign; this study does
not rerun or pool those different schedules.

## Decision gaps and representative traces

The frozen coordinate hash (`202610040302`) retains two target decisions per
street/position/model/mode, before looking at payoffs. All 96 strata are filled:
192 cases, 16/model/mode. For each case, 32 worlds select an alternative; 32
independent worlds estimate its paired advantage over the saved-policy root
mixture. Alternatives share compatible sampled worlds and reset continuation
streams. Native extra alternatives receive zero saved-root mixture weight.

| Mode / checkpoint | Cases | Descriptive mean gap (BB/decision) | Conditional intervals wholly positive / negative | Minimum range ESS |
| --- | --- | --- | --- | --- |
| Restricted B100M | 48 | +1.293 | 12 / 0 | 12.76 |
| Restricted B500M | 48 | +1.422 | 14 / 2 | 16.73 |
| Native B100M | 48 | +1.638 | 16 / 1 | 10.11 |
| Native B500M | 48 | +1.307 | 11 / 0 | 3.84 |

These means are **not occupancy-weighted policy losses or checkpoint rankings**.
States differ as policies act differently. Case intervals are unadjusted across
192 cases and include only sampled-world uncertainty conditional on the fixed
assumed range/continuation model; heuristic bias is omitted. Negative held-out
gaps remain recorded. A positive interval, weak showdown hand or lost stack is
not a proven error. No actual hidden deal is reused to represent expected value.

The [decision CSV](strong-rollout-hu20-v1-artifacts/derived/decisions.csv) contains
every selected observation, exact root mixture, candidate, independent-half means,
gap/interval, range ESS and lookup exposure. The
[representative table](strong-rollout-hu20-v1-artifacts/derived/tables.md#representative-decisions)
uses minimum frozen priority per street/mode in first-lineage B100M, not the
largest losses/gaps. Raw `.decisions.jsonl.gz` files retain all range holdings,
world/continuation seeds and candidate returns, with each first world's exact
own-observation/action trace. `.hands.jsonl.gz` retains factual public bounds,
policy probabilities and the rival's concise features/scores at every decision.
These are generated offline simulator records, **not browser-visible histories**.

Concrete next questions, without a v2 implementation here:

- Would broader/calibrated assumed ranges change the cases with ESS near 4 out of
  128 particles? Low ESS is a diagnostic lead, not proof that the range is wrong.
- Which gaps persist under independent range/continuation models and more worlds,
  particularly the representative river fold/bet decisions?
- Do native fallback states and button exposure explain any tails? Compare trained
  and fallback contexts before attributing them to a bucket or training mechanism.
- Can v2 meet the original control strength goal while retaining honest uncertainty?
  v1 is a frozen reference; no target-conditioned tuning was performed.

## Provenance, resources and verification

| Stage (M1) | Work | Wall seconds | Process peak |
| --- | --- | --- | --- |
| Calibration development | 768 hands | 3.51 | 115.81 MiB |
| Calibration confirmation | 3,072 hands | 15.34 | 177.72 MiB |
| Real timing pilot | 16 hands / 352 continuations, one load | 9.465 | 1.754 GiB |
| Final six-checkpoint run | 12,288 hands / 54,016 continuations | 682.42 | 3.126 GiB |
| Separate model-free audit | 12,288 replays / 192 cases / 844 branches | 6.23 | 549.11 MiB |

The 30-minute/6GiB worker guard was never reached. Six sequential loads, one per
export, share each loaded source across modes; no services/models remain running.
Measurements are whole-process peaks on macOS, not per-model isolated allocation.
M4 use was limited to an authorized read-only transfer of closed exports and
metadata (209,793,058 compressed export bytes), hash-verified locally; the window
was explicitly released. Doctor Research's jobs, checkout and originals are intact.

The final gameplay source is **`c2dc6fbe38d5e36b5259ea168e1573a99ce37920`**.
The opponent freeze source is `ff8c7d60eb16c64f6b3a4bcb02ea63769df5ea5c`,
canonical freeze digest `97f9e5dd1eca0b20ad4c380524570d788ac844237acc0261e4fd0af684d00fd6`.
Protocol/config preceded outcomes; the final budget was committed before the run.
Calibration source `9e4f655` predates the freeze. The first timing attempt failed
before any model load/hand due to an input-root name collision; its incomplete
output remains, followed by a separately retained successful attempt after a
runner regression fix. Later audit/formatter additions do not change frozen play.

Input byte sizes, all six SHA-256s, checkpoint lineage hashes, game/schema/current
extraction and B500M iteration identities are in the plan and authoritative
[transfer manifest](strong-rollout-hu20-v1-artifacts/b500-transfer-manifest.json).
B500M campaign scientific source is `17b4c9a08ed0765d0fb8f05240c0409b21e43977`.
Inference exports cannot resume training. No model binaries are committed here.

[Validation environment](strong-rollout-hu20-v1-artifacts/validation-environment.json):
M1 arm64, macOS 27.2, Python 3.11.15, NumPy 1.26.4, SciPy 1.17.1,
native engine `5db20e3d5d6862b32a7402035c1340b622d3b005`. All **17 focused tests**
pass: information isolation under changed actual hidden holdings/future decks,
determinism, exact legal short all-ins, native settlement/replay, chip-score/tail
arithmetic, separate selection/evaluation, outcome-blind pairing, frozen config,
fixture runner and audit tamper rejection. Fixture checks use generated small
exports; measured comparison uses the six verified real exports. Repository CI
runs the broader gates separately; its status is on the draft PR.

[Final audit](strong-rollout-hu20-v1-artifacts/final-audit.json) and individual
subdirectory manifests preserve replay/count/hash evidence. The
[evidence inventory](strong-rollout-hu20-v1-artifacts/evidence-manifest.json) pins
all raw and derived files. No human sessions, access tokens or private journals
are included. This is integration/diagnostic evidence, not a strength campaign.

## Reproduce

From the source checkout, install the audited dependencies using the documented
[source environment](../../readme.md#quick-start). Python 3.11 is the measured
platform; other platforms are not claimed tested by this report. Most tests and
all table/replay audits need **no** policy exports:

```sh
python -m pytest -q tests/diagnostics/test_strong_rollout.py tests/diagnostics/test_decision_counterfactual.py tests/diagnostics/test_strong_comparison.py
python -m scripts.audit_strong_hu20 --directory docs/reports/strong-rollout-hu20-v1-artifacts/comparison --out results/strong-audit-new.json
python -m scripts.report_strong_hu20 --evidence docs/reports/strong-rollout-hu20-v1-artifacts --out results/strong-tables-new
python -m scripts.calibrate_strong_hu20 --config configs/diagnostics/strong-rollout-hu20-v1.json --stage development --out results/strong-dev-new
python -m scripts.calibrate_strong_hu20 --config configs/diagnostics/strong-rollout-hu20-v1.json --stage confirmation --out results/strong-confirm-new
```

Use fresh output paths; scripts reject replacement of previous evidence.
Reproducing model comparisons requires **all six** exact inference exports in
one input directory named as the plan's `path` fields. First-lineage B100M is the
v0.4 release asset; the remaining research exports require coordinated retrieval
from their manifests. This report does not claim all six are public downloads.
Keep compressed bytes unchanged. The runner checks length/hash **before** loading,
then the adapter verifies game/schema/lineage/current-policy identity.

```sh
python -m scripts.compare_strong_hu20 --plan configs/diagnostics/strong-rollout-pilot.json --inputs results/strong-inputs --out results/strong-pilot-new
python -m scripts.compare_strong_hu20 --plan configs/diagnostics/strong-rollout-comparison.json --inputs results/strong-inputs --out results/strong-comparison-new
python -m scripts.audit_strong_hu20 --directory results/strong-comparison-new --out results/strong-comparison-audit-new.json
```

Do not turn reproduction into uncoordinated M4 work. These commands reproduce the
frozen baseline and estimates; they do not authorize training, v2 tuning or promotion.
