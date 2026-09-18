# Evidence for the next M4 training batch

**Recommendation: keep the current architecture, learning rate and sampler as
the control. If the next batch tests longer training, change only the number of
self-play iterations and retain both seeds. There is no demonstrated optimizer
or representation replacement to adopt from the completed studies.** Longer
training remains plausible, but the observed curves do not establish that it
will rescue this learner.

This analysis uses completed reports through PRs #85 and #86, plus a new read-only
analysis of the M4's saved validation hands. It launches no fitting, simulation,
rental or overnight job. The [new measurements](holdem-m4-duration-analysis.json)
retain every checkpoint, block-level rate and behavioral count. Reproduce with:

```sh
python -m scripts.analyze_m4_duration \
  --root results/m4-fullgame-completed \
  --out results/m4-duration-analysis.json
```

## Distinguish three kinds of more training

1. **More self-play iterations:** more rounds of collection, updated strategies
   and additions to the historical policy average. This is the overnight hypothesis.
2. **More optimizer steps per iteration:** fit each new role network harder on
   its retained replay. Earlier studies directly tested this.
3. **More roots per iteration:** collect more examples under each frozen strategy.
   The M4 increased both roots and fitting, so its comparison with older campaigns
   cannot identify either effect alone.

`fit_role` in `src/holdem/fitting.py` creates a fresh network and fresh Adam state
at each iteration. Extending 256 iterations to 1,024 does **not** give one network
four times as many consecutive optimizer updates; each new role model still gets
256 updates. A global learning-rate decay would therefore change the fitting
of successive fresh models, not continue one optimizer's schedule. Warm-starting
weights or Adam would be a separate algorithm change, not a routine resume.

## What changed as the M4 trained longer?

The new analysis pairs the candidate outcomes on the same 1,024 deal blocks,
averages the six seat rotations within each block, and verifies the exact same
baseline outcomes across checkpoints. Input files match the committed artifact
inventory. Positive differences below mean the later checkpoint earns more.

| Seed | Opponents | 64 → 256 gain, BB/100 | Nominal 95% paired interval | 192 → 256 gain |
| --- | --- | ---: | --- | ---: |
| 2026091802 | Scripted | +62.27 | −83.48 to +208.02 | −13.40 |
| 2026091803 | Scripted | −25.89 | −156.69 to +104.92 | −42.64 |
| 2026091802 | Random | +69.96 | −170.16 to +310.09 | −27.54 |
| 2026091803 | Random | +196.60 | −5.33 to +398.53 | +15.06 |

All twelve computed comparisons (64/128/192 versus 256, two seeds, two pools)
include zero even before multiplicity correction. These are exploratory,
post-hoc intervals conditional on the saved policies, not independent confirmation
or proof of equivalence. The upper bounds leave room for meaningful gains; the
results do not establish monotonic improvement, a learning plateau forever, or
what would happen after iteration 256.

The [fresh final tests](holdem-local-fullgame.md) are stronger evidence about the
fixed final policies: +40.81 and −0.69 BB/100 against random, neither passing the
predeclared adjusted profit/improvement criteria. Seed 2's +283.06 validation
estimate did not carry over to that fresh schedule. This does not isolate
overfitting from evaluation variation.

### Observable behavior and collection coverage

The following counts are reconstructed from public chip payments, including
all-in calls and raises. Each validation snapshot contains 6,144 candidate hands.

| Seed | Preflop all-in share at 64 | At 256 | Hands with a postflop decision at 256 | Collection roots reaching postflop, whole run |
| --- | ---: | ---: | ---: | ---: |
| 2026091802 | 45.49% | 41.13% | 26.30% | 4.35% |
| 2026091803 | 36.31% | 36.98% | 23.29% | 6.43% |

The first seed becomes somewhat less all-in-heavy; the second does not show that
trend. We have not established which particular all-ins are wrong. These are
behavioral warning signs alongside heavy losses, not optimal-action labels.
Collection roots and arena hands have different opponents, exploration and
policy mixtures, so their percentages are not an unbiased coverage-ratio estimate.
They do show that postflop decisions matter in evaluation while collection is
predominantly preflop. Uniformly adding more roots need not fix that imbalance.

## Evidence from previous runs

### Full-game training

The [original longer campaign](holdem-longer-training.md) trained three seeds for
512 iterations with 32 roots per role and 64 fitting steps. Final scripted-pool
profits were approximately −1,190, −1,422 and −968 BB/100. The M4 used 256
iterations, 128 roots and 256 fitting steps: twice the total scheduled roots and
optimizer steps, but fewer strategy updates. Its final scripted values were
−1,056 and −870 BB/100.

These are different training seeds, evaluation deals and hosts. The apparent
numerical improvement is not a controlled treatment effect or proof that the
higher-work recipe is superior. The M4 clearly did not turn the doubled scheduled
work into reliable profitability in this batch.

The [six-job branching campaign](holdem-branching-online.md) also reached 512
iterations. Second-decision expansion improved one seed clearly but failed the
predeclared cross-seed criterion; all final scripted-pool policies lost heavily.
More branching increased later-street records without consistently fixing play.
There is no empirical basis to switch the default sampler for the next run.

### Fitting, clipping and learning rate

The [72 frozen-replay fits](holdem-frozen-fitting.md) compared clipped/unclipped
64- and 256-step fitting with common initializations and minibatch prefixes.
Unclipped/256 reduced mean full-replay loss by 0.225–0.528%, but the subsequent
[played-policy comparison](holdem-policy-comparison.md) found scripted-profit
changes of −94.91, −228.59 and +158.27 BB/100. One seed regressed even under the
familywise interval; the condition for promoting that recipe failed.

All 786,432 M4 optimizer steps clipped at norm 1. That describes the executed
updates; it does not establish that gradients were unusable or that clipping
caused the poker losses. Removing clipping already failed to help consistently.
The sampled diagnostic losses are not directly comparable across iterations
with different replay and fresh initializations.

The reviewed Hold'em experiments do **not** provide a controlled learning-rate
comparison supporting a move away from Adam 0.001. A smaller or larger rate could
be useful, but choosing one now would be a hypothesis, not a fact-backed fix.
Keep it fixed for a duration comparison; do not change rate, clipping and duration
simultaneously and then attribute the outcome to one of them.

### Architecture and generalization, including completed PR #85

The [river representation study](holdem-representation.md) tested original,
scaled, wider, deeper and separate-card models. Larger models fitted training
examples better but none qualified on held-out boards. Mean validation decision
cost was 0.409 BB for original, 0.604 for wide and 0.505 for the card branch.
This finite shallow benchmark does not determine the best full-game architecture.

The [card-diversity study](holdem-card-diversity.md) did not confirm either primary
comparison under its complete validation/test criterion. Explicit card features
looked promising on that river-only test, but this was not a qualified production
replacement.

The completed [multi-street campaign](holdem-multistreet-campaign.md) matters more
than its earlier preliminary snapshots:

- Baseline tuning cost worsened from 0.205 to 0.467 BB as fitting increased from
  1,024 to 4,096 steps. Explicit features worsened from 0.173 to 0.227. The learned
  card branch improved from 0.281 to 0.262. More fitting was architecture-dependent.
- Explicit features regressed on all three validation seeds. The card branch
  improved two but regressed by 0.103 BB on the third; neither qualified.
- Card-branch training relative error was only 2.2–3.5%, while validation error
  remained 62.4–64.7%. It can fit the training mapping; transfer remains poor.
- The independent reference audit found matching best-action sets in 72/72
  overlapping training contexts and mean target-policy TV 0.00205. That weakens
  unstable labels as the explanation **in that audited reference overlap**.
  It does not validate production self-play targets or all held-out references.

These studies favor investigating generalization and coverage over assuming that
more width alone is the answer. They do not identify the main full-game bottleneck
or justify inserting the unqualified card/features models into an overnight run.
No closed candidate test split was opened for this analysis.

The [persistent critic](holdem-persistent-critic.md) also failed its declared
variance-times-cost screen on all three seeds. That specific prototype should
not be adopted as a ready fix for the production target tails.

## Decision for the next batch

| Setting | Recommendation | Evidence strength |
| --- | --- | --- |
| Architecture | Keep width-32 production model as the control | Tested alternatives have no consistent qualification; not proof width 32 is optimal |
| Learning rate | Keep Adam 0.001 | No qualifying controlled LR comparison |
| Gradient clipping | Keep norm 1 | Removing it did not consistently improve played policies |
| Fits/roots per iteration | Keep M4's 256 steps / 128 roots per role | Hold them fixed to isolate longer duration, not a claim they are optimal |
| Replay capacity | Keep 4,096 per role for the duration comparison | No controlled evidence supporting a particular replacement capacity |
| Sampler and exploration | Keep first-decision / 0.5 | Additional branching failed the cross-seed improvement criterion |
| Duration | Reasonable next hypothesis: 256 → 1,024 iterations | Plausible and testable, not established by these curves |
| Evaluation | Scripted-pool primary, random secondary; fresh final deals and both seeds | Directly matches the owner's v0.5 goal and guards against validation optimism |

A clean duration test would continue **both** completed seeds, preserving their
replay, sampler state and policy archives, while explicitly declaring the new
iteration horizon. Compare the saved iteration-256 policies with iteration-1,024
policies on identical **new** final deals; do not reuse the now-inspected final
schedule as unseen evidence. Keep the two training seeds separate, predeclare
absolute profitability and paired improvement endpoints and their interval
adjustment, retain all outcomes, and avoid picking an intermediate lucky model.
Both the old and new checkpoints need evaluation on that new schedule.

The current CLI binds resume to the original plan. A horizon extension needs an
explicit, validated continuation path and provenance record; silently editing
manifests is not acceptable. This report does not implement or authorize that
extension. If continuation is not implemented, fresh 1,024-iteration runs with
retained iteration-256 controls answer a related duration question, but do not
continue these exact trajectories.

A provisional local proposal is two parallel workers, a ten-hour wall-clock cap
and the existing 12 GiB combined memory ceiling. The last 768 iterations alone
project roughly four to five hours from observed 18–21 second training iterations,
before growing archive/checkpoint/evaluation costs; that is an extrapolation,
not an admission benchmark. Calibration must cover later archive sizes and
recovery memory before launch. The 256-iteration checkpoints are already about
0.48 GB each, average exports about 0.165 GB, and the M4 has only about 21 GiB
free at campaign completion. Repeating the 64-iteration save cadence to 1,024
without a storage plan is not appropriate. Define retention/offload, output
limits, evaluation cost and interruption behavior before setting the launch plan.

My decision is **not to stack speculative architecture and optimizer changes onto
the next run**. A bounded duration-only continuation is an honest way to test the
owner's hypothesis using the free M4, while acknowledging that the existing data
do not predict success. The strongest direction for a subsequent recipe change
is to identify specific costly decisions and their training coverage; choosing
the intervention still requires evidence. No extra training is started here.
