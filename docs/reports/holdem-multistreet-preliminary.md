# Multi-street campaign: preliminary report

**Snapshot: September 18, 2026, 14:37 UTC / 16:37 Madrid.**

Reference generation is running normally. Calibration and implementation checks
are complete; the nine campaign model fits have not started. There is no model
winner, playing-strength result or production change to report yet.

This is a dated progress report. It supplements the [frozen campaign
protocol](../holdem-multistreet-campaign.md) and [calibration and resource
record](holdem-multistreet-calibration.md); it does not replace their rules.

## Why we are running this experiment

Earlier full-game training produced weak policies. The [frozen-fit and policy
comparison](holdem-policy-comparison.md) found no consistent playing-strength
improvement from the tested clipping/fitting changes, despite substantial
changes in action probabilities. That narrowed the practical next questions
to target quality, coverage and representation; it did not rule out every
possible optimizer improvement.

The [river architecture comparison](holdem-representation.md) showed that
larger models could fit known boards better without making better decisions
on unseen boards. The subsequent [card-diversity experiment](holdem-card-diversity.md)
gave promising evidence for explicit visible-card features: the 48-board
feature model's mean test decision cost was 0.2168 BB versus 0.3967 BB for
the scaled 48-board control. Its overall comparison remained unconfirmed
because one validation seed failed. These are restricted benchmark costs,
not poker win rates.

The present question is whether the representation benefit extends across
flop, turn and river when entire board families are held out and fitting
duration is selected separately from final evaluation.

## What is being trained

This is supervised regret fitting on a restricted six-player Hold'em
reference problem using 2 BB starting stacks, specified private-card ranges
and fixed continuation policies. It is not a new full-game Deep CFR self-play
campaign. Each reference integrates compatible opponent holdings and future
cards; the model receives only the permitted observation. Hidden worlds are
used to calculate targets, not exposed as playing inputs.

The two reference continuation profiles are `uniform` and `increasing`,
combined with the declared 1:2 schedule. Conclusions are conditional on this
restricted setup, not arbitrary opponents, stack depths or poker populations.

| Split | Board families | Contexts | Purpose |
| --- | ---: | ---: | --- |
| Train | 24 | 576 | Fit model weights |
| Tuning | 8 | 192 | Choose fitting duration |
| Validation | 8 | 192 | Qualify the selected recipes |
| Sealed test | 8 | 192 | Evaluate baseline and qualifying candidates once |

Each family supplies four holdings, three streets and two situations
(open/facing), giving 1,152 contexts. Whole-family separation prevents variants
of the same board family from appearing in both training and held-out splits.

Three representations are compared: the scaled baseline, a separate learned
card branch, and explicit deterministic visible-card features. Seeds
941/947/953 give nine fits. Each uses batch size 32, learning rate 0.001 and
gradient clipping at 1.0, with checkpoints at 1,024, 2,048 and 4,096 steps.

One duration per representation is selected using mean tuning decision cost
across seeds, with the earlier duration winning ties. The 4,096-step checkpoint
is not automatically preferred. Validation qualification requires, for every
seed, at least 0.02 BB and 10% cost improvement, a positive gain after subtracting
two paired reference standard errors, and no more than 0.02 increase in
relative RMSE. This reference-noise screen is conditional on the evaluated
contexts; it is not a population-wide confidence guarantee.

## Completed finding: reference precision differs sharply by street

Calibration used 72 training-only contexts and nested streams of
8/16/32/64/128 worlds. Alternative actions share the same worlds, so precision
is measured on action differences rather than separate action values.

For each context, calibration takes the largest action-pair standard error;
the table reports the 90th percentile across the twelve contexts in each
street/situation group. The target was one standard error at most 0.10 BB.

| Street / situation | Frozen worlds | Paired SE summary (BB) | Status |
| --- | ---: | ---: | --- |
| Flop, open | 128 | 0.3074 | Unresolved at cap |
| Flop, facing | 128 | 0.3524 | Unresolved at cap |
| Turn, open | 128 | 0.2755 | Unresolved at cap |
| Turn, facing | 128 | 0.3333 | Unresolved at cap |
| River, open | 32 | 0.0703 | Resolved |
| River, facing | 16 | 0.0776 | Resolved |

Rivers satisfy the declared calibration summary. Flop and turn references
retain appreciable uncertainty. The protocol explicitly permits continuing at
128 worlds with an unresolved flag; no precision gate was retrospectively
relaxed. These labels must be described as Monte Carlo references with measured
uncertainty, not uniformly precise ground truth.

Under a simple independent-sampling square-root extrapolation, reaching
0.10 BB from the observed flop/turn summaries would take roughly 1,000–1,600
worlds, about 8–12 times the current count. This is a planning approximation,
not a measured requirement or authorization to scale the run.

The variation combines opponent holdings and future runouts. We have not
decomposed those sources or shown that either caused the earlier self-play
failure. Nor does 0.30 BB context-level uncertainty imply equally large
uncertainty in the aggregate paired model comparison.

## Completed finding: less reference bookkeeping preserves the answers

The first collector retained statistics for later hero decisions even though
this benchmark only uses root-decision values. Removing that unused
accumulation preserved the traversal and returned root statistics.

- Original calibration: 38.69 minutes, 36.53 GB memory high-water mark.
- Optimized calibration: 29.57 minutes, 7.46 GB sampled peak usage.
- The complete calibration precision traces and decisions match exactly.
- Twelve fresh world-prefix comparisons match saved action values and derived
  targets exactly; local equivalence tests include open and facing flop/turn.
- All 58 focused host tests passed, and CI passed for the optimized source.

The memory figures use different peak observations, so they are not a precise
controlled memory-reduction estimate. They nevertheless establish sufficient
headroom for this rental. Timing differences are observed wall times, not a
hardware-independent speedup claim. This changes the reference benchmark's
implementation, not the production Deep CFR sampler.

## Live progress and operations

The full pipeline started at 13:46:27 UTC. At the 14:37 UTC check:

- 199 of 1,152 production reference contexts were complete (17.3%).
- Approximately 51 minutes had elapsed since launch.
- All 32 compute workers were active, using about 98–99% of a CPU each.
- Current container memory usage was 7.65 GB out of 64 GB.
- The inspected driver and worker logs contained no errors.
- None of the nine campaign fits had started; collection precedes fitting.

Context costs differ greatly, particularly for unopened flops. Completed-context
percentage is not elapsed-work percentage. The status file's elapsed value is
recorded at the last completion event, not a live wall clock.

The provisional completion window remains 20:00–22:00 Madrid, including
fitting and verification. This is calibration-based, not a precise estimate
from the early completed-context count. A timing-only probe using repeated
training calibration data supported a 30-minute fitting reserve; its weights
and quality metrics are not campaign results.

The first automatic runtime admission failed. Its original six-hour cumulative
reference allowance and 90-minute fit reserve did not fit the padded estimate.
After the timing probe, a documented allocation of seven cumulative reference
hours (including both calibrations) plus a separate 30-minute fit reserve
passed within the same absolute deadlines and $10 campaign ceiling. The failed
admission remains in the record; scientific settings were unchanged.

The host has 32 vCPU and 64 GB RAM in the advertised 3 GHz pool, reporting an
AMD EPYC 9654. The quoted 5 GHz option was unavailable. Compute and storage are
approximately $0.963/hour. From provisioning at about 11:56 UTC to this snapshot,
estimated spending is $2.59, leaving approximately $22.79 of the owner-reported
$25.38 funded balance. These are rate-based estimates, not a refreshed invoice.

Scientific work stops by 22:10 Madrid; the independent provider watchdog
terminates the pod at 22:55 Madrid if it remains running. The local retrieval
watcher was found stopped at 16:17 Madrid and restored under macOS launchd;
it is active at this snapshot. The remote run and watchdog continued throughout.

## Additional audit after collection

Independent calibration and production references overlap on training
contexts. A supplementary audit will reuse those existing samples, without
changing qualification rules or ordering another reference campaign:

1. Compare action-value and regret vectors, including RMSE and correlation.
2. Compare action-gap signs and top-action agreement, handling ties explicitly.
3. Compare regret-matched policies through total-variation distance.
4. Evaluate each batch's policy against the other's values, alongside
   within-batch costs and paired policy-value differences.

Results will be separated by street and open/facing situation. Cross-batch
cost is not exploitability: maximizing noisy values can inflate it, and regret
matching need not choose the single highest-value action even with exact
targets. Policy disagreement between nearly equal actions has a different
meaning from disagreement involving substantial value differences.

If references are strategically unstable, stratified world sampling becomes a
candidate follow-up. Any change must preserve the declared distribution,
including probability mass represented by duplicate range entries, and use
uncertainty calculations appropriate to the new sampling design.

## Artifacts and remaining work

Code, plans and reports are tracked in Git. Raw experiments live in the ignored
local `results/` directory, which currently totals about 32 GB. Previous rental
archives were retrieved and checked before their shutdowns; large models and
raw outcomes may remain compressed. A fresh Git clone does not contain them,
and a durable second backup remains desirable.

The current campaign's full output has not yet been retrieved. Its driver is
configured to verify model results and reference provenance, package the
artifacts, and publish a checksum. The local watcher then downloads and verifies
the archive and file inventory before requesting compute termination. Volume
deletion follows verified retrieval. At this snapshot the Mac has about 44 GiB
available; the retriever checks space against the actual archive size.

Next: finish references, fit all nine models, apply frozen tuning/validation/test
rules, verify and retrieve artifacts, remove billable resources, perform the
supplementary stability audit, and publish the final interpretation through
PR #85. No representation is promoted and no playing-strength claim follows
from this preliminary report.
