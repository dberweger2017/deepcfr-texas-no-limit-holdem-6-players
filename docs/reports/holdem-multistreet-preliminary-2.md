# Multi-street campaign: second preliminary report

**Snapshot: September 18, 2026, 17:42:50 UTC / 19:42:50 Madrid.**

Reference generation has reached **879 of 1,152 contexts (76.3%)** after
3 hours 56 minutes. All 32 compute workers are active, memory usage remains
around 8 GB, and the inspected scientific logs contain no errors. The nine
model fits have not started, so there are no architecture scores or new
playing-strength findings yet.

This report follows the [first preliminary report](holdem-multistreet-preliminary.md).
It records progress under the same [frozen protocol](../holdem-multistreet-campaign.md),
not a new experiment or a change to model-selection rules.

## Purpose and experiment design

The campaign tests whether explicit visible-card structure or a separate
learned card branch improves decisions on unseen board families across flop,
turn and river. Earlier experiments showed that better training fit did not
reliably translate into better unseen-board decisions. Visible-card features
were promising, but their previous comparison did not pass every required
validation seed.

This is supervised regret fitting on a restricted six-player Hold'em problem
with 2 BB starting stacks, specified private-card ranges and fixed continuation
policies. It is not full-game Deep CFR self-play. Simulated hidden worlds supply
reference targets; model inputs remain restricted to the permitted observation.

The 48 board families supply 1,152 contexts: four holdings, three streets and
two situations per family. The splits remain 576 training contexts and
192 each for tuning, validation and sealed test. Models receive no validation
or test targets during fitting.

Three representations, each with seeds 941/947/953, give nine fits:

- Scaled baseline.
- Separate learned card branch.
- Explicit deterministic visible-card features.

The fitting recipe remains batch size 32, learning rate 0.001 and gradient
clipping at 1.0. Checkpoints at 1,024/2,048/4,096 steps are compared on tuning
data; one duration per representation is selected across seeds. Validation
then applies the declared decision-cost, relative-error and paired-reference
uncertainty checks. The baseline and all qualifying candidates proceed to the
sealed test once. More fitting is not automatically treated as better.

## Progress since the first report

| Check time (Madrid) | Completed contexts | Share of contexts |
| --- | ---: | ---: |
| 16:37 | 199 | 17.3% |
| 17:21 | 385 | 33.4% |
| 18:14 | 563 | 48.9% |
| 18:54 | 705 | 61.2% |
| 19:42:50 | 879 | 76.3% |

Another 680 contexts have completed since the first report; 273 remain.
The full pipeline started at 13:46:27 UTC. The status file records elapsed
time at the most recent completion, so current elapsed time is taken from
the driver rather than treating that field as a live clock.

Open flops are substantially more expensive than short facing/river trees.
Completed-context percentage is therefore not elapsed-work percentage. Workers
can stay busy while the completion counter changes slowly, then finish several
lighter contexts quickly. The observed progress is consistent with sustained
computation; no restart of the scientific campaign has occurred.

At this snapshot, the 32 compute workers report roughly 97.7–99.8% CPU usage
each. Current container memory is 8.09 GB of the 64 GB allocation, compared
with 7.65 GB at the first report. The kernel's 36.53 GB historical high-water
mark includes the original calibration; it is not current production memory.

## Scientific findings remain the calibration findings

The completed calibration established the following frozen budgets and
uncertainty summaries:

| Street / situation | Worlds | Paired SE summary (BB) | Precision status |
| --- | ---: | ---: | --- |
| Flop, open | 128 | 0.3074 | Unresolved at cap |
| Flop, facing | 128 | 0.3524 | Unresolved at cap |
| Turn, open | 128 | 0.2755 | Unresolved at cap |
| Turn, facing | 128 | 0.3333 | Unresolved at cap |
| River, open | 32 | 0.0703 | Resolved |
| River, facing | 16 | 0.0776 | Resolved |

Each summary is the 90th percentile across twelve training contexts of the
largest action-pair standard error in that context. It is not a per-context
confidence guarantee. River summaries meet the 0.10 BB planning target; flop
and turn do not. Continuing with explicit unresolved flags was part of the
original protocol.

The reference-collector optimization removed unused later-decision statistics.
The repeated calibration reproduced its predecessor's complete precision
traces exactly, and twelve detailed reference comparisons matched saved values.
Observed calibration time fell from 38.69 to 29.57 minutes, with much lower
observed memory usage. These implementation checks do not establish better
poker play.

The new evidence since the first report is operational: the optimized collector
has sustained nearly four hours of production work at this concurrency without
an observed memory problem or a logged scientific failure. It does not yet tell
us which representation generalizes best or whether production targets are
strategically stable.

## ETA, budget and recovery

Recent progress supports a provisional **75–100 minutes of reference work
remaining**, followed by approximately 20–30 minutes of model fitting and then
verification/transfer. Completion is provisionally expected around
**21:30–22:00 Madrid**. This is a working window, not a guarantee: the remaining
context mix and final slow workers can shift it, and fitting time comes from
the earlier timing probe rather than completed campaign fits.

The rental remains the same 32 vCPU / 64 GB EPYC 9654 host, at about
$0.963/hour including storage. Provisioning began around 11:56 UTC. Estimated
spending through this snapshot is **$5.56**, including setup and both
calibrations, leaving approximately **$19.82** of the owner-reported $25.38
funded balance. These figures are rate-based estimates, not a fresh invoice.
The campaign ceiling remains $10.

Scientific work has an absolute cutoff at 22:10 Madrid. The separate provider
watchdog terminates the pod at 22:55 Madrid if it is still running, retaining
the network volume for recovery. These deadlines have not been extended.

The local archive watcher is alive. A transient SSH polling timeout was
recorded earlier; polling now tolerates that timeout, and the macOS job was
configured not to relaunch after completed retrieval. This local operational
change did not alter the remote source, targets or fitting recipe. The watcher
is waiting for the archive, not downloading it yet.

The Mac has approximately 44 GiB available. Retrieval checks available space
against the eventual archive size, verifies its checksum and file inventory,
then requests compute termination. Disposable volume deletion follows verified
retrieval. The current campaign's complete raw outputs are still on Runpod.
Previous campaigns' verified archives remain in the local ignored `results/`
directory; a Git clone alone does not include those raw artifacts.

## What remains and how we will interpret it

1. Finish all declared reference contexts and all nine fits.
2. Apply frozen tuning, validation and sealed-test rules; retain unsuccessful
   candidates and all failures in the record.
3. Verify results and provenance, retrieve artifacts and remove billable resources.
4. Run the supplementary independent-reference stability audit on overlapping
   calibration/production training contexts using existing samples.
5. Publish the final report and review/complete PR #85.

The stability audit compares value/regret vectors, action preferences,
regret-matched policy distances and cross-batch decision costs by street and
open/facing situation. It will also retain within-batch costs and paired
policy-value differences: noisy maxima can inflate apparent costs, and regret
matching is not necessarily greedy even with exact values.

A model win with stable references would strengthen the representation case
within this benchmark. A model win with unstable references could partly
reflect robustness to label noise. No qualifying candidate with unstable
references would leave open whether the estimator masked useful differences.
These interpretations guide the next experiment; they are not automatic
root-cause diagnoses or model-promotion rules.

**Current decision:** finish the frozen campaign. No restart, new rental,
larger world budget or production promotion is warranted by the progress
measurements so far. Model-quality conclusions remain pending.
