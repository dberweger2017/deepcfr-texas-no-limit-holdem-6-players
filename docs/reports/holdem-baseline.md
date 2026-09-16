# First Hold’em baseline report

## Result

**The implementation runs end to end, but this campaign is incomplete and does not qualify a model.** Ten of twelve scheduled jobs completed with legal arena results. The unequal-stack seed 103 hit its collection limit during iteration 2; seed 107 in that scenario was not attempted after the failure. No failed seed has been dropped or replaced.

All ten completed models had negative observed win rates against the style opponents; eight of their unadjusted 95% intervals are wholly below zero. All ten comparisons against uniform-candidate play are statistically inconclusive. These are weak, barely trained policies, not competitive agents. The settings deliberately provide an implementation check, not enough training to assess convergence or useful model capacity.

## Frozen protocol

- Plan: [`configs/holdem/baseline-check.json`](../../configs/holdem/baseline-check.json).
- Source revision: `863de626f865709c3dc6ccb4ac233fbba3a3b697`; source SHA-256: `e0bcb392ec4e185975f5521eee72b75e56fa380799006f7be45b4f952ef5e496`. The first invocation began with a clean checkout.
- Three seeds (101, 103, 107) per four-, five- and six-player 100 BB scenario, plus a separate six-player 20/40/60/100/150/200 BB scenario.
- Two iterations, one traversal per role, width 16, eight fitting steps per role, capacity 128. Button rotates by iteration. Linear average includes uniform bootstrap and the first fitted collection profile.
- Each iteration is capped at 60 seconds and 50,000 collection nodes; each invocation at 900 seconds. No rental or paid compute.
- Thirty validation deal blocks per job, all seat rotations, five fixed style opponents and a paired uniform-candidate baseline. No model selection or promotion.

## Per-seed results

Rates and paired differences are BB/100. Intervals use independent deal blocks, not individual rotations. Evaluation deals are shared across training seeds, so these rows must not be pooled as independent extra samples. Intervals are unadjusted for multiple comparisons.

| Scenario | Seed | Hands, both arms | Candidate BB/100 | Difference vs uniform | 95% interval for difference | Result |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| four-100bb | 101 | 240 | -1701.67 | -242.92 | [-944.60, 458.77] | inconclusive |
| four-100bb | 103 | 240 | -1122.50 | 336.25 | [-270.14, 942.64] | inconclusive |
| four-100bb | 107 | 240 | -1555.00 | -96.25 | [-751.90, 559.40] | inconclusive |
| five-100bb | 101 | 300 | -1000.33 | 250.00 | [-569.74, 1069.74] | inconclusive |
| five-100bb | 103 | 300 | -1299.00 | -48.67 | [-656.26, 558.92] | inconclusive |
| five-100bb | 107 | 300 | -1282.33 | -32.00 | [-678.41, 614.41] | inconclusive |
| six-100bb | 101 | 360 | -1331.94 | -65.28 | [-663.00, 532.45] | inconclusive |
| six-100bb | 103 | 360 | -1310.28 | -43.61 | [-724.93, 637.70] | inconclusive |
| six-100bb | 107 | 360 | -893.33 | 373.33 | [-189.66, 936.32] | inconclusive |
| six-unequal | 101 | 360 | -775.00 | 127.50 | [-306.99, 561.99] | inconclusive |
| six-unequal | 103 | — | — | — | — | collection_limit |
| six-unequal | 107 | — | — | — | — | unattempted |

Completed **3,060 of 3,780 scheduled arena hands**, with zero invalid actions in the completed jobs. The missing 720 hands invalidate completion of the whole campaign; they are not counted as successful observations.

## Failure and recovery

The initial run stopped at `collection-2-4-0`, traverser 4 of unequal-stack seed 103, after 33,691 nodes in that traversal. Its iteration-1 checkpoint remained intact. The full local test suite was running during this invocation.

After the tests finished, recovery was attempted in a fresh process using the same plan and resource limits. All ten completed job reports, model/archive/replay identities and hand outcomes reproduced exactly. The unfinished iteration again aborted at the same root, this time after 39,150 nodes. Its marker still identifies iteration 1. Different node counts before the same wall-clock deadline are expected; timing is not a deterministic algorithm output.

We did not enlarge the budget, shorten the stack distribution, change seeds or discard the failure. The result exposes a collection-cost problem in this deep unequal-stack setting. Correct rollback makes it recoverable; it does not make the training recipe scalable. Measure the traversal tree and batching costs before a larger campaign.

The two original invocations predate structured experiment failure files; their exceptions and committed artifacts are summarized here. The PR adds `failure.json` for future invocations, including completed, unfinished and unattempted jobs, the error and elapsed time.

## Implementation checks

- **504 repository tests pass**, including 29 added by this PR; GitHub CI also passes on the implementation commit.
- Fresh-process uninterrupted, paused/resumed and reproduced training yield byte-identical final checkpoint and inference artifacts, replay/RNG state, reports and arena outcomes on the declared test fixture.
- Changing evaluation/save frequency preserves training and final hand outcomes. Failure injection verifies rollback and recovery from the last completed iteration.
- Own-reach weighting is checked against a hand calculation, including zero reach, previous-street card visibility and off-menu actions. Session checks cover six/five/four participants and replacement.
- Loader tests reject corrupt hashes, changed provenance, malformed records, mismatched archive generations, counters, roles and non-finite parameters. Inference exports cannot resume training.

These checks establish implementation behavior. They do not close the failed small-game readiness gate or the Hold’em learning exit condition.

## Reproduction and artifacts

The [machine-readable report](holdem-baseline.json) includes the frozen plan, full environment/engine fingerprints, per-seed comparisons and model/checkpoint/outcome hashes. Large artifacts remain outside Git in `results/holdem-baseline-v1` and `results/holdem-baseline-resumed-v1` in the project workspace; they have not been uploaded to durable remote storage.

To reproduce the original code and plan, check out the source revision above and run:

```bash
python -m scripts.train_holdem --plan configs/holdem/baseline-check.json --out results/holdem-baseline-v1
python -m scripts.train_holdem --resume results/holdem-baseline-v1 --out results/holdem-baseline-resumed-v1
```

Exact hashes require the recorded environment. Resource-failure timing/node counts depend on hardware and load; a faster machine may complete within the same limits. Fresh-process tests establish numerical recovery, while these recorded runs establish recovery of the ten completed evaluations and preservation of the interrupted job.

**Next:** integrate small-game snapshot-aware recovery and declare fresh readiness checks. Preserve this incomplete Hold’em report as the starting point for later collection profiling and a meaningful learning campaign. No model was promoted.
