# M4 full-game training with more work per iteration

**Both seeds completed 256 iterations, and all four final checkpoint/export
recovery hashes matched. Neither seed passed the predeclared random-opponent
competence check. Both still lose heavily against scripted opponents; no model
is promoted and the v0.5 playing-strength goal remains unmet.**

The [protocol](../holdem-local-fullgame.md), [plan](../../configs/holdem/local-fullgame.json)
and [machine-readable results](holdem-local-fullgame.json) retain the settings,
validation curves, final endpoints, resource measurements and recovery hashes.
This is a completed experiment with a negative strength result, not a failed
training process or a release qualification.

## Protocol and execution

Six-player, 100 BB, no-rake self-play used the existing width-32 learner,
first-decision sampler, exploration 0.5 and 4,096-record reservoir per role.
Each iteration collected 128 roots per role and fitted fresh networks for 256
Adam steps of batch size 32. Each seed completed 196,608 scheduled roots and
393,216 optimizer steps: twice the earlier 512-iteration campaign's scheduled
work, allocated across half as many policy updates. This changes collection and
fitting together; it does not isolate training duration as a cause.

Training source stayed pinned at `2b6bff36e4c092e22934604e7186a2e8661545ee` on the
owner's M4 (10 CPU cores, 16 GiB RAM). Seed 2026091802 began at 16:54 UTC on
September 18. At the owner's request, seed 2026091803 started at 17:47 UTC while
the first continued unchanged. Separate supervisor handoffs preserved both
worker processes and their original six-hour allowances. The combined worker
RSS ceiling was raised from 9 to 12 GiB by explicit owner instruction; the
7 GiB per-worker, 8 GiB campaign-output and 12 GiB free-disk limits remained.

Save and validation checkpoints were fixed at 64/128/192/256. Both seeds and
all checkpoints are retained. Final testing used only iteration 256, 4,096 fresh
deal blocks, all six seat rotations, random opponents and a paired uniform-action
candidate control. Validation used 1,024 blocks per checkpoint against random
and the scripted style pool. No checkpoint was selected using its poker result.

## Fresh final test against random opponents

All values are BB per 100 hands. Adjusted intervals use the declared Bonferroni
family of four endpoints: two seeds times absolute profit and paired improvement.
The intervals below jointly target 95% coverage for that family.

| Seed | Profit | Adjusted interval | Improvement over control | Adjusted interval |
| --- | ---: | --- | ---: | --- |
| 2026091802 | +40.81 | −100.04 to +181.66 | +142.65 | −9.45 to +294.74 |
| 2026091803 | −0.69 | −139.64 to +138.26 | +101.15 | −58.66 to +260.95 |

Neither absolute-profit interval nor either adjusted improvement interval clears
zero. Seed 1's *unadjusted* improvement interval is positive, but that does not
pass the campaign's declared multiplicity-adjusted check. Both original
`competence_check_passed` values remain false.

Seed 2's final validation profit against random was +283.06 with an ordinary
95% interval of +55.99 to +510.14. Its fresh-test estimate is −0.69. The promising
validation result did not establish reliable profitability on fresh deals.
These observations alone do not diagnose overfitting; sampling variation remains
a possible explanation. Both candidates use the same final evaluation schedule,
so their poker outcomes must not be treated as independent hand samples.

## Validation learning curves

| Seed | Iteration | Profit vs random | Profit vs scripted pool |
| --- | ---: | ---: | ---: |
| 2026091802 | 64 | +45.50 | −1,118.38 |
| 2026091802 | 128 | +125.74 | −1,110.02 |
| 2026091802 | 192 | +143.00 | −1,042.71 |
| 2026091802 | 256 | +115.46 | −1,056.11 |
| 2026091803 | 64 | +86.47 | −843.74 |
| 2026091803 | 128 | +135.25 | −880.23 |
| 2026091803 | 192 | +268.01 | −826.99 |
| 2026091803 | 256 | +283.06 | −869.63 |

The uniform-candidate control loses −1,262.13 BB/100 against the scripted pool.
At iteration 256, the learners improve on that by +206.02 and +392.50 BB/100,
but both remain strongly unprofitable. Their own final scripted-profit 95%
intervals are −1,219.08 to −893.14 and −1,014.62 to −724.64. These are validation
comparisons inspected repeatedly, not independent release confirmations.

The later scripted-opponent curves are roughly flat within their uncertainty.
More iterations could still help, but this batch does not establish insufficient
training time as the explanation. Fresh final testing against scripted opponents
was not part of this frozen run; the newly declared v0.5 gate needs a separate
predeclared confirmation when a candidate is ready.

## Fitting and collection diagnostics

| Seed | Clipped / total optimizer steps | Median loss after / before | Roots with any postflop collection | Collected records: preflop / flop / turn / river |
| --- | ---: | ---: | ---: | --- |
| 2026091802 | 393,216 / 393,216 | 0.997440 | 8,550 / 196,608 (4.35%) | 664,917 / 15,367 / 1,805 / 341 |
| 2026091803 | 393,216 / 393,216 | 0.995678 | 12,647 / 196,608 (6.43%) | 669,991 / 33,181 / 7,886 / 2,590 |

These are collection counts, not final replay composition or independent hands.
Each final reservoir holds 24,576 records across six roles. All 786,432 optimizer
steps clipped at norm 1. The median sampled diagnostic loss reduction within a
fit was about 0.26% and 0.43%; these diagnostics sample retained records and are
not a direct measure of playing strength or the full objective.

Largest inverse own sampling reaches were 176,400 and 320,000; largest absolute
importance-corrected regret updates were 35.81 million and 106.26 million BB.
Those are estimator values, not hand winnings. Neither sparse later-street
collection nor pervasive clipping alone establishes the cause of weak play.

## Resources and verification

| Seed | Worker minutes | Collection / fitting / replay seconds | Sampled peak worker RSS | Final checkpoint bytes |
| --- | ---: | --- | ---: | ---: |
| 2026091802 | 92.51 | 2,416.34 / 2,459.48 / 169.80 | 4.843 GiB | 477,866,086 |
| 2026091803 | 92.56 | 2,412.24 / 2,471.76 / 172.38 | 5.313 GiB | 481,322,839 |

Checkpoint saves took 58–64 seconds each. Fresh-process recovery and final
random-opponent evaluation took 210.61 seconds per seed, 7.02 minutes combined,
inside the shared 30-minute allowance. No resource limit or invalid-play failure
was reported. Compute ran on the owner's M4 with no rental charge.

Both final training checkpoints and average-policy exports reproduced byte for
byte on the same M4. Exact hashes are in the JSON report. This is same-host
recovery evidence, not a fresh full-campaign training rerun or cross-machine
bitwise reproducibility claim. The earlier host benchmark explicitly recorded
cross-machine numerical differences.

The adopted workers' exit codes are unavailable (`null`), because the replacement
supervisor was not their original parent. Completion was established from their
finished result artifacts; the separate recovery/final-test workers both exited
zero. Handoff snapshots, original logs and all outputs remain retained.

## Artifact retrieval and report checks

All **128 campaign files (6,741,329,099 bytes)** were retrieved to the controlling
Mac and matched their M4 SHA-256 hashes. Nine additional supervisor/dashboard
logs and event files (72,577 bytes) also matched. The
[verified inventory](holdem-local-fullgame-artifacts.json) records every file,
size, checksum and local/remote location. Original M4 artifacts remain intact;
no campaign data was deleted to free space.

The local campaign copy is
`/Users/dberweger/Local/deepcfr-local-training/results/m4-fullgame-completed`.
Monitoring files are alongside it in `results/m4-fullgame-support`. Large raw
artifacts remain outside Git; the results report and full inventory are committed.

Both final evaluation reports were rebuilt locally from their saved hand-level
outcomes and test plans. Every reconstructed report field and both adjusted
endpoint intervals matched the saved reports exactly. Each final test contains
49,152 completed candidate/control hands (4,096 blocks × six rotations × two
arms), with zero invalid actions; rotations are not independent statistical
samples. These checks performed no model fitting or additional poker evaluation.

All scheduled validation reports likewise record zero invalid actions. Full CI
passed for the Python implementation at `893f38f`; final report changes are
subject to the normal final-head checks before merge. No independent reviewer
was dispatched; the PR diff and existing review threads were checked in this task.

## Interpretation and next step

The batch demonstrates a working M4 training/recovery/evaluation pipeline and
some validation improvement over an untrained control. It does not demonstrate
reliable profit against random, and remains far below profitability against the
scripted pool. Do not promote either model or claim the v0.5 strength gate passed.

Analyze the retained fits, collection coverage and final-policy behavior before
fixing the next training plan. A longer overnight run remains a hypothesis to
budget and test, not an automatic continuation. No additional batch was started
as part of completing or merging this report.
