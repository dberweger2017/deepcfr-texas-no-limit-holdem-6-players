# Additional branching and coverage

**The second expanded own decision passes the predeclared variance/cost screen on all three sampling seeds.** Integrate it as an explicit training option for a bounded online comparison. It does not resolve the measured coverage gap, and no stronger policy has been trained or promoted.

Results of the [committed protocol](../holdem-collector-branching.md); [complete compact measurements](holdem-collector-branching.json).

## What completed

One local CPU job completed **1,008 sampling cells / 13,824 estimates**, 24 exact river references and **288 hero-hand coverage cases** in **347.59 seconds (5.79 minutes)**, below the 900-second cap. No failures, retries, fitting or rental spend. The study reuses the original seed-307 policies and the failed seed-503 critics; 601/607/613 are sampling seeds, not independent trained models.

Control expands the first own decision on each path; candidate expands the first two. Both sample later own decisions with the same exploration mixture and inverse own-prefix correction. Opponent policies and target payoffs stay fixed. Production training still uses the original collector, and its replay converter explicitly rejects the experimental second-branch records pending integration.

## Primary result: less noise for the measured work

Candidate/control ratios, lower is better. Each seed equally weights four roots on each of flop, turn and river, using the historical baseline. Passing requires both aggregate ratios ≤0.75 and no street ratio >1.25.

| Sampling seed | Variance × nodes | Variance × seconds | Screen |
| --- | ---: | ---: | --- |
| 601 | 0.0750 | 0.0684 | Passed |
| 607 | 0.0496 | 0.0462 | Passed |
| 613 | 0.0636 | 0.0579 | Passed |

That is **92.5–95.0% less variance × nodes** and **93.2–95.4% less variance × seconds**. All nine street/seed guards pass; the largest node/time ratios are 0.1772/0.1701, both on seed 601's flop. None of the 36 historical-baseline root/seed cells regresses on either cost product in this sample.

Average traversal size rises from **70.12 to 84.75 nodes**, about 21%; historical-baseline time rises from **64.53 to 72.72 ms**, about 13%. The variance reduction more than compensates on these probes. These timing measurements include current Python and frozen-state integrity checks; they are not a projected production speedup.

The diagnostic controls show the same broad direction. These are equal-weight means over all full-stack roots and sampling seeds, not additional promotion screens:

| Baseline | First: variance × nodes | Second: variance × nodes | First: variance × seconds | Second: variance × seconds |
| --- | ---: | ---: | ---: | ---: |
| zero | 6,475,649 | 416,170 | 5,835.02 | 345.11 |
| accounting | 6,247,525 | 271,563 | 5,525.78 | 226.55 |
| historical | 6,493,961 | 406,120 | 6,376.24 | 364.55 |
| learned | 6,941,189 | 466,042 | 6,627.63 | 417.14 |

Accounting remains a useful inexpensive control. Changing the baseline as well as branching would confound the next online comparison, so keep the historical baseline for that comparison.

### Limits of the result

There are only **eight replicates per full root per sampling seed**, under one frozen, weak playing profile. These variance estimates can be sensitive to rare outcomes. The full-stack roots are the retained synthetic six-way call/check contexts with a 6 BB pot; each root fixes its hidden deal. This measures conditional action-sampling variance, not hidden-deal uncertainty, full-game bias or playing strength. Variance × cost is an asymptotic equal-budget proxy, not an actual equal-time training comparison or a confidence interval.

## Exact river controls

Unopened-river variance × nodes, averaged across 12 contexts and three sampling seeds:

| Baseline | First own decision expanded | First two expanded |
| --- | ---: | ---: |
| zero | 271.269 | 22.099 |
| accounting | 286.830 | 22.099 |
| historical | 270.041 | 22.099 |
| learned | 195.358 | 22.099 |
| oracle | 20.188 | 22.099 |

Expanding the second decision removes the remaining own-action sampling in these restricted trees, so all baselines yield identical candidate estimates. It does not remove opponent/hidden-world sampling. The candidate is slightly worse than the first-decision oracle on this cost metric; an exact oracle is a diagnostic, not a practically free learned baseline. Its timing excludes exact-reference construction.

Facing-all-in controls have the same estimates and 11 nodes across every baseline and depth, with mean variance × nodes 3.091. Their remaining variance is not caused by own-action sampling. Within each depth, all baseline variants retain identical execution hashes and node counts.

## Coverage: more records do not necessarily mean broader experience

Each profile has two deals per button and six hero views per deal: **72 correlated hero-hand cases**, covering every relative position. Natural self-play is run once and reused across hero views: **12 distinct self-play hands per profile**, not 72 independent hands. Style-pool lineups vary with hero. Deals are shared across profiles and modes. Position-level counts are retained in the JSON.

| Completed fits | First: cases with postflop / 72 | Second: cases with postflop / 72 | First: postflop / all records | Second: postflop / all records | Cases exercising extra expansion / 72 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0, uniform bootstrap | 13 | 25 | 22 / 365 | 65 / 517 | 46 |
| 32 | 20 | 32 | 33 / 430 | 204 / 725 | 51 |
| 128 | 12 | 13 | 30 / 341 | 225 / 576 | 42 |
| 256 | 1 | 2 | 1 / 193 | 2 / 197 | 15 |

The added expansion is exercised in **154/288 cases**, satisfying the protocol's natural-collection condition for considering an online comparison. But at fit 128, a large increase in postflop records comes with just one additional case reaching postflop. At fit 256 both collectors have almost no postflop experience.

| Completed fits | Self-play cases with postflop / 72 | Self-play postflop / all hero decisions | Distinct self-play hands with a preflop all-in / 12 | Style-pool cases with postflop / 72 | Style-pool postflop / all hero decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 / 110 | 12 | 36 | 64 / 148 |
| 32 | 5 | 8 / 112 | 10 | 19 | 37 / 119 |
| 128 | 0 | 0 / 90 | 12 | 30 | 71 / 150 |
| 256 | 0 | 0 / 74 | 12 | 16 | 40 / 112 |

The mismatch is visible **before learning**, under uniform-over-candidate bootstrap. The result supports investigating how the initial policy and subsequent self-play generate experience; it does not prove one cause of later learning failure. A preflop all-in does not by itself imply that every player has no later decision, so both all-ins and actual postflop decision counts are reported.

Extra branching cannot produce betting decisions after all-in continuations. Improving the quality of available updates and improving coverage remain separate tasks. Changing opponent sampling, the bootstrap distribution, action priors or replay weights requires an explicit account of the intended target and any sampling corrections.

## Decision and next task

1. Integrate optional second-decision branching into sampled collection, replay conversion, configuration and checkpoints. Preserve the original default and verify target normalization and fresh-process recovery for both modes.
2. Declare a short resource pilot, then a bounded multi-seed online A/B from bootstrap: original versus additional branching, all other fitting/policy settings held fixed. Report equal work and measured cost, coverage by street/position and paired playing results. Choose its runtime and evaluation budget from the pilot; this report authorizes no new paid campaign.
3. Keep coverage as an explicit outcome. If online learning remains starved of postflop decisions, design a separate distribution intervention rather than silently adding one to the branching test. Representation/generalization remains unresolved too.

No longer run, architecture sweep or new critic is justified merely by passing this diagnostic. The useful advance is a concrete collector change with a measured mechanism and a clear online test.

## Reproduction and validation

Measured revision: `7307528a16c42288359afedadc31e36df52298e3`. Protocol commit: `0354db1`. Source fingerprint: `4f2c4ae8d34f292aacfcc1fdb734bbb64381280ff5401bdf8ef80534b5c2eb9a`. The compact JSON retains the plan, environment, source/input fingerprints, all sampling cells, reference values, position summaries and artifact hashes.

Raw artifacts occupy **16.96 MB** under local `results/collector-branching/`: all estimates, collector records/executions, natural public histories and progress summaries. They are not publicly hosted. The hash-pinned original training checkpoint and four critic checkpoints are reused from previous local studies. No new model checkpoint is produced.

The artifact verifier reproduces all 1,008 cells, all 288 coverage cases and the passing screen. It replays natural actions through the engine to check actor, street, payment, all-in status, public history and zero-sum settlement. Exhaustive small-tree tests check expected root values and every counterfactual regret update, including later sampled decisions, zero policy reach and nonzero baselines. Pre-change fingerprints cover both original collector modes. End-to-end tests check retained failures, archive indexing and the cost-screen guards.

All **619 repository tests pass**, including nine branching/study checks. Targeted lint and diff whitespace checks pass. No game rules or player-information boundary changed.
