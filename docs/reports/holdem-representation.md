# Reliable-target representation comparison

## Outcome

**No architecture qualified on validation.** More capacity and the separate card branch improved training fit, but did not reliably reduce decision cost on unseen boards. The validation rule selected no candidate; only the original model was evaluated on the reserved test set. Production defaults remain unchanged.

## What was tested

The [frozen protocol](../holdem-representation.md) compares five architectures on 288 exact six-player river contexts: 24 boards, six hero holdings per board and two betting situations. Sixteen entire boards train the networks, four choose at most one candidate, and four are reserved for the final test. The [compact results](holdem-representation.json) retain exact targets, all learning curves and final per-decision predictions.

A shared 48-component joint opponent range is generated from two templates and every global suit permutation. Conditioning on visible cards leaves 1–24 equally weighted components per context. All compatible worlds and both fixed public continuation policies are enumerated; their regret/value targets are combined with weights one and two. Unlike the earlier diagnostic, no new arbitrary hidden assignments are drawn for each board.

These are legal shallow river situations starting from 2 BB, with 1 BB left at the river. This is a conditional regression benchmark under explicit correlated ranges, not unrestricted 100 BB self-play or a professional poker benchmark. The networks see only player-visible features. No production model, sampler, loss or checkpoint format changes.

Each architecture ran 1,024 Adam steps on the same per-seed minibatch schedule (batch 32, LR 0.001, clip norm 1), for initialization seeds 811, 821 and 823. Width/depth/card comparisons use scaled numerical inputs; the unchanged and scaled width-32 models share their initial parameters. Wider/deeper/card-separated models have comparable parameter counts, but different compute costs.

## Final training and validation results

Relative error is regret RMSE divided by exact-target RMS. Decision cost is the expected BB forfeited versus the best immediate action under the fixed reference continuations. It is not exploitability, and the exact regret-matching policy need not greedily maximize these weighted Q values.

| Model | Seed | Training error | Validation error | Validation TV | Validation cost (BB) | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 811 | 16.06% | 73.46% | 0.206 | 0.4322 | 14.57 |
| original | 821 | 21.63% | 67.77% | 0.152 | 0.2509 | 15.16 |
| original | 823 | 16.63% | 74.04% | 0.214 | 0.5443 | 13.07 |
| scaled | 811 | 13.97% | 79.36% | 0.275 | 0.6723 | 19.62 |
| scaled | 821 | 12.94% | 69.18% | 0.196 | 0.3231 | 19.72 |
| scaled | 823 | 14.96% | 69.38% | 0.166 | 0.3069 | 19.48 |
| wide | 811 | 8.36% | 70.70% | 0.248 | 0.6062 | 23.05 |
| wide | 821 | 8.19% | 70.03% | 0.168 | 0.4679 | 22.59 |
| wide | 823 | 8.93% | 74.11% | 0.221 | 0.7370 | 20.69 |
| deep | 811 | 5.47% | 79.40% | 0.148 | 0.6073 | 22.55 |
| deep | 821 | 7.73% | 72.60% | 0.220 | 0.5448 | 22.12 |
| deep | 823 | 19.78% | 78.34% | 0.191 | 0.3919 | 20.45 |
| cards | 811 | 4.70% | 68.40% | 0.167 | 0.4672 | 24.12 |
| cards | 821 | 2.86% | 72.64% | 0.167 | 0.4669 | 21.90 |
| cards | 823 | 8.11% | 76.10% | 0.282 | 0.5814 | 20.72 |

## Selection and reserved test

The committed rule requires at least 0.02 BB and 10% lower validation decision cost on every seed, relative regret error no more than 0.02 above original, and training relative error at most 0.20. The scaled model passes on seed 823 only. Wider, deeper and card-separated models pass on none of the three seeds. None qualifies, so no alternative is evaluated on test and there is no second-choice retry.

Average validation decision costs are 0.4091 BB (original), 0.4341 (scaled), 0.6037 (wide), 0.5146 (deep), and 0.5052 (cards). The wider and card-separated models are worse than original on every seed by this measure. Some individual error/TV comparisons improve, but they do not establish better decisions.

The exact regret policy has mean decision cost 0.00146 BB on training, 0.00068 BB on validation and 0.00125 BB on test. This small nonzero reference cost is expected because weighted regret matching is not greedy maximization of weighted Q.

| Model | Seed | Test error | Test TV | Test cost (BB) |
| --- | ---: | ---: | ---: | ---: |
| original | 811 | 119.07% | 0.238 | 0.6441 |
| original | 821 | 118.44% | 0.206 | 0.4062 |
| original | 823 | 114.44% | 0.261 | 0.5100 |

The original network calls with Ac Kd on Qs Js 8d 5c 2h when facing the all-in in all three test seeds, again paying a 1 BB decision cost. Under this experiment’s newly declared shared range, the exact values are −1 BB for folding and −2 BB for calling. That result was recomputed; it was not assumed from the older two-world experiment.

### What the learning curves show

| Model | Mean training error at 1,024 | Validation cost at 128 | Validation cost at 1,024 |
| --- | ---: | ---: | ---: |
| original | 18.11% | 0.178 | 0.409 |
| scaled | 13.96% | 0.178 | 0.434 |
| wide | 8.50% | 0.178 | 0.604 |
| deep | 10.99% | 0.190 | 0.515 |
| cards | 5.23% | 0.179 | 0.505 |

These are descriptive curves, not permission to select an earlier checkpoint. Fitting the training mapping more closely coincides with worse average validation decision cost after step 128 for every variant. That supports a transfer/overfitting concern under this recipe. It does not establish that any architecture was optimized to its best possible result.

Errors are uneven across boards. On the double-paired validation board 9h 9s 5c 5d Ac, mean cost ranges from 1.1465 BB for original to 1.7291 BB for wide; on Ad Td 7d 4c 3h, original/scaled have zero cost while wider/deeper have about 0.27 BB. Board-level concentrations and the raw predictions are retained; four validation boards do not establish a universal architecture ranking.

15,359 of 15,360 fit steps clipped. That is a diagnostic of this fixed recipe, not a new clipping ablation or evidence that removing clipping would solve transfer.

## Cost and verification

All 576 context/profile references and all 15 fits completed in **1046.64 seconds (17.44 minutes)** on one local CPU thread. Exact enumeration visited 4,042,120 nodes. Recorded enumeration time was 718.87 seconds; fitting including scheduled train/validation measurements took 299.80 seconds. Calibration passed the separate 1,800-second stage gates with a 50% allowance. No deadline extension, failed arm, adaptive retry or rental was used.

The committed implementation was `1a56d81109cb6ba946b32f4c56d47ae5b1326eec`. The source/config fingerprints, environment and every model hash are retained. Fresh-process verification reloads all 15 weights and reproduces final training/validation and permitted test metrics exactly. A separate arithmetic audit checks target weighting, physical payoff bounds and the fold accounting value. All 647 tests and repository end-to-end checks passed on the implementation commit; the final report commit must pass CI before merge.

Raw results remain in `results/representation/`: `report.json`, `contexts.json`, `targets.pt`, all 15 model files, `selection.json`, calibration, verification and the execution source archive. These are local diagnostic weights, not resumable full-game checkpoints or public model downloads. The compact report includes artifact hashes and verification records. No paid compute was used; the $4.33 remaining CPU authorization is unchanged. Reproduction requires the recorded source and dependencies; test measurements must not be used to tune another candidate on this same reserved set.

## Decision and limits

Keep the current production encoder. Do not start a larger self-play run with a wider or deeper model based on these results. The specific separate-card MLP also does not qualify. These findings reject promotion of the tested configurations, not larger networks or poker-specific representations in general.

The next bounded diagnostic should separate **board diversity from explicit card relationships**: freeze the original/card-branch architectures, compare learning as training-board coverage grows, and include a control with exact own-hand/board features computed only from visible cards. Such features must not contain opponent holdings, reference Q values or range-conditioned equity. Use new validation/test boards because the current sets have now informed research decisions. Calibrate reference-generation cost first, retain common fitting controls and decision-cost measurements, and do not turn this into another width or learning-rate sweep.

This follows the observed pattern: added capacity fits known contexts better, while unseen-board behavior stays poor. A plain separate MLP still has to discover poker’s card relationships from only 16 training boards. More representative data and explicit legal card features are distinct hypotheses worth separating. Sparse full-game street coverage and extreme sampled targets remain unresolved independently of this diagnostic.

Only four validation boards and four test boards are held out. Initialization seeds share those contexts; they are not independent samples of poker games. The result is a finite benchmark comparison, with no population confidence claim. The range is small and deliberately structured, the stacks are shallow, and the histories vary only by the two diagnostic prefixes. No conclusion about long-history GRUs versus transformers, full-game convergence or required professional-model capacity follows from this experiment.
