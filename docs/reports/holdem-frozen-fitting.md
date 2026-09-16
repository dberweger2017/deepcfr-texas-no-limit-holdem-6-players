# Frozen Hold'em fitting comparison

All **72 fits completed**: three training seeds, six roles and four fitting settings. No new self-play hands, rentals, retries or seed selection. Production defaults remain unchanged. This measures how the optimizer fits existing replay, not how well the resulting policies play poker.

## Protocol and verification

The [protocol](../holdem-frozen-fitting.md) and [plan](../../configs/holdem/frozen-fitting.json) were committed before the run at `a302b36631b95b5af08e6cc561674206f97d322c`. The inputs are the longer experiment's hash-pinned iteration-256 checkpoints for seeds 307, 311 and 313. Every role was refitted from the same seeded fresh initialization with clip norm 1 or no clipping, at 64 or 256 steps. All arms share minibatch prefixes; width 32, Adam 0.001, batch 32, targets and production loss weighting are unchanged.

All checkpoint hashes/provenance validated. All 72 output-weight hashes verified. Every role's four arms had identical initial diagnostic loss. The norm-1 arms clipped every step; the unclipped arms clipped none. The control was compared with its original Linux checkpoint; small macOS floating-point differences are retained below. All 36 focused checks and the 574-test CI suite pass. Focused tests also verify the explicit default against the unchanged call, matched sample prefixes, replay/RNG preservation, full-replay measurement against the production loss, and deadline/configuration rejection.

The [complete JSON report](holdem-frozen-fitting.json) retains every role, both loss components, parameter differences, diagnostics, timings, source/environment provenance and weight hashes. No noisy target-action rankings are treated as ground truth.

## Full-replay regret loss

Changes below are relative to the same seed/role's clipped 64-step control; **negative means lower empirical loss**. The mean gives each of six roles equal weight. The summed column compares the six role objectives added together and is more sensitive to roles with large target variance. These are descriptive summaries, not confidence intervals or independent poker results.

| Seed | Clip | Steps | Mean role change | Summed objective change | Roles with lower loss |
| --- | --- | ---: | ---: | ---: | ---: |
| 307 | 1 | 64 | +0.0000% | +0.0000% | 0/6 |
| 307 | 1 | 256 | -0.2412% | -0.8366% | 4/6 |
| 307 | none | 64 | -0.0027% | +0.1297% | 3/6 |
| 307 | none | 256 | -0.3690% | -0.8733% | 5/6 |
| 311 | 1 | 64 | +0.0000% | +0.0000% | 0/6 |
| 311 | 1 | 256 | -0.0491% | +0.0228% | 4/6 |
| 311 | none | 64 | +0.0921% | +0.1322% | 3/6 |
| 311 | none | 256 | -0.2246% | -0.1736% | 5/6 |
| 313 | 1 | 64 | +0.0000% | +0.0000% | 0/6 |
| 313 | 1 | 256 | -0.1979% | -0.1465% | 4/6 |
| 313 | none | 64 | -0.0534% | -0.0165% | 4/6 |
| 313 | none | 256 | -0.5282% | -0.6419% | 5/6 |

### Every role

Entries are percent changes in full-replay regret loss against that role's clipped 64-step control. Preserve mixed effects instead of choosing the best role or training seed.

| Seed | Role | Clip 1, 256 steps | No clip, 64 steps | No clip, 256 steps |
| --- | ---: | ---: | ---: | ---: |
| 307 | 0 | +0.2416% | -0.0904% | -0.1336% |
| 307 | 1 | -0.0339% | +0.0075% | +0.0012% |
| 307 | 2 | -0.2034% | -0.0165% | -0.1441% |
| 307 | 3 | -0.2271% | +0.0631% | -0.0672% |
| 307 | 4 | +0.7215% | -0.4153% | -0.2775% |
| 307 | 5 | -1.9457% | +0.4352% | -1.5929% |
| 311 | 0 | -0.0436% | -0.1035% | -0.4160% |
| 311 | 1 | +0.1441% | -0.0487% | -0.0725% |
| 311 | 2 | -0.7357% | +0.1637% | -0.7199% |
| 311 | 3 | +0.5050% | +0.5773% | +0.3738% |
| 311 | 4 | -0.0173% | +0.0037% | -0.0833% |
| 311 | 5 | -0.1469% | -0.0401% | -0.4297% |
| 313 | 0 | -0.0522% | -0.1818% | -1.1201% |
| 313 | 1 | +1.1606% | +0.2567% | +0.7351% |
| 313 | 2 | -0.7720% | -0.3198% | -0.6045% |
| 313 | 3 | -0.6425% | -0.1487% | -1.5847% |
| 313 | 4 | -1.0796% | +0.1264% | -0.4036% |
| 313 | 5 | +0.1983% | -0.0531% | -0.1912% |

## Resources and controls

One CPU process and one Torch thread per seed, sequentially on the local Apple M1. Replay scoring evaluates all 4,096 retained records per role after each fit. It is **in-sample**; no generalization claim follows from a lower result. Raw loss includes estimator noise, so a small relative decrease does not establish that the useful signal was learned or that greater capacity is necessary.

| Seed | Loading seconds | Complete job seconds | Peak process RSS (GiB) | Largest control parameter difference |
| --- | ---: | ---: | ---: | ---: |
| 307 | 106.70 | 159.32 | 2.570 | 2.29e-06 |
| 311 | 97.42 | 155.42 | 2.808 | 3.18e-06 |
| 313 | 97.69 | 150.77 | 2.872 | 5.54e-06 |

Total campaign time, including extracting and verifying the three input checkpoints: **530.61 seconds (8.84 minutes)**, inside the committed 45-minute cap. No rental spending. The original longer-run archive remains intact.


Local artifacts live in `results/frozen-fitting/`: the original manifest and supervisor result, three seed logs/reports, verified checkpoint inputs, and each fitted role's state dictionary. Weight paths in the JSON are relative to `seed-SEED/`. These are diagnostic role weights, not snapshot-average exports or newly promoted models; future reproduction needs the pinned input archive and executed source revision.

## Interpretation and next decision

Removing clipping alone at 64 steps is not a consistent improvement: summed regret loss rises for seeds 307 and 311 and barely falls for 313. Increasing clipped fitting to 256 steps also produces role regressions and a slight summed regression for seed 311.

The unclipped 256-step arm gives the clearest empirical signal: mean role regret loss falls **0.369%, 0.225% and 0.528%**, with lower loss in five of six roles for each seed. Summed regret loss falls **0.873%, 0.174% and 0.642%**. That makes it a candidate for a future online comparison, not a production replacement. The effects are modest, one role per seed regresses, and the data were used for fitting. We have not established a corresponding improvement in policy decisions, convergence or poker winnings.

The useful conclusion is narrower than “clipping broke training”: the combination of more fitting and no clipping deserves comparison, while removing clipping by itself has no consistent support here. Do not repeat this sweep at more widths or budgets before checking the played policies.

Next, compare the saved current policy and snapshot-average policy on the same declared arena deals before committing to another online training recipe. This separates empirical fitting behavior from the policy actually used for play. Keep the existing recipe and all historical results; do not turn an in-sample loss change into a strength claim. If that comparison cannot isolate a useful change, focus the next experiment on target variance and replay coverage rather than repeating small optimizer sweeps.
