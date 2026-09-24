# Bounded tabular blueprint pilot

The first six-player, 100 BB blueprint plumbing test completed on the always-on M4. This is an implementation and cost check, not evidence of competent play or a v0.5 qualification.

## Frozen plan and result

The [pilot plan](../../configs/blueprint/pilot-v1.json) ran two iterations, one fresh root for each of six traversers per iteration, a 30,000-node and 100,000-entry cap, and five minutes per iteration. It used the compact [pilot abstraction](../blueprint-pilot.md). No rental was used.

| Measure | Result |
| --- | ---: |
| Iteration 1 | 1,523 nodes; 517 terminals; 231 entries |
| Iteration 2 | 1,110 nodes; 376 terminals; 408 total entries |
| Full process wall time | 1.68 seconds (prior to adding the card probe) |
| Peak resident memory | 348,454,912 bytes (about 332 MiB) |
| Arena | 24/24 scheduled hands completed; 0 invalid actions |
| Arena reproduction | 24/24 hands; identical retained outcomes |
| First-to-act preflop coverage | 2/169 canonical hand classes with trained entries |
| First-to-act policy variation | 3 distinct distributions, including the uniform fallback |

AA and 72o both use the uniform fallback at the fixed first-to-act probe. The two-block arena comparison is explicitly inconclusive: it has no usable confidence interval, and its BB/100 is not a strength estimate. The probe's sparse coverage shows why the pilot policy must not be promoted.

The uninterrupted two-iteration checkpoint and the run resumed from the one-iteration checkpoint have identical SHA-256 hashes. The arena export is separate from the training checkpoint and refuses to resume training.

All 750 repository tests pass on the M4 at the final code revision (186.54 seconds). The focused blueprint and arena tests also passed locally. No test or arena failure was omitted.

| Artifact | SHA-256 |
| --- | --- |
| Training checkpoint | `5ec58e9d49e5f1956ae666d202fc87c011d1192b20036e68f547812bcc2a84db` |
| Frozen current-policy export | `54d2edfceda2993bc6017fb390ec435416497222d912b53dd37571168ea66fd0` |

The complete artifact directory is `results/blueprint-pilot-v1-diagnostic/` on `m4` under the repository root. A hash-verified copy is at `results/blueprint-pilot-v1-m4/` in the local repository checkout. Both directories are ignored by Git and contain the checkpoint, export, manifests, card probe, hand-level arena rows, and reports. The M4 run used Python 3.11 and the pinned `pokers` engine. Its dependency versions and revision are retained in `manifest.json`.

## Decision

The first-test plumbing passes: complete six-role iterations, bounded failure behavior, exact resume, frozen legal play, and reproducible arena output work on the M4. Its 16 GiB memory is ample for this tiny pilot, but two iterations do not establish how a much larger tabular store will grow. No model is promoted and no v0.5 criterion is met.

Next, declare a coverage-oriented learning pilot with enough fresh deals to train many preflop classes, explicit memory and runtime caps, and a fresh scripted-pool validation schedule. Evaluate whether policy decisions become card-sensitive and whether the store remains within M4 RAM before choosing a rental or implementing online search.
