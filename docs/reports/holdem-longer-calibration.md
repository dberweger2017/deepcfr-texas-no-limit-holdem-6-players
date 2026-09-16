# Longer-run cost calibration

**The larger recipe completes, checkpoints recover exactly, and all four evaluation suites are valid.** The final plan is 512 iterations for each of three independent seeds, with an estimated **2–4 hours elapsed when run in parallel**. The campaign has not started, and no rental was created.

## What was measured

The [calibration protocol](../holdem-longer-training.md#local-cost-calibration) and executable configuration were committed at `7f3f268` before measurements. Seed 997 ran four iterations on six-player 100 BB Hold'em with width 32, 32 roots per role per iteration, 64 fresh-fit steps, batch 32, and replay capacity 4,096 per role. A final evaluation used 30 blocks in each of the style, random, previous-model and crossplay suites. The initial 128-iteration campaign proposal was increased to 512 using these cost measurements before any campaign seed was used; the learning recipe itself was unchanged.

| Iteration | Collection (s) | Fitting (s) | Replay (s) | Total (s) | Nodes | Stored records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.12 | 7.27 | 0.50 | 9.93 | 9,429 | 1,004 |
| 2 | 2.90 | 4.75 | 0.14 | 7.82 | 4,214 | 1,494 |
| 3 | 2.07 | 5.16 | 0.12 | 7.38 | 3,587 | 1,910 |
| 4 | 4.33 | 6.07 | 0.31 | 10.74 | 4,488 | 2,392 |

Total collect/fit time was **35.87 seconds** (mean **8.97 seconds/iteration**). Saving the iteration-4 training checkpoint took **9.99 seconds** and wrote **33,207,041 bytes**. Checkpoint work is material even in this small run and must be included in the larger budget.

| Evaluation suite | Hands, both paired arms | Wall seconds | Candidate mean / p95 action latency (ms) |
| --- | ---: | ---: | ---: |
| Styles | 360 | 1.04 | 0.56 / 1.15 |
| Random | 360 | 1.05 | 0.49 / 1.01 |
| Previous model against styles | 360 | 1.03 | 0.42 / 0.77 |
| Crossplay against five previous-model copies | 360 | 1.91 | 0.41 / 0.89 |

All 1,440 evaluation hands completed, with zero invalid actions or failed hands. Poker outcomes and all intervals are retained in the [JSON report](holdem-longer-calibration.json) and raw bundle, but were not used to choose settings or select a model.

The entire experiment took **54.40 seconds** (57.06 including process startup/shutdown), on an Apple M1 with one Torch thread and 16 GiB system RAM. Peak process RSS was **671,875,072 bytes**. This is whole-process high-water memory, including save/evaluation work; the replay was not yet saturated and the archive had only four profiles.

## Time estimate and budget

A constant-cost extrapolation gives **76.5 minutes of collect/fit work per seed** for 512 iterations. Scaling the measured evaluation work to 256 blocks and four evaluation boundaries gives another **2.9 minutes per seed**. Neither estimate accounts for mature replay/archive serialization, longer action histories as play changes, remote CPU speed, three-process contention or artifact transfer. In particular, extrapolating the initial 10-second checkpoint unchanged would understate the cost.

Provision **2–4 hours elapsed** for three independent seed processes, including setup, evaluations and artifact retrieval. Each process has a **three-hour work ceiling**; the rental has a **four-hour / $3.50 ceiling** within the existing $7.33 CPU authorization. This is not a provider quote or a booked rental. Verify current offers and measure the selected host before publishing a more precise ETA. No early stopping or extensions based on poker results.

The committed [main plan](../../configs/holdem/longer-05.json) now has 512 iterations, checkpoints every 64 and evaluations every 128. There are 98,304 scheduled traverser roots per seed, 294,912 across the three seeds, and 147,456 scheduled evaluation hands across the four suites. Seeds 307, 311 and 313 remain untouched. Keeping width 32 supplies a fixed first learning baseline; this does not establish that the network has enough capacity for strong poker.

## Verification and retention

All **565 repository tests pass**, including archived inference, paired equality of identical models, private per-seat samplers, missing-source reproduction, reference hashes/table compatibility, timing failure records, learning-curve recovery and single-seed CLI selection. Tests also retain the earlier exact-gradient and training-recovery checks. Timing measurements are separate from deterministic model state.

A separate process restored the final calibration checkpoint and reran all four evaluation suites with no additional training. Ten model/result/curve/outcome files match byte for byte, including the recovery checkpoint, current inference export, pinned reference and all four sets of hand outcomes. The local retained archive contains 53 verified files from the original and recovered bundles, logs, the verification inventory and test output:

- Archive: `results/longer-calibration.tar.gz`
- Size: 16,997,428 bytes
- SHA-256: `82a3371f7ddae18138b8ce52ea1cac259aaabfe29fae9319ea7f53ae07173c46`

Keep all resumed-run segments: copied learning curves retain earlier rows, while earlier checkpoint files remain in their original segment. Future release artifacts should include every seed's final average policy and full recovery state, source/dependency manifests, benchmark version and retrieval instructions. Model binaries stay outside Git.

The experiment manifest marks the workspace dirty because documentation edits and the unrelated untracked `.claude/` directory were present; the training implementation and calibration configuration were committed, and source bytes are separately pinned. There were no training/evaluation failures, retries, rental charges or model promotions in this calibration. This prepares the longer v0.5 experiment; it does not publish v0.5 or claim improved poker strength.
