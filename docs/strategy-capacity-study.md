# Strategy capacity and longer training

## Question

Does separating strategy-network capacity from advantage learning make the Deep
CFR baseline converge reliably as training continues? The previous CPU pilot
found a promising width-128 fit on three existing Leduc reservoirs, with a narrow
margin at the 0.15 exploitability limit. This study starts new training seeds,
examines longer trajectories, and keeps confirmation data separate.

The owner authorized the remaining **$10 total Runpod CPU budget**, including the
few cents already used by PR #49. GPU funding is separate. More cores are intended
to reduce elapsed time; neither four hours of runtime nor spending the budget is
a target. No additional payment or automatic account top-up is authorized.

**Completed:** the [results report](reports/strategy-capacity.md) retains all 144
outcomes. No recipe qualified; confirmation was skipped and the rental terminated.

## Implementation contract

`Config.strategy_hidden` controls the strategy network's two hidden layers.
When omitted, it follows `hidden`, which still controls advantage networks.
Training snapshots restore each network at its own width, and inference exports
load the strategy width. Tests cover unequal widths through save/load,
fresh-process resume, and public-observation policy inference.

Strategy fitting uses an independent initialization/minibatch random stream and
does not feed back into advantage training or traversal sampling. A regression
test compares all collected arrays, reservoir random states, traversal state,
advantage fit metrics, and advantage weights under different strategy capacities
and fitting schedules. A second test checks that a shared-collection refit matches
fresh end-to-end training with the same recipe at every selected checkpoint.

This allows one collection trajectory per exploration seed, followed by several
fits on each saved reservoir. It changes neither the sampled targets nor their
weights. The confirmation phase runs the chosen recipe from scratch through the
normal campaign runner; it does not reuse exploration data.

## Predeclared protocol

[strategy-capacity-v1.json](../configs/solver/strategy-capacity-v1.json) is the
authoritative configuration. The source and settings are committed before the
rental campaign. No settings or thresholds may change during execution.

### Hardware calibration

Candidate: one Runpod 5 GHz compute-optimized pod with 32 vCPUs, 64 GB RAM, and
20 GB temporary disk. The earlier console showed $1.12/hour for compute; verify
the actual quote before deployment. No persistent volume or GPU is required.

Run one 24-iteration Leduc job with seed 887, the same 1,024 traversals/player and
4,000 advantage fitting steps as the main experiment, and the baseline strategy
fit. Record its duration and a rough linear projection to 480 iterations. This
projection is a sizing estimate, not a convergence prediction; growing replay and
host contention may change actual runtime.

Then compare 4, 8, and 16 workers on the same sixteen four-iteration jobs, seeds
907–922. Require exact equality of non-timing reports and policy hashes at each
worker count. Choose the count with the shortest total wall time. The exploration
stage has twelve jobs, so actual concurrency never exceeds twelve there. Each
worker uses one Torch/BLAS thread. This is an ordered practical calibration, not
a claim about statistical significance or performance on other workloads.

### Exploration

Seeds: **211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271**.

Each seed collects 480 iterations of Leduc with advantage width 64, 1,024
traversals/player/update, 4,000 advantage fitting steps, batch size 256, learning
rate 0.001, and capacity 200,000 for each reservoir. These collection settings
preserve the earlier baseline. Evaluate and snapshot the baseline every 120
iterations; retain the 360-iteration baseline evaluation too. Compare all recipes
at **120, 240, and 480**:

| Recipe | Strategy width | Strategy fitting steps |
| --- | ---: | ---: |
| baseline | 64 | 6,000 |
| wider | 128 | 24,000 |
| longer-fit | 128 | 48,000 |
| largest | 256 | 24,000 |

This yields 144 reported recipe/checkpoint/seed outcomes from twelve independent
collection trajectories. Baseline outcomes reuse the original fit; 108 additional
fits reuse saved replay. No early learning-curve result changes the remaining
schedule. Record exact exploitability, game value, value error, fitting metrics,
coverage, sample counts, policy hashes, checkpoint hashes, and wall time.

### Selection

Only the **final 480-iteration** result is eligible for selection. A recipe must
meet both original Leduc limits on **every one of the twelve exploration seeds**:
exploitability at most **0.15**, absolute equilibrium-value error at most **0.10**.
Every planned fit and seed must be present, with failures retained. A timeout,
missing result, invalid state, or source mismatch prevents selection.

Among eligible recipes, select the lowest worst-seed exploitability, then the
lowest mean exploitability; an exact tie uses the recipe name. The baseline can
win if it is best. Earlier checkpoints explain the learning curve but cannot
replace the predeclared final checkpoint. Fitting error is diagnostic, not the
selection criterion. If no recipe qualifies, record `no_candidate` and do not use
confirmation seeds to tune it.

### Confirmation

Fresh seeds: **401, 409, 419, 421, 431, 433, 439, 443**.

Freeze the selected recipe, all source/configuration hashes, the selection report,
and hashes of all exploration reports before starting confirmation. Run eight
complete Leduc trajectories from scratch for 480 iterations. All eight must meet
the unchanged 0.15 exploitability and 0.10 value-error limits at the final iteration.

Also run the existing Kuhn recipe on the same eight seed numbers in that separate
game: 100 iterations, width 64, 1,000 advantage steps and 6,000 strategy steps, with
the original 0.03 exploitability and 0.03 value-error limits. Check both games
against the retained tabular report and the independent equilibrium solver.
Both all-seed gates must pass before milestone 3 can close. Keep every failure;
do not replace a disappointing seed or select an earlier checkpoint.

These checks establish small-game learning evidence. They do not qualify a
professional-strength Hold'em agent or promote a deployment model.

## Runtime, cost, and recovery

Local plans keep the 840-second limit. Longer runs must explicitly declare
`execution: "cpu-campaign"`, which permits a bounded training budget up to 7,200
seconds. This study allocates 5,400 seconds to each collection/confirmation run,
7,200 seconds to each full exploration worker including refits, and a process
watchdog with 60 seconds for startup and final writes. Each orchestration phase
also has a six-hour ceiling. These are safety limits, not intended durations.

The operator must enforce the **cumulative rental** limit independently: terminate
within seven hours of provisioning or earlier if the remaining CPU budget would
be exceeded. At the indicated rate, seven hours is about $7.84 plus temporary
disk, within the remaining authorization. Verify that calculation against the
actual quote. Estimate the remaining work after calibration and do not start a
phase that cannot fit inside the remaining time/budget. Stop as soon as the fixed
work is complete. No automatic extension, retries, or additional seeds.

The parent reaps active worker process groups on failure or interruption and
retains completed outputs and logs. It cannot stop provider billing. Copy and
hash-verify all results before stopping the pod, then terminate it and verify
zero remaining cost. Keep no billable storage behind. Training snapshots permit
same-source/runtime recovery through the existing explicit resume interface;
this study does not silently resume or discard failed attempts. Saved outputs
are immutable by directory name.

## Running the study

Use Python 3.11 and [requirements-pilot.txt](../requirements-pilot.txt). In the
reviewed checkout, run calibration first:

```bash
.venv-pilot/bin/python -m scripts.run_strategy_study \
  --study configs/solver/strategy-capacity-v1.json \
  --phase calibrate --out results/strategy-capacity-calibration
```

Use the measured worker count for exploration (16 shown only as an example):

```bash
.venv-pilot/bin/python -m scripts.run_strategy_study \
  --study configs/solver/strategy-capacity-v1.json \
  --phase explore --workers 16 --out results/strategy-capacity-exploration
```

If selection succeeds and the remaining rental budget is sufficient:

```bash
.venv-pilot/bin/python -m scripts.run_strategy_study \
  --study configs/solver/strategy-capacity-v1.json \
  --phase confirm --workers 16 \
  --exploration results/strategy-capacity-exploration \
  --out results/strategy-capacity-confirmation
```

Commit compact reports, complete learning curves, and provenance hashes. Preserve
raw replay/checkpoints outside Git with retrieval instructions and archive hashes.
