# Frozen Hold'em fitting diagnosis

**Completed:** all 72 declared fits finished within the local budget. See the [results](reports/holdem-frozen-fitting.md); the protocol below is retained unchanged.

## Question and fixed comparison

The longer six-player experiment clipped every optimizer step and barely changed
its sampled diagnostic loss. Neither fact establishes the cause of weak play.
Before changing the training recipe, refit the retained iteration-256 replay with
**clip norm 1 or no clipping**, crossed with **64 or 256 optimizer steps**.

Run all six roles of seeds **307, 311 and 313**, in that order, regardless of seed
307's result. Its first fit provides a runtime check, not a selection screen.
[The plan](../configs/holdem/frozen-fitting.json) pins the three input checkpoints.
Use fresh width-32 models and Adam at 0.001, batch size 32, the existing weighted
MSE and unchanged sampling/iteration/root normalization. All four arms share the
original initialization and minibatch random streams; 64-step runs use the prefix
of the 256-step sequence. No new hands are collected and no trainer defaults change.

## Measurements and interpretation

Retain each role/seed/arm, both diagnostic losses, unclipped gradient maximum,
clipped-step count, fit time, peak process memory, and hash-pinned fitted weights.
Score **every retained replay record** after each fit, reporting regret and value
components separately using the production objective's normalization. Also report
the fraction of records whose predicted regrets are all nonpositive. The original
64-step clipped fit is a control; compare its parameters with the saved model.
Cross-platform floating-point differences may prevent bitwise reproduction of the
Linux training run on macOS, so record the difference rather than hide it.

Full-replay losses measure empirical fitting, **not held-out generalization**.
Noisy sampled action targets are not true action values. Role results within a
training seed are correlated; do not treat 18 roles as 18 independent seeds.
Report paired changes by role and by seed, including regressions. A consistent
reduction in regret loss across seeds would justify an online confirmation of that
change. Mixed effects call for inspecting variance/coverage before scaling. This
experiment alone cannot promote a model or prove stronger poker play. The separate
current-policy versus snapshot-average arena comparison follows this diagnosis.

## Execution and limits

One CPU worker at a time, one torch thread, at most **15 minutes per seed and
45 minutes total**. No rental. Extract only the three selected checkpoints and
manifests; keep the original archive intact. Each seed uses a fresh process to
release loaded checkpoint memory before the next. Preserve all completed cells,
logs and partial reports. Stop on a hash/provenance mismatch, numerical failure,
invalid checkpoint or deadline; mark remaining seeds unattempted, never omit them.
Do not extend the run after inspecting losses. The supervisor enforces worker
timeouts; archive preparation counts against the total budget.

```sh
python -m scripts.check_holdem_fitting \
  --plan configs/holdem/frozen-fitting.json \
  --archive results/longer-05.tar.gz \
  --out results/frozen-fitting
```

Commit the protocol and validated runner before starting. Record the executed
revision and environment with the results. Publish compact measurements separately
from large checkpoints. This is a bounded diagnosis of the existing sampled
Hold'em trainer, not another small-game tuning campaign.
