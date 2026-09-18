# Local full-game training with more work per iteration

**The owner authorized launch on the M4 after successful host benchmarks.**
The [M4 report](reports/holdem-m4-benchmark.md) records that admission. The separate
Runpod representation campaign in PR #85 is unchanged.

## Purpose and fixed recipe

Test whether the existing six-player learner produces useful poker when each
fresh network receives more collection and fitting work. This uses full-game
self-play from the start of each hand, not restricted reference targets. It does
not integrate the newer card representations, alter the bootstrap distribution,
or change the production default model.

The [executable plan](../configs/holdem/local-fullgame.json) declares six players,
100 BB stacks, blinds 1/2 integer chips, no rake or antes, and seeds
**2026091802 and 2026091803**. Run the seeds sequentially, one CPU thread each,
from uniform bootstrap. Both must be retained regardless of results.

| Setting | Earlier longer run | This batch |
| --- | ---: | ---: |
| Iterations per seed | 512 | 256 |
| Collection roots per role per iteration | 32 | 128 |
| Fresh Adam steps per role per iteration | 64 | 256 |
| Batch size | 32 | 32 |
| Replay capacity per role | 4,096 | 4,096 |
| Network width | 32 | 32 |
| Learning rate | 0.001 | 0.001 |

First-decision sampling, exploration 0.5, historical value baselines,
gradient clipping at 1, fresh role fitting and snapshot averaging remain intact.
Collection is bounded by 1,000,000 nodes and each whole iteration by 600 seconds.
Save full checkpoints and evaluate at 64, 128, 192 and 256.

Each seed schedules 196,608 roots across six roles and 393,216 optimizer steps:
twice the earlier total of each. Iteration 128 matches the earlier campaign's
scheduled root and optimizer-step counts, but not necessarily its wall time or
visited nodes. It has fewer policy updates and archived profiles. This jointly
changes collection and fitting allocation; it cannot isolate their effects.

At full replay, 8,192 replacement draws touch approximately 86.5% of 4,096
records in expectation, versus 39.4% for 2,048 draws. These are stored records,
not independent hands or information sets. More collection does not enlarge the
reservoir or guarantee better postflop coverage. Earlier frozen 256-step refits
did not consistently improve play. This proposal does not assume underfitting
is the proven cause of failure.

## Measured local feasibility

A separate seed, 2026091801, ran four timing iterations before the owner's
instruction to prepare only. The retained [timing summary](reports/holdem-local-timing.json)
records 160.54 seconds total, 113.14 seconds in training and 0.883 GB peak
training RSS. Saving the fourth checkpoint took 34.25 seconds and wrote
129,998,209 bytes. Tiny eight-block evaluations were pipeline checks, not model
selection. The raw pilot remains under the original checkout's ignored
`results/local-training-proposal-20260918/timing-run/` directory.

The original pilot used an Apple M1 with 16 GiB RAM and suggested 3–5 hours per
seed. The subsequent [M4 benchmark](reports/holdem-m4-benchmark.md) supports using
the M4 instead, with 2–3 hours per seed and unchanged six-hour ceilings. Earlier
completed 512-iteration jobs used about 5 GB peak process memory.

## Runtime and storage limits

The runner launches one seed per process and enforces a six-hour elapsed limit
per seed. Both jobs have at most twelve hours, followed by a shared 30-minute
recovery/final-evaluation allowance. Watchdog polling is every five seconds;
termination escalates to kill after five more seconds. Limits are wall-clock
allowances, not promises that both jobs finish.

Before launch require 20 GiB free. During every worker, stop if free space falls
below 12 GiB, output exceeds 8 GiB, or worker RSS exceeds 7 GiB. The worker uses
one Torch thread and does not spawn collection subprocesses. Limits are sampled,
so transient growth between checks is possible. Source and environment are
recorded; source must be committed and the worktree clean. Output directories
are new and never overwritten. Any worker failure stops the batch; completed
artifacts and the failure reason remain. No result-based retries or extensions.
A resource failure is an incomplete batch, not evidence against the learner.

Recheck M4 free space immediately before launch. PR #85 retrieval remains on
the M1 and does not share the M4 output volume. Do not
delete other experiments to make room. There is no paid compute and no change
to the Runpod budget. Interrupted jobs require an explicit reviewed continuation,
not an automatic restart with another fresh time allowance.

## Evaluation and interpretation

At each scheduled checkpoint, evaluate the snapshot average against the existing
five-style pool and against five copies of the repository's `random` policy.
Each suite uses 1,024 deal blocks, all six seat rotations and paired trained /
uniform-candidate arms: 12,288 table hands per suite. Both seeds use root seed
2026091810 and the validation split. Retain every checkpoint and repeated
validation result; do not choose a best checkpoint from these scores.

The fixed final endpoint is iteration 256. After both training jobs finish, load
each final checkpoint in a fresh process and reproduce the checkpoint and average
export hashes with no further training. Verified duplicate recovery bytes can be
removed; original checkpoints and inference exports remain.

Each final average then plays a fresh **4,096-block test**, root seed 2026091811,
`test` split, against random opponents and paired uniform-candidate controls.
That is 24,576 hero hands per seed, 49,152 table hands counting both arms.
Split-prefixed arena seeds separate final test from validation; the experiment
constructor also checks training/evaluation deal overlap.

Report absolute BB/100, paired improvement over uniform-candidate play, valid
hands and uncertainty over independent deal blocks. The two seeds times the two
endpoints use four Bonferroni-adjusted two-sided t intervals (98.75% each,
familywise nominal 95%). A seed passes the competence check only if both adjusted
lower endpoints exceed zero. The batch passes only if both seeds pass. Zero
observed variance or insufficient samples does not manufacture a confidence
interval. Report inconclusive results honestly; two seeds do not establish
population-wide training reliability.

Style-pool scores remain visible as a weakness check, with comparison to the
[earlier results](reports/holdem-longer-training.md). Historical published scores
are context, not fresh paired comparisons. No automatic default-model promotion,
release claim or professional-strength claim follows from this run.

Retain per-iteration street/position coverage, roots with postflop decisions,
sampling extremes, clipped steps, fitting metrics and stage timings already
emitted by the trainer. Preserve hand outcomes and learning curves. This PR does
not add a diagnostic sweep or change the original checkpoint schedule.

## Authorized M4 launch

The owner gave the conditional go on September 18; host checks passed. From the
committed M4 checkout with its `.venv` environment:

```sh
.venv/bin/python -m scripts.local_fullgame \
  --plan configs/holdem/local-fullgame.json \
  --out results/local-fullgame
```

On macOS, keep the machine awake for the approved run (for example launch the
command through `caffeinate -i`). The supervising runner must remain alive;
force-killing it or shutting down the Mac is not a supported unattended handoff.
The runner handles normal interruption and SIGTERM by terminating its worker
group. No launch agent or scheduled start is created during preparation.

The root `status.json` and worker logs expose progress. Each `seed-...` directory
contains the trainer's reports and artifacts; each `final-...` directory contains
recovery hashes, the frozen test plan, raw outcomes and adjusted endpoint report.
Following completion, update this PR and the roadmap with every result, measured
cost and artifact hashes before considering merge. A human-play UI is separate
release work; these are exports for the existing headless policy interface.

## Preparation validation

[Recorded checks](reports/holdem-local-preparation.json): 22 focused tests pass,
including watchdog expiry, worker failure, no launch with an exhausted allowance,
no second seed after a first-seed failure, final-test split and adjusted intervals.
The already-trained four-iteration pilot was loaded in a fresh process; its full
checkpoint and average export reproduced byte for byte. A 30-block test-split
smoke completed 360 hands with zero invalid actions. This verification performed
no optimizer steps and is not a campaign-seed or playing-strength result. Lint and
diff whitespace checks pass. Full CI is required before launch; no campaign is
launched by CI.
