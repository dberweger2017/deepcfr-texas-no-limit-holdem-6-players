# Local full-game training with more work per iteration

**Running on the M4 after owner authorization and successful host benchmarks.**
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
**2026091802 and 2026091803**. Run the two seeds concurrently, one CPU thread each,
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
below 12 GiB, output exceeds 8 GiB, or worker RSS exceeds 7 GiB.
Concurrent training also stops if combined worker RSS exceeds 12 GiB, as
authorized by the owner on September 18. The worker uses
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

## Launch status — September 18, 2026

The [launch record](reports/holdem-local-launch.json) pins revision `2b6bff3`, with
full CI passed for its unchanged runner implementation at `7b4e237`. The M4 also
passed all 23 focused checks after updating to that source. Training started at
16:54 UTC / 18:54 Madrid under a persistent `nohup`/`caffeinate` supervisor, with
about 40 GiB free. Seed 2026091802 completed its first three iterations normally;
seed 2026091803 is queued. This is a launch snapshot, not a live status or a result.
Source remains pinned on the M4 while the batch runs. Check
`results/local-fullgame/status.json` there for current state. The laptop must stay
powered and awake; keep its lid open for unattended execution.

## Concurrent execution — September 18, 17:49 UTC

At the owner's request, both declared seeds now train concurrently. The
[handoff record](reports/holdem-local-parallel.json) verifies unchanged worker
PIDs: seed 2026091802 continued without a restart, and seed 2026091803 began at
17:47 UTC. At 17:49 UTC they had completed iterations 161 and 7, respectively,
using about 2.5 GiB combined RSS and approximately one CPU core each.

The [handoff supervisor](../scripts/m4_parallel_takeover.py) first stops the old
coordinator, validates and preserves its records, then replaces only that
coordinator. The existing training process groups continue. The original
coordinator cannot launch a duplicate seed. A second handoff raised the combined
RSS ceiling from 9 to the owner's requested 12 GiB; both workers again continued
unchanged. Existing per-worker six-hour elapsed allowances and 7 GiB ceilings
remain; final verification and testing still run sequentially within the shared
30-minute allowance. An adopted worker's completion is verified through its
completed result artifact because it is no longer a child whose exit code can
be collected.

The remote training checkout stays clean at `2b6bff3`. Only the separately
recorded supervisor is deployed under ignored `results/`; the trainer source
fingerprint is checked before handoff. Supervisor implementation is recorded
at `ae51c76`. Thirteen focused tests pass, including real subprocess handoffs
with one or two existing workers, prevention of the old parent's duplicate
launch, and termination of both workers on a combined memory violation.

## Simple live TensorBoard

The owner's requested four-chart dashboard is running on the M4, with one run
per seed: completed iterations (target 256), seconds per iteration, and validation
profit against random opponents and against the scripted style pool in BB/100,
each with its 95% interval. Poker points
arrive every 64 iterations; they are preliminary validation, not final-test
results. The second seed has no poker point until its first scheduled evaluation.

The reader was deployed from `e93263e` to ignored
`results/monitor_holdem_simple.py` and launched with `--simple` against the two
seed directories. Events live in `results/tensorboard-local-fullgame-simple-v2`;
monitor and TensorBoard logs/PIDs use the `results/local-fullgame-` prefix.
TensorBoard listens only on the M4's `127.0.0.1:6006`. On the controlling Mac,
an SSH tunnel exposes [the dashboard](http://127.0.0.1:16006/#custom_scalars).
Reconnect the tunnel if needed:

```sh
ssh -N -L 127.0.0.1:16006:127.0.0.1:6006 m4
```

The reader polls saved records every five seconds; TensorBoard reloads its event
files every 15 seconds. The browser's reload button refreshes the visible view.
The reader never loads model checkpoints or changes training. Sixteen focused
monitor/supervisor tests pass, including real event files and the four-chart
layout. Both runs and their charts were verified in the browser.
