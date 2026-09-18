# Continuous M4 baseline

The owner authorized one baseline training run with no fixed iteration or elapsed
limit, until explicitly stopped, with the existing four-chart TensorBoard view.
Future modified agents are separate runs; this authorization starts only one.

The [recipe](../configs/holdem/continuous-m4.json) uses fresh seed 2026091901,
uniform bootstrap, six players at 100 BB, width 32, Adam 0.001, norm-1 clipping,
256 fresh fitting steps per role, batch 32, 128 roots per role, replay 4,096 per
role, first-decision collection and exploration 0.5. It does not resume or select
one of the completed seeds. Production learning code is reused unchanged.

There is genuinely no overall iteration or wall-clock target. Each iteration
still has the 600-second training bound; checkpoint and evaluation phases have
30-minute guards. Training/evaluation deal disjointness is checked at every
iteration, rather than constructing an infinite schedule in advance. The manifest
records the continuous recipe and null total limits. It is a continuous-run
manifest, not a finite `train_holdem --resume` manifest.

Save and evaluate every 64 iterations, against the existing scripted pool and
random opponents on the same 1,024-block validation schedule as the completed
M4 campaign. These are monitoring results, not new held-out release evidence.
There is no automatic final test, promotion, restart, or additional seed launch.

## Storage and resource protection

Keep the latest two complete recovery checkpoints and latest two policy exports.
Retire older files only within this new output directory after a newer file is
published; record retired filenames/hashes in `retention.jsonl`. Older recovery
boundaries will no longer be resumable. Each remaining checkpoint retains the
full policy archive, current models, replay, reports and seed-based iteration
state. Retain every evaluation report, learning curve and timing record; compress
raw outcomes losslessly to `.json.gz`, checking decompressed SHA-256 before
removing the uncompressed copy. The completed PR #86 artifacts are untouched.

Supervision polls every five seconds. Stop on worker RSS over 7 GiB, combined
continuous-worker RSS over 12 GiB, less than 12 GiB free disk, or this run's output
over 8 GiB. Require 20 GiB free at launch. Resource or phase failure stops the
worker and preserves its previous published checkpoint; it does not silently
restart. Sampling between checks and growing policy archives mean a genuinely
infinite run on finite hardware is impossible. These safeguards may stop it
before the owner does. Every future parallel job must join this resource policy.

## Start, inspect and stop

From the clean, committed M4 checkout:

```sh
nohup caffeinate -i .venv/bin/python -m scripts.continuous_holdem \
  --plan configs/holdem/continuous-m4.json \
  --out results/baseline-continuous-2026091901 \
  > results/baseline-continuous-2026091901-launch.log 2>&1 < /dev/null &
```

The sibling `-supervisor.json` records supervisor/worker PIDs, memory and storage.
The run's `status.json` records phase and completed iterations. Request a graceful
stop by creating the stop file:

```sh
touch results/baseline-continuous-2026091901/STOP
```

It finishes the current bounded operation, saves the last completed iteration,
and records `stopped`; it does not start another evaluation after noticing STOP.
SIGTERM/SIGINT to the supervisor request the same stop. Emergency resource stops
may discard work since the previous checkpoint. Stop-file requests are persistent
and restarting is an explicit separate action.

The monitor uses `scripts.monitor_holdem --simple` and reads saved files only.
The new baseline appears beside both completed seeds at the existing forwarded
TensorBoard URL, `http://127.0.0.1:16006/#custom_scalars`.

## Verification

Launched on the M4 on September 18, 2026 at approximately 21:19 UTC from clean
revision `5cd7e0bd15e374b7b4afec5c184a0716744d0110`. The [launch snapshot](reports/holdem-continuous-m4-launch.json)
confirms two completed iterations and an active supervised worker. All 17 focused
tests passed locally and on the M4; a separate supervised three-iteration smoke
stopped normally. TensorBoard lists the new baseline alongside both completed
seeds. The first scheduled playing-strength evaluation is at iteration 64.

The bounded three-iteration smoke uses a separate seed and tiny fitting budgets.
It verifies successful stop, two retained recovery checkpoints/exports, all six
compressed outcome files, all six learning-curve entries and byte-identical
checkpoint recovery. The original finite runner and monitor tests remain in the
focused validation set. The CLI's `--stop-after` is for such finite verification;
it is omitted from the actual continuous launch.
