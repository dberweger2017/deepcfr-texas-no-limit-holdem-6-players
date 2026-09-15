# Fresh small-game readiness confirmation

## Question and frozen decision

Can one fixed end-to-end Deep CFR recipe meet the original absolute small-game
limits on every reserved training seed? This is an implementation-readiness
check before substantial Hold'em training, not a model promotion or a claim
about professional playing strength.

The [decision record](decisions/strategy-fitting-next.md) explains why cosine
strategy fitting is the next implementation. The historical fitting study remains
`no_candidate`: its additional 0.005 paired-regression veto failed. That veto
is explicitly **not** part of this prospective readiness test. We retain and
report every paired regression, without claiming uniform improvement.

The executable plan is [neural-readiness-v1.json](../configs/solver/neural-readiness-v1.json),
SHA-256 `c856f6430128896f65c43f1fb05a87a5f4cdc3c5053dfd25ed2a5767098a6de8`.
The runner verifies these bytes and requires a clean committed checkout for
confirmation. Freeze this PR before using the reserved seeds; use that same
revision throughout the campaign. Manifests retain the resolved configuration,
source hashes (including the runner), environment, machine, quote, and rental
start. No confirmation seeds or paid compute were used in this implementation PR.

## Training and scoring

Run each game from scratch with seeds **401, 409, 419, 421, 431, 433, 439, 443**.
These are eight independent training replicates per game, not sixteen independent
replicates of one game. Each seed has separate deterministic streams for
collection, reservoir admission, network initialization, and minibatches.

| Setting | Kuhn | Leduc |
| --- | ---: | ---: |
| Iterations | 100 | 480 |
| Traversals per player per iteration | 1,024 | 1,024 |
| Advantage hidden width (two layers) | 64 | 64 |
| Advantage updates per player per iteration | 1,000 | 4,000 |
| Strategy hidden width (two layers) | 64 | 128 |
| Updates per fresh strategy fit | 6,000 | 48,000 |
| Strategy learning rate | Constant 0.001 | Cosine 0.001 → 0.00001 |
| Advantage learning rate | Constant 0.001 | Constant 0.001 |
| Batch size | 256 | 256 |
| Capacity of each reservoir | 100,000 | 200,000 |
| Evaluated/saved iterations | 20, 40, 60, 80, 100 | 120, 240, 360, 480 |
| Scored iteration | 100 only | 480 only |
| Maximum exploitability | 0.03 | 0.15 |
| Maximum absolute value error | 0.03 | 0.10 |
| Training time limit per seed | 800 seconds | 5,100 seconds |

Other algorithm conventions remain unchanged: alternating player updates,
uniform reservoir/minibatch sampling, loss weights `2*t/T`, fresh networks and
Adam for each fit, gradient norm clipped to 1, and one deterministic CPU thread
per worker. Cosine sets the rate before each update and spans both endpoints.
The reference report hash and independent equilibrium checks are inherited from
the original campaign machinery.

Both limits must pass for **every** seed in both games. Earlier checkpoints are
diagnostics and recovery artifacts; do not choose among them. Missing runs,
invalid outputs, process failures, or exhausted budgets leave the campaign
inconclusive. Completed runs outside a numerical limit fail readiness. Keep all
attempts and failures. No threshold changes, extra seeds, longer fits, or automatic
retries after looking at results.

## Paired fixed-rate comparison

After each completed Leduc run, load its final verified training snapshot into a
separate solver. Check that its strategy hash matches the final report. Refit
only the strategy network for 48,000 updates at constant 0.001, with the same
width, replay, iteration weights, batch size, initialization seed, and minibatch
stream as the candidate. Save that control policy, evaluation, value error,
fitting metrics, source snapshot hash, and candidate-minus-control exploitability
difference. A positive difference is a regression. Report all eight pairs.

This costs **eight additional fits**, with a 600-second limit per fit. There are
no intermediate-checkpoint controls, additional fitting replicates, or Kuhn
controls: Kuhn already uses the constant recipe.

Sharing collection is justified by the solver boundary: traversal policies use
advantage networks only, and fitting uses local random generators. Tests compare
all replay entries, generator states, and advantage weights after changing
strategy fitting. A separate end-to-end test verifies that this snapshot refit
matches the fixed-rate policy from an independent constant-rate training run.
The candidate snapshot and inference export are never overwritten.

A worse fixed-rate comparison cannot veto an otherwise passing readiness run.
It also cannot be hidden by the aggregate: the report lists every positive delta.
An absolute pass does not turn the old exploration screen into a success.

## Resource and shutdown plan

Use one CPU rental, at least 16 vCPUs and 32 GB RAM, with eight single-threaded
workers. Start the eight Leduc jobs first, then fill free slots with Kuhn jobs.
Prior measurements put the older Leduc collection near 38–39 minutes per seed
on the previous host; new fits and different hardware add uncertainty. Allow
roughly an hour of useful work, then measure actual completion. Do not fill a
time allocation with extra training.

| Limit | Ceiling |
| --- | ---: |
| This rental, including compute, storage and transfer | $2.50 |
| Eligible all-in hourly quote | $1.20/hour |
| Entire rental, starting at provisioning | 7,200 seconds (2 hours) |
| Runner wall time | 6,600 seconds (110 minutes) |
| Individual worker, including oracle checks and control | 6,000 seconds |
| Retrieval/shutdown reserve | 600 seconds (10 minutes) |

The $2.50 ceiling fits within the existing $8.57 conservative CPU authorization;
it is not new funding. Compare actual quotes before provisioning. At the maximum
hourly quote, two hours costs $2.40, leaving $0.10 for charges not already included
in the quote. Decline offers whose total cannot fit the cap. GPU budget is separate.

Pass the actual provisioning timestamp and all-in hourly quote to the runner.
It subtracts setup time and the retrieval reserve from available runtime and
rejects an excessive quote or expired rental. Its supervisor kills and reaps
workers on error or timeout. Training and control fitting also check deadlines
internally. It stops when all jobs finish and never extends work to spend money.

**Process shutdown does not stop rental billing.** Before launch, establish a
separate provider stop/termination mechanism with a deadline at the rental cap,
and a completion/failure follow-up for early retrieval and shutdown. Verify
that it works on the chosen provider; do not depend on an interactive terminal
remaining connected. This PR creates no rental or unattended follow-up.

Retrieve logs, manifests, reports, policies, and training snapshots; archive them
with SHA-256, verify the downloaded archive and contents locally, then terminate
the pod and remove billable storage. On failure or approaching the cap, stop
compute promptly and retrieve available evidence within the reserve. Record
actual charges or a conservative quote-based bound, including setup and cleanup.
Never reconnect a terminated pod from a previous study.

## Commands and artifacts

Short local validation, using only seeds 101, 103, and 107 with two iterations,
small networks, serial workers, and a five-minute supervisor cap:

```sh
python -m scripts.run_neural_readiness --smoke --out results/readiness-smoke
```

This exercises both games, collection, fitting, final controls, child processes,
and reporting. It reports `smoke_completed`, never a readiness pass.

After the committed checkout, rental preflight and shutdown mechanism are ready:

```sh
python -m scripts.run_neural_readiness --run \
  --hourly-usd "$READINESS_HOURLY_USD" \
  --rental-started-at "$READINESS_RENTAL_STARTED_AT" \
  --out results/neural-readiness
```

Set the timestamp to the actual timezone-qualified provisioning time, such as
`YYYY-MM-DDTHH:MM:SS+00:00`; the price must include recurring storage charges.
The runner retains one job directory per seed, per-game verified summaries,
all paired comparisons and regressions, and an overall report. Failed workers
leave their logs and partial artifacts; the overall report remains inconclusive.
Do not restart the whole campaign in a new output directory after a failure.

Publish the full seed table, absolute decisions, paired deltas, fitting errors,
run/environment hashes, all failure records, and compute cost. If readiness
passes, close milestone 3 and proceed with the corrected Hold'em learner.
Otherwise preserve the outcome and make a bounded architectural decision,
including Single Deep CFR. Independent Hold'em engineering can continue in either
case; substantial training remains gated.
