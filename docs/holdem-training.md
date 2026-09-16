# Hold'em replay and role fitting

[PR #60](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/60) lets the replacement learner run a complete **collect → admit replay → fit
roles → publish models** iteration. It uses the existing
[external-sampling collector](holdem-collection.md), [betting targets](holdem-betting.md)
and [decision inputs](holdem-encoding.md). This is a tested training loop, not a
validated competitive agent. [PR #61](holdem-baseline.md) adds averaged play,
persistence and the arena runner. Fresh small-game readiness and substantial
training remain ahead.

## Replay ownership and admission

A `RoleReservoir` belongs to one physical seat. It stores immutable `ReplaySample`
records containing the complete per-candidate target, iteration, collection
profile fingerprint, action seed and target index within the traversal. Exact
observations and actions stay attached to the target. These are host-side training
records; the network receives only the candidate inputs, never their payoff labels
or collection metadata.

One record represents one traverser decision, with all its action targets kept
together. Candidate count, regret magnitude and payoff do not affect admission.
The reservoir keeps every record until full. For incoming record number `n`, it
then draws a uniform integer in `[0, n)` and replaces that slot only when it falls
below capacity. Thus every encountered record has equal retention probability.
The `seen` counter advances even when a record is not retained.

Each role has its own admission RNG. Sampling a fitting minibatch uses a separate
caller-owned RNG and samples uniformly with replacement. Diagnostics use another
stream. Fitting cannot change which future examples the reservoir admits.
Cloning a memory copies its slot list, counters and RNG state while sharing only
immutable records. The replay fingerprint covers all of these, plus role,
capacity and its version, for reproducibility checks.

`split_collection` checks the complete scheduled traversal list and its
iteration/profile/seed ownership before admission. Targets must match all
recorded traverser decisions without omissions or duplicates. Their histories
must extend the same root and retain the same private owner. Replay validation
regenerates the public candidate input and checks finite, aligned policy/value/
regret vectors and their baseline relationship. A rejected admission changes no
reservoir state. These checks protect the internal data contract; they are not
cryptographic authentication of externally supplied simulation results.

## Iteration-weighted fitting

`fit_role` creates a **fresh network and Adam optimizer** for each nonempty role
memory, following the existing small-game baseline's fresh-fit convention. It
does not continue optimizing the previous collection model. The initialization,
minibatch and diagnostic streams have separate names and include iteration and
physical role. Fitting runs on deterministic CPU float32 and restores the caller's
CPU RNG state and thread/determinism settings.

For replay record `j` collected at iteration `t[j]`, during current iteration `T`:

```
error[j] = sum over candidates a of (
    (predicted_regret[j,a] - target_regret[j,a])²
  + (predicted_value[j,a]  - target_value[j,a])²
)
loss = mean over sampled records j of (2 * t[j] / T) * error[j]
```

The weight is applied to both heads and every candidate. It is linear in the
collection iteration, matching the reference loss scale; it is not normalized
by the sum of the minibatch's weights. Targets from a future iteration are
rejected. Uniform admission plus weighted fitting is deliberate: there is no
prioritized replay or duplicate weighting during reservoir replacement.

A candidate with zero policy probability still has its own target and gradient.
The action-conditioned network learns from branch payoffs, not from a sizing
head copying its prior guess. Regrets and values remain in BB. The auxiliary value
head retains the betting interface's equal loss weight and shared encoder; its
effect on eventual learning quality still needs evaluation.

Each fit takes its configured number of Adam steps, with norm-1 gradient clipping
and nonfinite-gradient rejection. A fixed uniformly selected subset of retained
records measures the loss before and after fitting. This is an **in-sample fitting
diagnostic**, not held-out poker evaluation or evidence of playing strength.
Changing the diagnostic subset size does not change the optimizer updates. A
noisy fit is not silently retried or extended because its diagnostic worsened.

## Whole-iteration publication

`HoldemTrainer` starts every physical role with an explicit uniform policy and an
empty reservoir. A step proceeds as follows:

1. Rotate the button for this iteration (unless explicitly disabled), freeze the
   current profile and collect all scheduled traversals. No fitting
   happens during collection.
2. Validate the completed batch against the trainer's table, seed, iteration,
   traversal count and frozen profile, then separate its targets by physical role.
3. Clone the memories and admit the new records into those staged copies.
4. Fit each active role with a nonempty staged memory. Other roles retain their
   prior model; a role that has never received a target remains uniform.
5. Check the original frozen profile, the newly fitted models and the deadline,
   then publish the models, memories, collection-profile archive, iteration counter
   and report in one state
   replacement.

A role can legitimately finish a traversal without acting. No extra deals are
sampled just to fill that role's memory. If its memory already contains older
examples, those still support the scheduled fresh fit; if it is empty, the report
records no fit. On a sparse table, absent physical seats stay absent from fitting.
There is no pooling of private records or network parameters across role memories.

If collection, admission, fitting or the final deadline check fails, the previous
trainer state remains intact. This includes a failure after earlier roles have
finished fitting. Retrying from that state with the same algorithm inputs and
sufficient limits reproduces a clean run. The guarantee covers this serial API's
internal updates; it does not support concurrent callers mutating trainer state.

The update schedule remains **simultaneous collection followed by role fits**,
as declared in the collector contract. It is not an alternating fit-after-each-
player schedule. Reports record both the profile used for collection and the
newly fitted profile, along with per-role new/seen/stored counts and diagnostics.
`new_samples` counts incoming records, including ones reservoir sampling discards.

```python
from src.game.hand import Table
from src.holdem.fitting import FitConfig
from src.holdem.training import HoldemTrainer, TrainConfig

trainer = HoldemTrainer(
    Table(tuple(f"player-{i}" for i in range(6)), (200,) * 6,
          small_blind=1, big_blind=2, chip_unit="1"),
    TrainConfig(seed=7, capacity=128,
                fit=FitConfig(width=16, steps=8, batch_size=8)),
)
report = trainer.step()
current_policy = trainer.current_profile()
```

`current_profile()` returns an isolated current regret-matched policy.
`average_policy()` returns the collection-aligned playing mixture. Complete
checkpoint/resume and inference export are documented in the [baseline contract](holdem-baseline.md).

## Validation

```bash
python -m scripts.check_holdem_training --players 6
```

The bounded check runs two collect-and-fit iterations at 100 BB with width-16
networks, eight fitting steps per role and capacity 128, then repeats from scratch.
It compares iteration reports, model fingerprints, replay contents, counters and
reservoir RNG states. The JSON includes configuration, table, source fingerprint,
revision and engine/environment identity. A test repeats the command in a fresh
Python process and checks the same results. These short checks are not a training
campaign or a test of convergence.

The focused tests include an exhaustive capacity-two/five-record reservoir check:
each of the ten possible retained subsets occurs equally often across all 60
possible admission draw sequences. A controlled fit with contradictory labels
at iterations 1 and 3 learns the expected 1:3-weighted mean rather than the
unweighted mean. Gradient tests check every action and both heads directly.

Other checks cover action/observation/iteration/profile alignment, incomplete
batches, admission rollback, RNG separation, source/frozen model isolation,
two-iteration updates at four/five/six players, sparse physical seats, empty
memories, later-role failure and exact retry. Averaging, recovery, learning
readiness and competitive evaluation have not been established by these tests.

**Validation:** all 42 focused tests and all 475 repository tests pass locally.
Ruff and `git diff --check` pass. No paid compute or training campaign was used.

## Following work

Averaged play, recovery and the first arena report are delivered in
[PR #61](holdem-baseline.md). Complete small-game snapshot recovery and fresh
readiness checks before substantial Hold’em training. The historical validation
counts above describe PR #60; current baseline evidence is in the
[first report](reports/holdem-baseline.md).
