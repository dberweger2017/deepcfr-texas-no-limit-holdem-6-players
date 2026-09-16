# Averaged play, recovery and the first Hold'em report

The replacement trainer now has an end-to-end path: collect all roles, fit their
replay, retain the collection strategies, export the average, and evaluate it in
the fixed arena. This completes the remaining milestone 4 engineering. Its
learning-readiness and strength conditions remain separate requirements.

## Playing the average

Iteration `t` retains exactly the frozen profile used to collect that iteration,
with linear weight `t`. The first profile is explicitly uniform over legal
candidates. A just-fitted profile enters the average only after it has been used
for collection. Failed iterations publish neither new weights nor an archive
entry.

`trainer.average_policy().distribution(observation)` reconstructs all of the
owner's earlier decisions from the public event prefixes. Each prefix receives
only the board cards available at that time. A snapshot's weight is multiplied
by its probabilities of the owner's earlier actions; opponent actions do not
contribute likelihood factors. Log weights prevent long histories from
underflowing. An unreachable own history has an explicit uniform completion.
Own actions outside the versioned abstraction are rejected; opponents can make
any legal bet because their exact actions remain public input.

`trainer.average_policy().player(seed)` is the practical playing interface. It
selects one iteration with linear weights at the start of each hand and retains
that policy for the hand. Each player has independent snapshot and action
streams; physical seats select the corresponding role model. The policy accepts
only `Observation`, including when the table's participants change. It rejects
repeated or backwards observations within a hand. A new player instance starts
a new random stream; callers should keep an instance throughout a session and
use unique public hand identifiers.

The behavioral mixture and one-snapshot-per-hand play implement the same
own-reach weighting. Sampling a fresh snapshot at every decision would not.
No evaluation policy receives the simulated deal seed, hidden cards, other
players' observations, or opponent policy identities.

## Complete recovery

Training checkpoints are immutable, hash-pinned artifacts at completed iteration
boundaries. They contain the table/configuration, current models, complete
collection archive, role reservoirs with exact targets and provenance, reservoir
counters/random states, and iteration reports. Seed-derived collection and fit
streams are reconstructed from the saved configuration and iteration number.
The button rotates deterministically between training iterations, so a physical
role does not remain tied to one position. Fixed-button experiments can explicitly
set `rotate_button=false`.

Each role fit deliberately creates a fresh network and Adam optimizer. There is
no optimizer carried between iterations. The checkpoint records that contract;
an interruption during collection or fitting restarts the entire uncommitted
iteration. Mid-fit recovery is not supported and would need a new contract.

Artifact publication writes and fsyncs a temporary file, then creates the final
name atomically without overwriting an existing checkpoint. A separate immutable
marker is published only after the checkpoint exists. Recovery selects the last
complete marker; an interrupted write cannot replace a good checkpoint.

The artifact contains canonical JSON records and CPU float32 weights loaded with
PyTorch's `weights_only=True`. Record types come from a closed registry. Loading
checks hashes, format/encoding/action contracts, profile alignment, replay
ownership, counts and collection provenance. Training also requires an identical
run manifest. The experiment runner checks source, environment, rules and plan
before accepting recovery. Inference exports contain only the archive and its
provenance; they cannot be used to resume training.

Snapshots store every iteration's role models. Memory and artifact size therefore
grow linearly with iterations. This is intentional for the correctness baseline;
measure archive cost before larger campaigns.

## Running a declared experiment

```bash
python -m scripts.train_holdem --plan configs/holdem/baseline-check.json --out results/holdem-baseline
python -m scripts.train_holdem --reproduce results/holdem-baseline --out results/holdem-reproduced
```

For an intentional stop and fresh-process recovery:

```bash
python -m scripts.train_holdem --plan configs/holdem/baseline-check.json --stop-after 1 --out results/holdem-paused
python -m scripts.train_holdem --resume results/holdem-paused --out results/holdem-resumed
```

Use a fresh output directory. After a failure, resume the failed directory in the
same way. `failure.json` records the exception, elapsed time and completed,
unfinished and unattempted jobs; it is diagnostic metadata, never recovery state. Completed job checkpoints are reused; evaluation is regenerated where
needed. Historical intermediate evaluation files remain in the original run;
recovery's result summarizes the final boundary. The final result, weights,
replay, archive and hand outcomes must match uninterrupted execution. Timing is
reported separately and is not expected to match.

Save and evaluation intervals are independent of fitting. Evaluation loads a
frozen inference export and has separate random streams, so its frequency cannot
change the training result. Every final boundary is saved and evaluated. The
runner checks that the declared collection deals and evaluation deals are
disjoint, rotates seats, uses paired candidate/uniform-baseline deals, retains
hand outcomes and failures, and reuses the arena's block-level uncertainty.
A failed hand invalidates the evaluation instead of reducing its denominator.

## Predeclared implementation report

`configs/holdem/baseline-check.json` declares three independent training seeds
(101, 103, 107) in each of four scenarios: four/five/six players at 100 BB, and
six players at 20/40/60/100/150/200 BB. Each is trained separately. Two iterations,
width 16, eight fitting steps per role, 128 records per reservoir and one root per
role per iteration deliberately bound this to implementation evidence. The total
local runtime cap is 900 seconds per invocation, with a 60-second whole-iteration
cap and 50,000 collection-node cap. No rental is involved.

Final evaluation uses 30 independent deal blocks per scenario/seed, all seat
rotations, and five frozen style opponents against a paired uniform-candidate
baseline. Each seed is reported separately; repeated evaluation deals across
training seeds are not independent additional deal samples. There is no
best-checkpoint selection, no hyperparameter selection, and no promotion rule:
**this check cannot promote a model, whatever its observed win rate.**

Acceptance requires legal completed evaluations, exact reproduction, and
interrupted/uninterrupted agreement. BB/100 and 95% intervals describe this
short-run comparison only. Two iterations cannot establish convergence or useful
capacity. The small-game readiness gate remains failed until a fresh declared
confirmation passes; substantial Hold'em training and professional-strength
claims remain gated.

The [first recorded run](reports/holdem-baseline.md) is incomplete: ten jobs
completed, one hit its collection limit twice and one was not attempted. All
completed results are inconclusive; the bounded check did not pass in full.
