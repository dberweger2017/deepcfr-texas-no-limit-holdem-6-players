# M4 daytime replay and paper-inspired architecture comparison

The owner authorized three fresh training jobs on September 19, 2026, after the
completed overnight comparison. This batch stops at iteration 1,024. No model
promotion, additional seeds or paid compute is authorized by this protocol.

| Arm | Architecture | Replay per role |
| --- | --- | ---: |
| current-4k | Existing width-32 GRU and action-conditioned heads | 4,096 |
| current-16k | Identical model | 16,384 |
| paper-16k | Rank/suit/card embeddings, residual card tower, width-64 public-history GRU, residual joint trunk and LayerNorm | 16,384 |

All use fresh training seed 2026091902, first-decision sampling, exploration 0.5,
128 roots per role, 256 fresh fitting steps per role, batch 32, Adam 0.001 and
norm-1 clipping. Six players, 100 BB, no rake, unchanged legal observations,
action menu, Q/regret loss and snapshot-average semantics. A/B initialization
uses identical role/iteration RNG streams; trajectories and retained records
can diverge. C uses the same seed schedule with different parameter shapes.
This is one new training seed, not three independent replications.

C follows the representation principles in Brown et al., Deep CFR, §5.1:
https://proceedings.mlr.press/v97/brown19b/brown19b.pdf . It is a hybrid, not a
reproduction. Existing canonical suit encoding and unordered hole/flop groups
feed 64-dimensional rank+suit+card embeddings. Four pooled groups feed a
256→192→192→64 card tower. Non-card static context and event history feed a
64-wide GRU and betting branch; board indicators are excluded from that event
path. The joint state is normalized before the existing action-conditioned
regret and Q heads. Width, depth, embeddings and normalization change together;
a positive result identifies the whole architecture package, not a component.

## Evaluation and decisions

Scripted validation at 64, 128, 256, 512, 768 and 1,024 uses 1,024 blocks and
root seed 2026091910. Random validation is secondary at 256, 512 and 1,024.
At 1,024 each arm freezes and evaluates the scripted pool on 2,048 fresh blocks
with root seed 2026091920, in a separate `final` directory. Deal schedules are
checked disjoint from validation and from every generated training root.
The fresh suite uses the existing arena validation schema but its seed and
files are separate and never used in intermediate evaluations.

Predeclared primary comparisons: current-16k minus current-4k; paper-16k minus
current-16k. Cluster rotations by deal block; use Bonferroni-adjusted 97.5%
individual two-sided intervals to control 95% familywise coverage across the
two comparisons. Also report each arm's absolute profit and nominal interval.
No selection of intermediate checkpoints and no automatic production change.
Independent-seed evidence remains necessary for a default or release claim.

Existing collection, fitting/clipping, timing, and evaluation outcomes are
retained. Derive shove rates from saved actions during analysis. No new expensive
action-value probes or large diagnostics suite blocks this training batch.
Report elapsed cost as well as iteration count; equal fitting steps do not mean
equal compute across architectures.

## Storage, supervision and recovery

The merged inode-based storage fix is retained: hard links count once. Full
checkpoints every 64 iterations retain only the latest published state. Keep
one rolling average export and pin the final iteration-1,024 checkpoint/export.
Retain all small reports and losslessly compressed raw evaluation outcomes.
Checkpoint JSON records are now compressed losslessly inside the existing ZIP
container; weights and mathematical state are unchanged. Final evaluation
reuses a hard link to the frozen export rather than serializing a second copy.
Publication precedes retirement, so transient space for the previous checkpoint
and the new checkpoint is included in the launch estimate.

No per-worker or aggregate RAM cap, per the owner's standing instruction.
Measure RSS. Enforce 12 GiB free disk, 16 GiB unique output per job, 600-second
training-iteration bound, 30-minute checkpoint/validation bounds and a one-hour
fresh final evaluation bound. A STOP file requests a checkpointed stop. Failure
is recorded, with no automatic restart. Disk/RAM needs grow with the snapshot
archive; short pilots are not guarantees of final usage.

Launch only after invariance, finite learning, checkpoint/export recovery,
storage-accounting and finite campaign tests pass, plus separate-seed M4 pilots
with the actual learning budgets. Record parameter counts, fitting loss before/
after, checkpoint sizes and estimated final storage. Runtime is measured, not
promised to finish during daylight.

## Verified launch — September 19

[PR #89](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/89)
launched all three workers from clean revision `2c23594`.
The [launch record](reports/holdem-day-paper-launch.json) freezes their configs
and PIDs; the [pilot report](reports/holdem-day-paper-pilot.json) contains
measured timing, recovery and fitting results. All 747 repository tests passed
locally in 287 seconds. Nine focused architecture/continuous/storage tests
passed on the M4. Each production-budget pilot completed four iterations on
separate seed 2026091995, then passed byte-identical checkpoint recovery and
exact exported-policy distribution checks. A/B first fitted profile hashes
match; all six paper first fits reduced diagnostic loss, with a combined
7.3% reduction. This is a learning-plumbing check, not playing strength.

The controls average 18.3 seconds per early iteration; paper averages 22.7.
Linear early-cost estimates give 5.2 and 6.5 training hours respectively,
excluding evaluation and checkpoint time. Costs can grow with the archive.
There are 25,602 parameters per control role and 172,866 per paper role.
At 1,024 iterations, their raw six-role float32 archives are approximately
0.59 and 3.96 GiB. Reserving three archive copies across all arms allows about
15.4 GiB for overlapping rolling checkpoint/export publication; allow another
2 GiB for compressed replay, metadata and evaluation output. These are storage
estimates, not a RAM reservation. Peak RAM can exceed steady-state training RSS
during serialization, and there is no owner-imposed RAM cap.

On a complete old 16K replay checkpoint, JSON record compression reduced
1,281,410,158 bytes to 124,590,398 bytes in a streaming measurement. To restore
space, only the 8.8 GiB duplicate M4 transfer archive was removed after its
SHA-256 matched the local archive and the recorded overnight digest. Original
M4 overnight folders and the local archive remain intact. Free disk was about
37 GiB before launch, above the estimated batch peak plus the 12 GiB floor.

TensorBoard uses the existing port 6006/tunnel 16006, with the new monitor
writing under `tensorboard-m4-continuous/daytime`. First validation points arrive
at iteration 64. Training stops at 1,024 and the frozen fresh-deal scripted
evaluation runs automatically before recording completion.
