# Frozen Hold'em policy comparison

**Completed:** see the [three-seed results](reports/holdem-policy-comparison.md). All twelve comparisons finished; no fitting recipe is promoted. The pre-run protocol below remains unchanged.

## Questions

The [fitting study](reports/holdem-frozen-fitting.md) found small empirical loss
improvements, with mixed role effects. This experiment asks two separate questions:

1. Do the four refitted current profiles make different decisions and play
   differently against the fixed style pool?
2. Does the original iteration-256 current profile play differently from its
   original historical snapshot average?

No new fitting or self-play collection occurs. No refit is inserted into the
historical archive. Production defaults and model promotion are unchanged.

## Frozen inputs and diagnostics

[The plan](../configs/holdem/policy-comparison.json) pins the committed fitting
report by SHA-256. That report pins all three input checkpoints and all 72 refitted
role-weight files. Use every role of seeds 307, 311 and 313 regardless of results.
Verify checkpoint, manifest and model hashes before evaluation. Record complete
profile fingerprints, the historical archive fingerprint list's digest, source
revision, plan, dependencies, schedules and output hashes.

On every retained replay record, compare clip-1/64 with clip-1/256, unclipped/64,
unclipped/256 and original current-256. Report total-variation distance (half the
sum of absolute action-probability differences), fractions above 0.05 and 0.10,
entropy in nats, nonpositive-regret fallback frequency and action mass grouped by
fold/check/call/raise. Group by physical role and street, including counts; preserve
an all-streets summary. Missing streets have zero observations, not zero effect.

Use the actual production regret-matching function, including its argmax fallback
when all regrets are nonpositive. No target argmax is treated as ground truth.
These are uniformly weighted **retained replay records**, potentially repeated or
correlated, not unique information sets, policy visitation frequencies, or a
held-out generalization test. Coverage counts verify the preflop-heavy replay
hypothesis directly. Action-kind mass hides sizing differences; total variation
still distinguishes every legal candidate size. Positive regret rescaling should
leave the policy unchanged even when regression error changes.

## Paired arena

Six-player, 100 BB, no-rake Hold'em under the existing rules and legal betting menu.
Use the five-style evaluation pool and **1,024 fresh validation deal blocks**, root
seed **120913**, all six seat rotations, and unchanged action/deal/opponent streams.
The schedule is identical across arms and training seeds. The original campaign's
validation seed 91891 is not reused.

| Candidate | Paired control | Question |
| --- | --- | --- |
| Clip 1 / 256 steps | Clip 1 / 64 steps | More fitting |
| Unclipped / 64 steps | Clip 1 / 64 steps | Remove clipping |
| Unclipped / 256 steps | Clip 1 / 64 steps | Both changes |
| Original current-256 | Original average-256 | Current versus historical average |

Combine each refit arm's six role networks into one immutable profile. Each player
receives only its own legal observation, and each identity has an independent
random stream. Current-profile play uses the same action-sampling stream as
snapshot play; only historical average play samples a historical component once
per hand. Opponent copies share no private state.

Each comparison schedules 1,024 × 6 × 2 = **12,288 table hands**. Twelve comparisons
across three seeds total **147,456 scheduled hands**, including repeated control
arms and correlated seat rotations. Repeated controls are not additional evidence.
No separate control-versus-itself arena run is needed; deterministic tests verify
that identity case and replay diagnostics measure original-versus-refit control
numerical differences.

## Analysis and decision rules

Report every candidate/control rate, paired BB/100 difference, nominal 95% block
interval, latency, completed/failed hands and invalid actions. Also report
**Bonferroni familywise 95% intervals across all 12 planned comparisons**, using
Student-t intervals over paired block means. The correction does not make shared
schedules or roles independent and does not cover future experiments.

Predeclare **100 BB/100** as a coarse diagnostic material-effect margin. This is
large relative to competitive poker edges and is not the professional-release
standard. An adjusted interval wholly above +100 or below -100 indicates a
material benchmark difference. An interval entirely inside [-100, +100] supports
similar performance only at this coarse resolution. Overlap with zero otherwise
remains inconclusive, not equivalence. Never extend hand count after inspecting
intervals. Report training seeds separately; do not pool 18 roles as independent
runs or promote the most favorable seed.

The primary fitting hypothesis is unclipped/256 versus clipped/64. Consistent
positive paired estimates across all seeds, with family-adjusted intervals clearing
zero in all three, justify prioritizing an online confirmation. Mixed/inconclusive
results do not select an optimizer winner. Current beating average identifies a
benchmark difference, not an averaging implementation defect or a general robustness
claim. If the policies remain weak without clear refit gains, favor target-variance
and replay-coverage work over another small optimizer sweep. No outcome here alone
promotes a default model or establishes professional strength.

## Execution and limits

Run locally, one seed/process and one Torch thread at a time. **15 minutes per seed,
45 minutes total**, including checkpoint loading and diagnostics. No rental. The
saved fitting inputs already contain the three checkpoints; no archive re-extraction
is needed. Preserve raw hand outcomes as compressed JSONL, all diagnostics, model
provenance, timings, partial results and failures. Stop on hash/provenance errors,
invalid actions, accounting failures, non-finite predictions or deadline. Mark
remaining seeds unattempted. Do not change the schedule or recipe after results.

```sh
python -m scripts.compare_holdem_policies \
  --plan configs/holdem/policy-comparison.json \
  --refits results/frozen-fitting \
  --out results/policy-comparison
```

Validate invariance under regret scaling, sensitivity to sign changes, fallback
behavior, street denominators, isolated/public-only current play, same-stream
one-component average equivalence, paired schedule identity, deterministic outcome
files, familywise interval calculation, and retained failure traces before starting.
Commit the protocol and validated implementation before measurements; publish all
results, including inconclusive comparisons, in the same task's PR.
