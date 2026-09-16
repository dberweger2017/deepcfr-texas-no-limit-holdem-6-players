# Longer Hold'em training and release comparisons

## What trains

This is real neural counterfactual-regret self-play on **six-player no-limit Texas Hold'em**, 100 BB starting stacks, the documented no-rake cash rules, and complete player-visible current-hand history. All six physical roles collect and fit; opponents in training are the frozen current self-play profile. Evaluation opponents never supply training targets.

The recipe uses the validated sampled Deep CFR variant: first-own-decision expansion, sampled continuations, frozen value baselines and snapshot-average play. The game engine implements legal no-limit actions, while the learner uses the documented finite [betting menu](holdem-betting.md). It is not an exhaustive solver over every possible chip-sized raise. The earlier Kuhn/Leduc work is complete; this experiment trains Hold'em networks.

## Longer experiment

[Executable plan](../configs/holdem/longer-05.json):

| Setting | Value |
| --- | --- |
| Independent training seeds | 307, 311, 313 |
| Iterations | 512 per seed |
| Collection | 32 roots per role per iteration; 98,304 roots per seed |
| Sampler | First-decision expansion; exploration 0.5; frozen baseline |
| Model width | 32, unchanged from the pipeline pilot |
| Fitting | Fresh role models/Adam; 64 steps, batch 32, LR 0.001 |
| Replay | 4,096 records per role, with the full admitted-stream count retained |
| Recovery checkpoints | Every 64 iterations and the final boundary |
| Played-policy exports and evaluations | Iterations 128, 256, 384, 512 |
| Evaluation schedule | 256 independent blocks per suite; fixed validation seed 91891 |
| Iteration ceiling | 250,000 visited nodes / 180 seconds |
| Per-seed process ceiling | 3 hours including evaluation and checkpoint work |
| Intended execution | Three independent processes, one seed each; one Torch thread per process |

This is 8× as many roots per iteration and 4× as many fit steps (with twice the minibatch size) as the pilot, over 512 rather than 6 iterations. Keep this one recipe fixed. Do not select the best seed or stop early because a plotted result looks good. Retain failure records and the last complete checkpoint if any limit is reached. Positive win rate is not an entry requirement, completion criterion or reason to extend the budget.

**Initial planning estimate: 2–4 hours elapsed for the three seeds in parallel, including evaluation and artifact handling.** This is a provision, not a measured remote ETA. The [local calibration](reports/holdem-longer-calibration.md) measures 7.38–10.74 seconds per early iteration and 9.99 seconds for its first checkpoint. This supports increasing the initial 128-iteration proposal to 512 before any campaign seed is used; a rented host's first completed iterations must refine it again, with growing replay/archive costs reported. The three-hour child-process limit is a ceiling, not a promised completion time.

**Rental envelope:** at most four hours from provisioning to termination and **$3.50 total**, including setup, evaluation, retrieval and retained storage charges, within the existing $7.33 CPU allowance. Choose an offer only after verifying its current price and CPU/RAM allocation; no rental was created by this plan. Allow at least three independent workers and enough measured RAM headroom. Retrieve and hash-check artifacts before termination; keep no billable storage afterwards. GPU spending remains separate. If the current offer or measured throughput cannot fit these limits, shorten the declared work or present a revised budget before launching the main campaign; do not silently extend an active run.

Run one declared seed per process, each into a fresh directory:

```bash
python -m scripts.train_holdem --plan configs/holdem/longer-05.json \
  --seed 307 --out results/longer-05-seed-307
```

Repeat for 311 and 313 in separate processes. `--seed` rejects values outside the committed plan and cannot override a resumed experiment's seed. Running the plan without `--seed` runs all three sequentially under one shared three-hour ceiling; that is not the intended rental arrangement.

## Fixed performance benchmarks

At every evaluation boundary, play all four suites with the same seat rotations and paired deal schedule. Each suite has 256 blocks × 6 seats × 2 arms = **3,072 table hands**. That is 12,288 hands per checkpoint, 49,152 per training seed, and **147,456 evaluation hands** across the complete campaign. Correlated rotations, paired arms, repeated checkpoints and copies of a model are not independent training seeds.

| Suite | Candidate and paired control | Five opponents |
| --- | --- | --- |
| `styles` | New model vs uniform-candidate policy | Fixed style pool |
| `random` | New model vs uniform-candidate policy | Random policy |
| `previous` | New model vs pinned previous model | Fixed style pool |
| `crossplay` | New model vs pinned previous model | Five independent copies of the previous model |

The random policy samples legal action kinds and uses minimum/all-in raise sizes; it is not uniform over every legal chip amount. Uniform-candidate play is a different baseline that samples the learner's betting menu. Random play is a basic competence check. Stronger conclusions need the style pool, historical comparisons, fresh held-out evidence and eventually the professional benchmark.

The first anchor is the six-player seed-211 iteration-6 pilot model, chosen by the smallest declared seed, **not its score**. Its file and SHA-256 are in the plan. Its provenance and weakness remain visible. Obtain it from the retained [pilot archive](reports/holdem-sampled-pilot.md); fresh checkouts intentionally do not contain model binaries. The experiment snapshots the verified reference bytes into its own `models/` directory, so later resume/reproduction uses those bytes even if the original file moves. No archived model learns during evaluation, and copies have independent private random streams and histories.

## Metrics and checkpoints

Each job writes:

- `training-timing.jsonl`: per-iteration collection, replay, fitting and total wall seconds; visited nodes, admitted/stored records, archive count, process peak RSS and success/failure. These measurements never enter model state, seed streams or checkpoint hashes.
- `checkpoint-timing.jsonl`: checkpoint serialization time and bytes.
- `timing-ITERATION[-SUITE].json`: evaluation wall time and decision latency. Report mean and p95 latency, throughput, and the machine/runtime alongside strength.
- `learning-curve.json`: every evaluated iteration and suite, BB/100, paired difference and 95% block intervals, hand count, invalid actions and policy hash. Resume preserves prior evaluations; re-evaluating a boundary replaces that row rather than counting it twice.
- `evaluation-*` and `outcomes-*`: full reports and raw hand outcomes, including failed evaluations.
- `artifacts.jsonl`: exact model/checkpoint paths, hashes, iterations and training seeds. Full recovery checkpoints retain replay/counters/RNG; smaller average-policy exports are the models used for play and cross-version comparisons.
- Existing iteration reports retain both-head fitting losses, maximum importance weights/regret updates, pre-clipping gradient norms and clipped-step counts. Loss includes estimator noise and is not a playing-strength metric.

Peak RSS is process-wide high-water memory, not current live tensor memory. Archive storage and replay size grow; six-iteration throughput is not a guarantee about iteration 512. Retain all seeds and scheduled checkpoints, with source/dependency manifests and these measurements. Store artifacts separately from Git and include retrieval instructions and hashes in the campaign report.

## Comparing v0.5, v0.6, v0.7 and later

Keep the final policy export for **every training seed** at each version. Give it a readable alias plus source revision, training recipe, training seed, iteration and SHA-256. A version name alone must never identify mutable model bytes. If a future network or observation schema changes, retain its older inference adapter or isolated runtime with the saved source and dependencies; do not reinterpret old weights under new semantics. The generic arena accepts `format: "holdem-average-v1"` alongside the legacy adapter, so any two compatible saved models can be candidate/baseline/opponents in a standard `scripts.run_arena` plan.

Keep benchmark rules, stack sizes, opponent pool, seat/deal schedule and decision limits fixed when plotting progress. Preserve the original anchor as newer models join the comparison bank; changing the pool creates a new benchmark version, not a comparable continuation of the old curve. Run new and old models on the same deals for paired uncertainty, include all seeds, and report regressions as well as gains. A direct multiplayer matchup is a useful diagnostic and can be non-transitive; it is not a universal rating.

Use validation curves for development. Freeze a new final-test protocol and opponents/data before a strength claim; repeated peeking at these curves does not turn them into independent confirmation. v0.5 remains a research release. v1.0 retains the professional qualification requirements.

## Local cost calibration

Before measuring, freeze [longer-calibration.json](../configs/holdem/longer-calibration.json): seed 997 (outside the campaign), four iterations at the exact larger training settings, one final evaluation of all four suites with 30 blocks, and a **15-minute local ceiling**. No rental and no tuning from poker outcomes. Retain the entire result and failure if it stops. Use stage times, peak RSS and artifact bytes to refine the time/storage estimate; the actual campaign seeds remain untouched.
