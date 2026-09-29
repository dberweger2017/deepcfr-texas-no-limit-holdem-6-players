# Three-player 20BB learning pilot

**Campaign completed:** [verified results](reports/tp20-m4.md),
[artifact audit](reports/tp20-m4-audit.json) and
[playable model card](tp20-model-card.md). The frozen plan below remains unchanged.

Dependent on draft #112; no merge or promotion. This is three actual players,
20BB reset each hand, 50/100 integer-chip blinds, standard deck, no rake/ante.
A fold preserves three-seat history. The K1 collector, linear iteration regret
updates, min/pot/conditional-jam menu, two-raise cap and named legacy postflop
card descriptor are unchanged. Training and inference remove only free folds.
Canonical button-relative ordered histories use `tp20-ordered-history-card-baseline-v1`;
artifacts use `holdem-tp20-blueprint-v1` and reject HU20 or other stack counts.
Final-current C is fixed before playing outcomes. No multiplayer convergence
or exploitability certificate is claimed.

## Frozen resource decision

The completed [outcome-free preflight](reports/tp20-resource-preflight.json)
and [decision](reports/tp20-resource-decision.json) freeze **20M nodes per seed**,
checkpoints at 2M/5M/10M/20M, and 4,096 confirmation blocks per primary lineup.
Measured throughput is 16,238 nodes/s; training plus export peaks at 0.543 GiB;
save/export take 3.02/2.47 seconds; evaluation including loads reaches 192.7 hands/s.
A 2x training slowdown projects 2.05 hours for three seeds, plus four-checkpoint
saving/export allowances. A 3x evaluation allowance plus reporting projects
1.90 hours for the 382,464 prescribed hands. Conservative 20M crossplay RSS
projects 8.15 GiB; 50M fails the projected entry and three-profile memory bounds;
100M also exceeds the seven-hour training allocation. These are conservative
projections, not promises of resource use. Every real limit remains active.

## Preflight procedure

`configs/blueprint/tp20-m4.json` first specifies an outcome-free 5M-node preflight,
including native evaluation through all six primary lineups, saves/exports,
entry growth and a fixed independent all-street observation set. Before main
training, replace the null budget with the largest feasible 20M/50M/100M common
budget using conservative training, memory, disk and evaluation projections.
Commit the decision and preflight digest before main training. All three seeds
start from zero. Fixed checkpoints include 20M when the budget exceeds it.
No checkpoint or extraction is selected using returns.

One heavy child runs sequentially on M4. The absolute ten-hour deadline starts
with preflight and covers every phase, including reporting. Training reserves
at least two hours before that deadline; evaluation ends fifteen minutes early.
RSS is capped at 10.5 GiB, free disk at least 8 GiB, additional system swap at
most 0.5 GiB, and reported memory free percentage at least 5%. The supervisor
samples child RSS each second and system resources every thirty seconds.
Training stop signals request cancellation during collection and do not raise
inside profile publication. A bound terminates the current child, retaining partial results and the last
completed training iteration; no rerun or relaxed plan follows automatically.

## Frozen evaluation and learning diagnostics

Each primary lineup consists of two independently randomized copies of one
#112 style: loose-passive, loose-aggressive, tight-passive, tight-aggressive,
pot-pressure, or uniform on the new training menu. Mixed secondary lineups
are declared in the configuration. Odd blocks reverse opponent order, paired
identically across arms; all opponents have separate random streams.

The same deal/button is played at each of three hero positions. Confirmation
compares each final seed and same-game uniform. Within each independent
rotation block, average each seed's candidate-minus-control differences;
then weight the six lineups equally. The primary 95% interval uses independent
block variation within each fixed lineup, with a Welch/Satterthwaite t interval
for the stratified mean. Per-seed and per-lineup effects, absolute returns,
secondary mixed lineups and crossplay are reported separately. Proposed
confirmation is 4,096 blocks per primary lineup, frozen after resource preflight.

Development uses disjoint deals at every fixed current-policy checkpoint.
Independent observation density is collected once from same-game uniform,
before training, and includes all streets. Report decision-weighted trained
coverage and update-count quantiles on that fixed set and each policy's own
reached observations. Training new entries, traverser visits, repeated
contributions, outer iterations, actual completed nodes, overshoot and discarded
work remain separate quantities.

Crossplay compares the earliest and final hero checkpoints against the two
other seeds' frozen final checkpoints, with paired deals and balanced opponent
order/hero seats. Aggregate selfplay chip conservation is not evidence of
strength. All hand attempts, legal failures, schedules and hashes are retained.

## Commands

```sh
python -m scripts.run_tp20_campaign --plan configs/blueprint/tp20-m4.json \
  --root results/tp20-m4-20260928 --stage preflight
# After committing the outcome-free resource decision:
python -m scripts.run_tp20_campaign --plan configs/blueprint/tp20-m4.json \
  --root results/tp20-m4-20260928 --stage main
```

Large artifacts remain on M4; compact reports/manifests go into the draft PR.
A terminal human-versus-two-trained-bots entry point and exact retrieval
instructions accompany the artifacts. Demo seeds are chosen by fixed order.
No six-player experiment, rental or model promotion follows automatically.
