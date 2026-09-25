# Three-player flop local CFR pilot

## Question and fixed comparison

Does a bounded local CFR subgame solver run correctly and feasibly on the M4,
and what direction does its exploratory paired result take against corrected
rollout search? The [frozen M4 plan](../configs/blueprint/local-cfr-m4.json)
uses the saved 12M-entry six-player blueprint, SHA-256
`c6649a3e8ab061c70c5f6192d91f1f4090f15d68f4821cf3b50f2746f816c845`.
No table training or 58M checkpoint load is part of this PR.

The candidate solves only flop decisions with exactly three nonfolded players
able to act, no all-in participant, and one common pot eligibility set. It
delegates all other situations to the corrected search from PR #105. The
baseline always uses that corrected search. Both arms use the same frozen
blueprint and paired six-seat deal, opponent, and seat-rotation schedule.

The primary exploratory schedule is 128 scripted-pool rotation blocks under
validation seed `2026092601`; the secondary random schedule is 64 blocks under
validation seed `2026092602`. These seeds have not been used in #103 or #105.
The budget is fixed before either run. Report both absolute BB/100 values and
candidate-minus-baseline 95% block-clustered intervals; do not extend a run
after seeing an interim interval or call this v0.5 confirmation.

## Solver and limitations

The solver reconstructs the public beginning-of-flop state from the legal
observation. Each seat, hero included, gets a private-card range from its
public preflop actions; sampled joint holdings cannot collide with each other
or the board. The hero's actual cards condition only its own traversals and
the final decision. Earlier hero flop actions for that actual hand are fixed.
Earlier opponent flop decisions are optimized in the root traversal; a
separate traversal ensures the reached current hero information set is
updated. An observed raise absent from the abstract menu enters at its exact
raise-to size on that observed public path.

Linear-weighted external-sampling MCCFR updates separate information sets
for each player. The flop subgame ends at the next street or a second flop
raise. At that limit, each still-active player chooses among unchanged,
fold-biased, call-biased, and raise-biased blueprint continuations as a CFR
action tied across indistinguishable leaf worlds. Future boards and exact
terminal chip payoffs come from the existing engine. The actual decision uses
the final completed iteration's strategy. Weighted average strategies update
the ranges after observed flop actions; they are not the played policy.

This is a sampled 96-holding-per-seat Python pilot, not the full Pluribus
solver or an equilibrium certificate. Its blueprint has low trained
continuation coverage in #105. The 12M checkpoint is one training lineage,
and the paired arena is exploratory rather than an independent-seed strength
confirmation.

## Execution and acceptance

Run the focused tests and a small six-seat smoke first. The M4 run has a
three-hour wall guard, 10.5-GiB process RSS guard, 30-GiB free-disk guard,
and five seconds per eligible decision. It needs no paid host. Every eligible
attempt records its completed cycles, sampled nodes, leaf choices, range
updates, elapsed time, and outcome. TensorBoard flushes paired rates and
solver telemetry every eight completed blocks or after five minutes at the
next block. Interim strength charts are descriptive.

The feasibility gate requires a valid, reproducible arena, no illegal action,
at least one eligible attempt, at least 95% eligible completion, at least 32
full cycles in every completed solve, maximum measured solver time at most
5.05 seconds, and peak process RSS below 10.5 GiB. If it fails, preserve the
negative result and leave this PR draft. A positive paired interval is not a
condition for completing this pilot and does not promote the player.

```sh
python -m scripts.evaluate_blueprint_search \
  --checkpoint /Users/dberweger/Local/blueprint-04-backups/checkpoints/c6649a3e8ab061c70c5f6192d91f1f4090f15d68f4821cf3b50f2746f816c845.json.gz \
  --expected-sha256 c6649a3e8ab061c70c5f6192d91f1f4090f15d68f4821cf3b50f2746f816c845 \
  --plan configs/blueprint/local-cfr-m4.json \
  --out results/blueprint-local-cfr-m4 \
  --tensorboard
```

Retain raw run artifacts under ignored `results/`, verify their checksums,
and commit the compact result and report to this PR. No model is promoted.
