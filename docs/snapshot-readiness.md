# Snapshot training and fresh readiness

**Completed September 16, 2026:** all eight seeds in each game pass. See the
[full results and artifact audit](reports/snapshot-readiness.md). The protocol
below is retained as declared before the run; these confirmation seeds are now
consumed, not a fresh test set for another experiment.

## Decision

Train the existing small-game Deep CFR solver with a complete archive of the
advantage policies that actually generated play. Evaluate the exported archive,
without fitting an additional strategy network. This follows the
[snapshot decision](decisions/snapshot-average.md); it does not revise the
[failed strategy-network confirmation](reports/neural-readiness.md).

The question is whether this deployable average passes the original readiness
limits across fresh training seeds. Passing would close the small-game gate,
not establish strong multiplayer Hold'em play or qualify a release.

## Recovery and evaluation

`average: "snapshots"` selects archive recording in the experiment runner.
Each iteration stores player 1 before its update and player 0 after its update,
matching the alternating traversal policies. Linear iteration weights and the
player's own action reach produce the behavioral average. Evaluation batches
public information sets through the exported networks; it never substitutes the
solver's exact accumulated average. That diagnostic is checked independently
against the export and cannot supply policy probabilities.

A training checkpoint contains the entire archive, both current networks,
reservoir contents and admission counts, random streams, iteration progress,
fit reports, source hashes and environment. Only a completed iteration can be
saved. A failed archive publication invalidates the live solver; recovery uses
the preceding immutable checkpoint. An inference export cannot resume training.

`checkpoint_interval` saves recovery state independently of evaluation. The new
campaign saves every 20 Kuhn iterations and 60 Leduc iterations, but scores only
the final policy. The default network-average path remains available to reproduce
historical experiments. Its strategy-related configuration fields are retained
in this plan for comparison, but are unused by snapshot training.

## Frozen experiment

The executable plan is
[snapshot-readiness-v1.json](../configs/solver/snapshot-readiness-v1.json).
The runner pins its SHA-256 and requires a clean committed checkout. Commit the
protocol before running any reserved seeds; use that revision for the campaign.

Run **503, 509, 521, 523, 541, 547, 557, 563** from scratch in each game. These
are eight replicates per game. Collection, fitting, reservoir admission and
initialization retain separate deterministic streams.

| Setting | Kuhn | Leduc |
| --- | ---: | ---: |
| Iterations | 100 | 480 |
| Traversals per player per iteration | 1,024 | 1,024 |
| Advantage width | 64 | 64 |
| Advantage updates per player per iteration | 1,000 | 4,000 |
| Advantage learning rate | 0.001 | 0.001 |
| Batch size | 256 | 256 |
| Reservoir capacity | 100,000 | 200,000 |
| Final exploitability ceiling | 0.03 | 0.15 |
| Final absolute value-error ceiling | 0.03 | 0.10 |
| Training deadline per seed | 800 seconds | 5,100 seconds |

Both limits must pass for **all eight seeds in both games**. Use the existing
hash-pinned tabular reference and independent equilibrium checks. No earlier
checkpoint selection, added seeds, parameter sweep or threshold relaxation.
Missing jobs, invalid outputs or exhausted budgets make the campaign inconclusive;
completed policies outside a limit fail readiness. Preserve every attempt.

There is no paired strategy-network refit: the preceding experiment already
isolated its approximation loss, and this campaign tests readiness of the
replacement. It cannot claim paired improvement on different seeds.

## Resources and execution

Use one CPU pod with at least 16 vCPUs and 32 GB RAM, eight single-threaded workers,
and enough temporary disk for all checkpoints (20 GB or more). Start Leduc first
and fill free worker slots with Kuhn. Check actual CPU allocation and available
memory over SSH before launch. The previous campaign took about 54 minutes on
similar hardware; allow roughly an hour, with timing uncertainty.

| Limit | Ceiling |
| --- | ---: |
| Entire rental, including storage and transfer | $1.50 |
| All-in recurring quote | $0.65/hour |
| Rental time, including provisioning and setup | 2 hours |
| Runner time | 110 minutes |
| Individual worker | 100 minutes |
| Retrieval and shutdown reserve | 10 minutes |

This uses the existing $10 CPU authorization, with $7.92 conservatively remaining
before this run. It does not spend the separate GPU budget. The runner subtracts
setup time and the shutdown reserve from its deadline. Failed or timed-out workers
are killed and reaped. **Stopping a process does not stop rental billing.**

Before launch, arrange a provider stop deadline and completion follow-up. Retrieve
and verify reports, manifests, logs and checkpoint hashes, then terminate the pod
and remove its temporary storage. Stop at the cap even if the statistical result
is incomplete. Record the quote, provisioning/termination times, conservative
cost and verified artifact location. Do not rent a second host or retry failed
seeds without a new documented decision.

Smoke checks use seeds 101, 103 and 107, two iterations and small networks:

```sh
python -m scripts.run_snapshot_readiness --smoke --out results/snapshot-smoke
```

After tests pass, commit/push the protocol, verify the rental quote and shutdown
plan, then run on that exact checkout:

```sh
python -m scripts.run_snapshot_readiness --run \
  --hourly-usd "$SNAPSHOT_HOURLY_USD" \
  --rental-started-at "$SNAPSHOT_RENTAL_STARTED_AT" \
  --out results/snapshot-readiness
```

The start time must include its timezone. Smoke runs report `smoke_completed`,
never readiness. Publish every seed's final exploitability, value error, archive
hash, source/environment provenance and failures. If readiness passes, proceed to
a meaningful Hold'em baseline; if it fails, retain the result and make a bounded
algorithm decision rather than extending this run.

## Implementation checks before confirmation

All **513 repository tests pass**. New checks compare batched export evaluation
with public policy queries and the independent played-average diagnostic in both
games; verify replay/RNG/archive equality after recovery; reproduce identical
export bytes in fresh processes; reject inference-only, altered and mismatched
checkpoints; and recover after archive-publication failure. The six-job smoke
completed across both games using non-confirmation seeds. No reserved seeds were
used by these checks.
