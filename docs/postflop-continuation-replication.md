# Fixed-work postflop continuation replication

This dependent PR starts from the exact head of draft PR #109. Its only
algorithmic candidate change is `postflop_replicates=4`; the control uses the
existing traversal with `postflop_replicates=1`. Both retain the same six-seat
training game, key schema, abstract action menu, regret matching, linear
iteration weight, terminal payoff, and current-policy extraction. The
[Pluribus paper and technical supplement](https://www.science.org/doi/10.1126/science.aay2400),
supplement pp. 14–18, describe external-sampling blueprint CFR and pruning. The K=4
conditional replication here is **our proposed extension**, not an algorithm
attributed to Pluribus. It does not solve a flop subgame to convergence.

## Sampling and estimator contract

An ordinary root traversal samples a complete hidden deal and opponent
actions. On each traverser branch's **first entry into the flop**, if the
traverser is neither folded nor all-in, the candidate retains that sampled
prefix. It includes the exact public action history and flop, every private
hand in the training simulation (including folded players), stack and pot
state, and legal actions. Four continuation worlds rebuild the native hand
from that prefix. The engine deals private cards in two rounds, clockwise
starting left of the button. The unrevealed 37-card suffix is independently
shuffled for each world; opponent actions each use a distinct derived random
stream. The rebuilt event history must equal the retained prefix exactly.
No second replication layer occurs on those continuation paths.

For fixed prefix `h` and the unchanged outer-iteration policy `sigma`, each
continuation is sampled from the ordinary conditional chance/opponent law.
The returned value and every downstream information-set increment are:

```
U_K(h)       = (U_1(h) + ... + U_K(h)) / K
Delta_K(I,a) = (Delta_1(I,a) + ... + Delta_K(I,a)) / K
```

An unvisited information set contributes zero in that replicate. Ancestors
receive `U_K`, and downstream regrets and the existing average accumulator
are scaled by `1/K` before the complete outer iteration publishes. The
existing iteration multiplier remains applied once at each traverser
information set. A failed or resource-interrupted iteration publishes no
regret, average, visit, or iteration change. No replicate is kept as an
unweighted extra pass.

Conditional linearity gives the same expected value and regret increment as
K=1 under a fixed profile, for the declared game. This says nothing about
equilibrium convergence, the quality of the abstraction or the estimator's
benefit **per unit compute**. Tests exercise the production deck replay and
aggregator, including a nonuniform finite hidden-information reference with
an intermittently visited information set. A pinned K=1 checkpoint fixture
checks byte identity with #109.

`node.visits` continues to count **raw traverser node visits**, including
each replicated visit. It is not a count of independent outer updates. The
new telemetry separately records completed outer iterations, sampled flop
prefixes, continuation samples, raw traverser visits, normalized update mass,
unique contributing information sets per iteration, and per-street counts.
Conditional value and regret variability are measured across each four-draw
batch; reconstruction time and replayed action count are separate from
traversal nodes. A sidecar counts how many additional completed outer
iterations contributed to each key, without rewriting legacy checkpoint
node fields.

## Comparison gates

The same immutable 5.83M checkpoint is loaded independently for each of
three paired continuation seeds and both K values. The M4 resource preflight
completed 96 K4 steps with 1,781,022 nodes in 175.86 seconds, including a
21.52-second checkpoint load, and 4.98 GiB peak RSS. A separate 48-hand
evaluation preflight took 22.05 seconds, including load, at 4.96 GiB peak.
These measurements preceded the frozen main protocol in
[`postflop-replication-m4.json`](../configs/blueprint/postflop-replication-m4.json):
20 million additional traversal nodes per run, milestones at 5/10/15/20
million, at most 250,000 nodes and 300 seconds per complete outer iteration,
9.5 million entries, and at most 250,000 nodes of completed-iteration
overshoot. The memory ceiling is 10.5 GiB RSS and the hard campaign limit,
including loads, saves and evaluation, is ten hours. The runner reserves
the final 15 minutes for checkpoint saving and reporting.
Every replicated node is charged; both arms stop only after a complete
iteration crosses the work threshold. The observed overshoot, discarded
work, throughput and memory are retained. The six outputs have explicit
parent/output hashes, button-zero table, key schema, sampler version and
source revision in their lineage manifests. Descendants can use the
compatible lookup only with a matching verified manifest; the parent hash
cannot stand in for a changed output.

Primary play comparisons on fresh paired six-rotation blocks are K4 minus
K1 and K4 minus `U_safe`, both canonical-safe. Parent canonical-safe and the
existing `tight_aggressive` hero are references. The current regret-matched
policy is used throughout. The aggregate over three saved seed pairs first
averages differences **within a block**, then estimates a block-level
interval. The random-opponent suite is a secondary regression check.
Scripted and random coverage will be separated. More visits or lower
conditional variance alone will not count as a playing-strength gain.

The fresh playing schedule uses 4,096 scripted-pool and 1,024 random-opponent
rotation blocks, with seeds 2026100101/2026100102, respectively. A separate
independent `U_safe` observation set uses 128 scripted and 32 random blocks
with seeds 2026100201/2026100202; it is frozen before candidate training.
The two scripted primary claims use two-sided 97.5% block-clustered intervals,
a Bonferroni adjustment across K4–K1 and K4–`U_safe`. Each seed pair is also
reported separately; the three contrasts are averaged within each evaluation
block before forming the aggregate interval. Other intervals are descriptive.
The frozen schedule is not extended based on observed profit. No player is
promoted from this experiment.
