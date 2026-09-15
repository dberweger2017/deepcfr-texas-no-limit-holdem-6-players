# Preserve the strategies we actually averaged

## Decision

Implement an exact, snapshot-based average over the small-game advantage
policies before doing more learning-rate or network-size experiments. Start with
an inference archive and correctness checks; integrate training recovery and a
prospective learning campaign in a separate task. Full-history and variable-seat
Hold’em encoding can proceed independently. This decision spends no rental budget
and does not change the failed readiness result or promote a model.

The [fresh confirmation](../reports/neural-readiness.md) identifies why this is
worth implementing. Kuhn 431's exact/replay averages pass while its neural policy
fails. Leduc 433's exact average narrowly passes at 0.148821, its replay average
fails at 0.153602, and its fitted policy reaches 0.160619. Removing just the final
network fit would leave that replay gap. The small margin in the underlying
Leduc strategy remains a convergence concern.

## Alternatives

| Approach | What it addresses | Cost or limitation | Decision |
| --- | --- | --- | --- |
| More strategy fitting or larger networks | Approximation of the retained replay average | Does not remove replay sampling error; another tuning screen would need new confirmation | Stop this line for now |
| Larger strategy reservoir | Replay variance | Still requires fitting; changes memory/compute without isolating the extraction problem | Keep as a later comparison |
| Exact played-average table | Both strategy extraction gaps on our reference games | Enumerating Hold’em information sets is infeasible | Retain as the independent test oracle |
| All advantage-policy snapshots | Avoids strategy replay and a separately fitted average network | Storage grows with iterations; full action-distribution queries are more expensive | Implement and verify |

[Single Deep CFR, sections 5.1–5.3](https://arxiv.org/pdf/1901.07621) describes two
ways to use stored iteration policies: sample one with linear iteration weights
at the start of a hand and retain it throughout, or query a distribution by
weighting each policy with the player's own reach probability. The latter uses
only that player's previous decision observations and actions. Resampling a
snapshot at every decision or simply averaging predictions does not implement
the same strategy. The paper's two-player results are not a six-player guarantee.

## Repository-specific alignment

One iteration here contains player 0's update followed by player 1's update.
Strategy samples belong to the *opponent* during each traversal phase. Therefore
the policies contributing weight `t` to the existing played-average diagnostic
are:

- Player 1's network **before** either update in iteration `t`.
- Player 0's network **after** its update in iteration `t`.

The first player-1 entry is the initial zero-output network, including the
existing highest-legal-prediction fallback. The final fitted player-1 network
has not contributed to that iteration's average. Saving both networks only after
the iteration would produce the wrong comparison.

Capture one complete pair after a successful iteration; keep the earlier
player-1 network while the solver advances. Fits replace network objects. Copy
weights into the archive so subsequent edits cannot change a published policy.
Failed iterations cannot publish a partial pair. Require complete history from
iteration one; the previous campaign saved only sparse training snapshots, so it
cannot retrospectively supply this archive. No result will be fabricated from
its final weights or diagnostic table.

## Query contract

For snapshot `t`, let `r_t` be the product of probabilities it assigns to this
player's own earlier actions, evaluated on the observations available at those
moments. Return `sum(t * r_t * policy_t) / sum(t * r_t)`. Do not include opponent
or chance reach. Use a uniform legal distribution if every snapshot assigns zero
own reach; this only specifies behavior off the mixture's reachable paths.

The small-game API accepts the current `InformationSet` and the complete ordered
sequence of the same player's earlier `(InformationSet, action)` records in this
hand. Validate their history prefixes, ownership, own card and board-reveal
stages. Inference does not accept an engine state or a game tree. Its path
validation is specific to Kuhn/Leduc's alternating actions and two-round format;
it is not the later Hold’em observation adapter.

A separate hand-policy factory samples exactly once from weights `1..T` and
returns an object that keeps the chosen network and its own action random stream.
Its distribution is conditional on that selection; use the full mixture API for
exact best-response evaluation and probability inspection. Create separate hand
policies for each player and each new hand.

## Cost model and limits

With this repository's 48-input, two-hidden-layer, three-output network, width
`h` uses `h*h + 53*h + 3` float32 parameters. At width 64 that is 7,491 parameters,
29,964 bytes per snapshot. Two players over 480 iterations require 28,765,440
parameter bytes (27.43 MiB), before serialization and Python/Torch overhead. The
width-128 average network alone uses 92,684 parameter bytes; compare the archive
against both that network and the strategy replay it could replace, not only
against the network.

A sampled hand uses one network per acting player. An arbitrary full-distribution
query processes every retained snapshot at the current and previous own decisions:
work grows as `T * (own decisions + 1)`. The first implementation keeps the small
archive in CPU memory and batches a query's decision path within each network.
Disk-backed storage, device batching and reach caching are later optimizations.
Do not truncate or reservoir-sample the snapshot archive without treating that as
a new approximation with its own evaluation.

These are exact parameter counts and algorithmic costs, not a Hold’em hardware
quote. The eventual history encoder and shared-seat model will change the counts.
No large-game memory, latency or strength claim follows from this small archive.

## Acceptance for the first implementation

- A two-policy counterexample distinguishes own-reach weighting from naive
  averaging. Later observations do not inject a newly revealed board into an
  earlier decision.
- Exhaustive Kuhn/Leduc information-set checks equate the mixture's realization
  probabilities with the weighted mixture of its constituent policies, including
  zero-reach paths.
- Short deterministic runs match the existing exact played average after every
  recorded iteration, with the correct alternating-update alignment. Recording
  changes neither fitted networks, replay, traversal randomness nor fitting RNG.
- Published policies survive later training/model mutation. Partial or missing
  iteration history is rejected.
- One sampled hand retains its chosen policy. Ownership and incomplete/incorrect
  own-history records fail explicitly; hidden-state inputs are rejected.
- Hash-pinned inference exports round-trip in a fresh process and reject invalid
  shapes, non-finite tensors, inconsistent counts and altered bytes.

These are implementation checks on tiny local fixtures, not another readiness
campaign. Existing baselines, frozen configurations and report outcomes stay
intact. Subsequent work must integrate complete archive recovery, provenance and
reporting before training a new candidate. A new convergence test needs fresh
seeds and a committed protocol; it may still fail because advantage learning is
unchanged.

## Implementation and validation

The first implementation is [the snapshot-average module](../../src/solver/neural/average.py).
Its [20 focused tests](../../tests/test_snapshot_average.py) pass; the complete
repository suite passes **318 tests**. The checks enumerate all 300 Kuhn/Leduc
information sets, include a hand-calculated own-reach counterexample, and compare
four tiny training iterations per game against an otherwise identical recorder-
free solver. The averaged policies agree with the existing exact diagnostic
within `1e-6`; replay entries, fitting records, traversal state, global random
state and current policies remain unchanged. Final fitted baseline policies also
match. These tiny fixtures use seed 101 and are not a new convergence campaign.

The first full-suite attempt encountered a source-fingerprint mismatch because
the module was edited during the run. The guard correctly rejected it. The
complete suite was rerun after source edits stopped and passed; no source check
was weakened.

### Recording and inference

Create `StrategyArchive(game, hidden)` alongside a fresh `DeepCFR` solver, then
call `record_iteration(solver, archive, deadline=...)` instead of `solver.step()`.
The recorder calls the existing step and appends the correctly aligned pair.
It is bound to that live solver, requires consecutive history and refuses a
partial failed iteration. The existing runner is not yet wired to this recorder.
The wrapper still collects the old strategy replay for comparison; removing that
work is part of the later training integration.

`archive.policy(player)` freezes the current snapshot sequence. It has two query
paths:

```python
# Current observation plus this player's earlier decisions in the same hand.
probabilities = policy.distribution(observation, own_decisions)

# Create once per hand and keep this object for every decision in that hand.
hand_policy = policy.sample_hand(seed=hand_seed)
action = hand_policy.choose_action(observation)
```

`own_decisions` contains `OwnDecision(information, action)` values in observed
order. Its earlier observations retain the board visible at the time; using the
current board in an earlier round is rejected. Observations and legal-action
sets come from the rules engine's public interface. This validator checks the
query's ownership, structure and matching history; it does not replace the rules
engine's legal-action generation.

`save_archive(archive, path)` atomically publishes a new immutable file and returns
its SHA-256. `load_archive(path, digest)` verifies the entire file before
weights-only CPU loading. Format, feature encoding, averaging/alignment modes,
iteration counts, player pairs, tensor shapes/dtypes and finite weights are
checked. A fresh-process round trip passes. Existing files cannot be overwritten
through the export function.

These are **inference archives**, not complete training checkpoints. Loading one
does not provide replay, RNG, fitting provenance or a safe continuation of its
original solver. The recorder rejects extending a loaded archive or attaching it
to a different live solver. Archive-aware recovery, source/configuration binding
and campaign reporting must land together before a new learning run.

## What follows

The next PR should implement complete public-history and variable-seat Hold’em
encoding, with seat/suit symmetry and observation-boundary checks. That advances
milestone 4 without another small-game sweep. The remaining milestone 3 task is
archive-aware training/recovery and a prospective readiness protocol after that
integration is validated. Keep the original failed confirmation unchanged, its
seeds consumed, and substantial training gated. No rental was used for this PR.
