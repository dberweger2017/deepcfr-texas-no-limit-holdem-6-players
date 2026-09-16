# Training with a second expanded decision

## Integration

The [frozen collector study](reports/holdem-collector-branching.md) passed its variance/cost screen. This task exposes that collector through `training.sampler: "second-decision"`, keeping `first-decision` as the default. The network, fitting recipe, bootstrap, opponent policies, exploration and snapshot averaging do not change.

Replay admission verifies expansion depth along each own-action prefix and preserves inclusion probabilities, conditional targets and scheduled-root normalization. Complete checkpoints retain the configured sampler; reload rejects records whose expansion depth disagrees. Exhaustive expected-gradient checks and 4/5/6-player recovery tests cover both modes. Collection telemetry records street/position counts and roots with postflop decisions. The existing TensorBoard reader exposes these numerical fields automatically.

## Local cost pilot — declared before measurements

Run [first-decision](../configs/holdem/branching-pilot-first.json), then [second-decision](../configs/holdem/branching-pilot-second.json), sequentially on one CPU thread each. Both use seed **701**, outside the subsequent campaign; six-player 100 BB, 32 roots per role, width 32, 64 fitting steps, batch 32, capacity 4096 and the existing baseline/exploration/optimizer settings. Each runs **four iterations**, saves at two and four, and evaluates the final average against styles and random on eight paired deal blocks. These tiny evaluations exercise the pipeline; they cannot rank strength.

Each process has a **450-second cap**, 900 seconds for both. Preserve any failure and do not retry based on its poker result. Measure iteration stage times, node and record counts, coverage, peak RSS, checkpoint bytes and serialization time. Verify hashes and checkpoint/export recovery without adding training steps, within a separate 300-second verification budget. Fresh-process next-step equivalence is covered by implementation tests.

The pilot estimates early costs only: replay and the snapshot archive grow later. Use this evidence and current rental pricing to freeze the subsequent multi-seed campaign's work, checkpoints, evaluation schedule and total rental limit before any campaign seed is used. The owner has authorized up to the remaining **$6.13 Runpod CPU allowance**, including setup, storage and retrieval; GPU funding is separate. Do not treat that ceiling as a spending target. Retain all artifacts and terminate the rental after verified retrieval.

```bash
python -m scripts.train_holdem --plan configs/holdem/branching-pilot-first.json \
  --out results/branching-pilot-first
python -m scripts.train_holdem --plan configs/holdem/branching-pilot-second.json \
  --out results/branching-pilot-second
```

## Online question

Start both arms from bootstrap and compare original versus second-decision collection, keeping every other learning setting fixed. Use independent training seeds, common evaluation deals, retained checkpoints and both equal-iteration and measured-cost views. Predeclare the final paired playing comparison and uncertainty before launch. Coverage and decision behavior remain diagnostics; lower regression loss does not select a winner. Weak or inconclusive poker is a valid outcome. Invalid states, non-finite values, failed recovery or resource exhaustion stop the run.

The exact online workload follows the pilot. No production model is promoted by this integration or its resource checks.
