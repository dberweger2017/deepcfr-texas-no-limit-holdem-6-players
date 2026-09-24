# Blueprint to search: next changes

The checkpoint-0.4 training recipe remains frozen until its final checkpoint and tests are complete. This work prepares the next player without claiming that search is already present or that the current policy is strong. The [Pluribus supplement](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf) motivates range-aware, depth-limited re-solving, but our implementation should be judged by our own fixed arena and latency measurements.

## First PR: locate the fallback

The current blueprint policy chooses uniformly when its exact information key has no entry. More training can fill rare keys but cannot generate a public history containing an action outside the trainer's action menu. Replay each observed action from its public decision event and classify whether it was in that menu. At each candidate decision, report four counts by street: trained on an in-tree history, trained after an off-tree action, fallback on an in-tree history, and fallback after an off-tree action. Record whether the first off-tree action was a raise size or the per-street raise cap. The latter counts how often coverage requires a change in the action model; the in-tree fallback count measures sparse table visitation. Neither count is itself a strength result.

Use [check_blueprint_offtree.py](../scripts/check_blueprint_offtree.py) with a hash-pinned checkpoint and the separate [diagnostic schedule](../configs/blueprint/offtree-diagnostic.json). It reads public decisions only and never opens the sealed checkpoint-0.4 test schedule. Its 64 blocks per opponent set are for architecture diagnosis, not a win-rate claim. Run it after the 0.4 final checkpoint is fixed, on a machine that can load that checkpoint without extending the current bounded rental solely for this diagnostic.

## Following PRs

1. **Public belief state.** Represent each player's possible private hands, remove known cards, and update action likelihoods using only observations that player could have seen at the time. Test exact enumeration on tiny games and avoid treating sampled hidden deals as policy inputs. Preserve card correlations or document the approximation.
2. **Bounded river re-solver.** Start with river decisions using legal current bets, explicit ranges and a fixed compute/latency budget. A timed-out search returns the baseline blueprint action. Compare the same frozen checkpoint with and without search on paired fresh deals, including scripted opponents and off-tree bets.
3. **Depth-limited earlier streets.** Add continuation strategy choices and turn, then flop search only after the river implementation passes legality, information-boundary, and latency checks. Decide from measured strength whether to revise the blueprint abstraction, train longer, or expand search.

Do not call a stronger abstract player a Pluribus reproduction. The paper used a much richer blueprint and online search; both our model quality and deployment latency must be measured independently.
