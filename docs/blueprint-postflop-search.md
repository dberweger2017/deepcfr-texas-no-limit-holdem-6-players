# Postflop search comparison after checkpoint 0.4

## Question

Does bounded, range-aware postflop reasoning improve the **same finished checkpoint-0.4 blueprint** against random and scripted six-player opponents? The current player consults a sparse table and plays a uniform abstract menu whenever its key is absent. The checkpoint-0.4 campaign must finish under its original protocol before this comparison starts. Search changes the playing policy; it does not resume or retrain the table.

The design draws from the [Pluribus technical supplement](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf): update private-hand beliefs from public actions, include observed off-tree bet sizes, and use blueprint-based fold/call/raise-biased continuations after a search depth limit. This implementation is a **bounded sampled rollout search**, not Pluribus's multiplayer MCCFR subgame solver. It optimizes the next action against sampled belief worlds and continuation profiles; it does not solve a multiplayer equilibrium or guarantee resistance to exploitation.

## Playing policy

At every postflop decision, sample legal opponent two-card candidates that exclude the player's known cards and board. Weight each candidate by the fixed blueprint's likelihood of that opponent's observed actions; an off-menu raise receives weight from nearby blueprint raise sizes and retains a small nonzero floor. Independently sampled seat ranges are joined into collision-free deals. Replaying **the exact observed actions and amounts** from the public hand start must reproduce the current observation before a simulated branch is used. The player never reads the arena's hidden state or deal seed.

For each abstract legal root action, search the current betting round under the blueprint policy in common sampled worlds. On a multiway flop, the depth limit may occur after a second raise. At a leaf, sample one of four fixed continuation profiles per player: unchanged blueprint or a fivefold fold, passive action, or raise probability bias. Roll out those profiles to exact chip settlement. A river branch reaches settlement directly. The decision is the action with the highest sampled mean payoff. If the time limit expires or sampled ranges cannot form a compatible deal, play the unchanged blueprint action sampled from the same action RNG stream. Any public replay mismatch, illegal action, or blueprint schema mismatch invalidates the comparison.

The search retains the blueprint's raise menu. It does not add arbitrary new raise sizes for the player. It **does** preserve an opponent's actual off-menu wager in the reconstructed game and posterior update, avoiding a fictitious nearby wager. Unlike Pluribus, this search roots at the current decision and reweights public ranges from the fixed blueprint rather than retaining a solved strategy from earlier decisions in the street. Leaf profiles are sampled rather than optimized by CFR, and the private ranges use 96 candidates per seat rather than all 1,326 combinations. These approximations are explicit risks for both playing strength and latency.

## Fixed comparison

[The comparison plan](../configs/blueprint/postflop-search-comparison.json) uses the same final checkpoint hash for both arms. The candidate adds search; the baseline is direct regret-matched blueprint play. Deals, seats, opponent draws, and action seeds are paired. The random benchmark has 512 blocks (3,072 candidate hands); the scripted-pool benchmark has 256 blocks (1,536 candidate hands). Both use new validation seeds, separate from checkpoint-0.4 validation and its sealed final random test. The original sealed test must run **once** on the chosen 0.4 checkpoint, as already committed; these search validation schedules cannot be relabeled as that test.

Report candidate and baseline BB/100, the paired difference and confidence interval, completed hands, invalid actions, search attempts/completions/fallbacks, both overall candidate-decision and search-only latency median/95th/max, and peak process RSS. Retain a compact row for every hand. A failed or incomplete run stays failed or incomplete; it is not a win claim. Favor search only if the scripted-pool direction improves without an unacceptable random-play regression and the measured completion rate and latency make live play practical. The result may instead show that the sparse blueprint or approximate beliefs limit this search.

## Execution boundary

The plan caps the whole comparison at four hours, process RSS at 50 GiB, and each postflop decision at one second. Use one 64-GB memory-optimized CPU pod if loading the final checkpoint stays below the RSS cap; otherwise size a separate larger-memory rental from measured load before spending. At the September 24 quoted $0.447/hour for the 64-GB pod, four hours is about $1.79 CPU plus storage. Do not silently extend the checkpoint-0.4 training segment or delay its final artifact retrieval and pod cleanup. Keep the comparison on its own budget and output directory. Verify and retain checkpoint and result hashes before terminating its rental.

```sh
python -m scripts.evaluate_blueprint_search \
  --checkpoint FINAL_CHECKPOINT \
  --expected-sha256 FINAL_CHECKPOINT_SHA256 \
  --plan configs/blueprint/postflop-search-comparison.json \
  --out NEW_COMPARISON_DIRECTORY
```

The implementation and plan can be reviewed before 0.4 finishes. The comparison results and decision belong in this PR after the final checkpoint has been fixed. No other training recipe or 0.4 artifact is changed.

## Preliminary inference check

A local inference-only check loaded the earlier 5,834,622-entry blueprint checkpoint in 33.63 seconds, with 4.62 GiB peak process RSS. One six-player flop decision under the fixed 12-world, 96-candidate, one-second configuration completed search in 0.294 seconds without fallback. This single hand on a much smaller table checks the execution path; it does not predict completion rate or playing strength for the final 0.4 checkpoint. The paired comparison above is still pending.
