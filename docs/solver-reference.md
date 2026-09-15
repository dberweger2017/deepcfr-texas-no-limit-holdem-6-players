# Small-game CFR reference

This is the first part of roadmap milestone 3: validate regret minimization on games we can solve. It provides separate Kuhn/Leduc rules, simultaneous tabular CFR, external-sampling regret updates, exact best responses, and an independent sequence-form equilibrium solver. It does not change the Hold'em engine or the legacy neural trainer.

## Games and information

Both games are two-player, zero-sum, use a one-unit ante per player, and measure **net profit in ante units per hand**. Player 0 acts first in each betting round. There are no blinds, rake, stack limits, all-ins, or seating changes in these toy games.

| Rule | Kuhn (`kuhn-ante1-v1`) | Leduc (`leduc-2bet-ante1-v1`) |
| --- | --- | --- |
| Deck | Three ordered ranks, one card each | Three ordered ranks, two physical cards each |
| Private cards | One per player | One per player |
| Public cards | None | One card after the first betting round |
| Betting | One round; one-unit bet; no raise after a bet | Two rounds; bet/raise increments 2 then 4 |
| Round cap | One bet | Two bets total: opening bet and one raise |
| Showdown | Higher rank wins | Pair with the board wins; otherwise higher private rank; equal hands split |
| Maximum absolute payoff | 2 | 13 |

Leduc's rules follow the [University of Alberta description, section 5.1.1](https://poker.cs.ualberta.ca/publications/schauenberg.msc.pdf). Explicitly naming the two-bet cap matters because other variants use different limits.

`State` is privileged simulator data. `information_set()` returns only the acting player, their card rank, the public board rank, complete public betting history with round boundaries, and legal actions. It contains no opposing private card, future board, deal seed, or hidden node identifier. Leduc's physical duplicate cards are retained for chance probabilities; strategically irrelevant suits are omitted from the information key. This symmetry reduction is specific to Leduc and is not a decision about Hold'em features.

Chance deals cards without replacement, including the public card only after the first round closes. Calling matches the outstanding wager; checking twice or calling closes a round. Folding returns the winner's opponent contribution as net profit, so unmatched wagers are not counted as winnings.

The complete compiled trees have 58 nodes/12 information sets for Kuhn and 9,457 nodes/288 information sets for Leduc. Compilation checks that every node in an information set has the same prior sequence of the player's own information/action pairs (perfect recall) and the same depth. The depth restriction keeps the evaluator simple and is an explicit limit of this reference compiler.

## CFR update convention

Both players use one frozen strategy profile per iteration. Their regret changes are applied together afterward. Regret matching normalizes positive cumulative regrets over legal actions; if none are positive, it uses the uniform legal distribution. Negative regrets are retained. There is no CFR+, discounting, linear iteration weighting, replay prioritization, or neural fitting here.

For full-tree CFR, each decision's action-minus-policy continuation value is multiplied by **chance reach × opponent reach**, excluding the updating player's own reach. Contributions from all hidden nodes in the same information set are summed. This follows the counterfactual regret construction in [Zinkevich et al., 2007](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf).

The reported strategy is the own-reach-weighted average:

```
average(I, action) = sum_t own_reach_t(I) * policy_t(I, action)
                    / sum_t own_reach_t(I)
```

Perfect recall makes own reach equal across an information set. We accumulate it once per information set per iteration, without multiplying by the number of hidden deals. An information set with zero cumulative own reach uses the uniform legal fallback. The average includes the profile used for each iteration's updates; it does not include the newly generated profile after the final update.

## External sampling

An iteration performs one traversal for each updating player. Each traversal samples chance and opponent actions from their distributions, enumerates every legal action of the updating player, and accumulates sampled action-minus-policy values. The two traversals use the same frozen profile and a solver-owned Python random generator. Global Python/NumPy random states are untouched.

Sampling already supplies chance/opponent reach in expectation. Multiplying the sampled regret by those reaches again would bias it. The test suite enumerates every possible sampling trace in Kuhn under nonuniform and partly deterministic profiles and checks that the expected update equals full CFR. See [Lanctot et al., 2009](https://papers.nips.cc/paper_files/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf) for external sampling.

**Averaging remains an exact full-tree pass**, identical to the full solver. This intentionally separates the regret-estimator check from sampled averaging. It is not a claim of an entirely sampled implementation or a template for large-game throughput. [OpenSpiel also distinguishes full from sampled averaging](https://raw.githubusercontent.com/google-deepmind/open_spiel/master/open_spiel/python/algorithms/external_sampling_mccfr.py); our implementation uses simultaneous frozen-profile updates and counts each information set once.

## Exact evaluation and the independent oracle

A best response must choose the same action in every hidden state belonging to one information set. The evaluator works backward through the tree, sums action values over those states using chance/opponent reach, chooses one maximizing action, then applies it to all of them. It evaluates actions even where the supplied responding policy had zero reach. It never chooses a different action after looking at an opponent's hidden card. This matches the information-set requirement illustrated by [OpenSpiel's best-response implementation](https://raw.githubusercontent.com/google-deepmind/open_spiel/master/open_spiel/python/algorithms/best_response.py).

For a profile with first-player value `v`, report:

```
gain_0         = best_response_value_0 - v
gain_1         = best_response_value_1 + v
NashConv       = gain_0 + gain_1
exploitability = NashConv / 2
```

All quantities are in ante units per hand, not BB/100. “Exact” means exhaustive tree evaluation without sampled rollouts or approximate responding agents; arithmetic still uses float64. Tiny negative gains from numerical rounding are clamped to zero; materially negative gains fail the evaluation.

The independent oracle uses [sequence-form linear programming](https://doi.org/10.1006/game.1996.0050) via SciPy/HiGHS. Each player's realization weights satisfy flow conservation at their information sets. Terminal utilities and chance probabilities form the payoff matrix. We solve the maximizing and minimizing programs separately, convert realization weights into a behavioral policy, and require their values, residuals, and exact best-response evaluation to agree. It shares the game tree with CFR, so separate rule/utility tests remain necessary. Kuhn additionally has the analytic first-player value **−1/18**; the specified Leduc program gives approximately **−0.085606424078**.

Tests compare Kuhn best response with all 64 deterministic policies for each player, verify hand-calculated regret updates and reach-weighted averages, and check both equilibrium profiles and perturbed exploitable profiles. This is stronger evidence than a decreasing training loss, but still limited to these games.

## Run and retain evidence

From the repository root, using the usual installed dependencies:

```bash
python -m scripts.check_solver --plan configs/solver/smoke.json --out results/solver-smoke
python -m scripts.check_solver --reproduce results/solver-smoke --out results/solver-replay
python -m scripts.check_solver --plan configs/solver/reference-v1.json --out results/tabular-reference
```

The smoke only checks execution and reproduction; its loose thresholds are not convergence criteria. The [declared acceptance protocol and results](reports/tabular-validation.md) cover the full local reference campaign, all seeds, and final-iteration thresholds.

Output directories must be new. The bundle contains:

- `manifest.json`: resolved plan/hash, source-file hashes, git revision and dirty status, Python/platform/NumPy/SciPy versions, game profiles, update/averaging conventions, and units.
- `report.json`: independent oracle results, every scheduled evaluation, final acceptance status, seed, completed/requested iterations, strategy hashes, and wall times.
- `<game>-<method>-<seed>-strategy.json`: average policy probabilities keyed by explicit information sets, with only legal action entries.

The runner checks time limits between iterations and enforces both per-run and total budgets of at most 840 seconds. Oracle compilation/evaluation and the final artifact write may finish after the last deadline check; these are small bounded operations in the supported games, not hard operating-system termination. A timed-out run retains its completed evaluations and strategy and cannot pass. Later jobs receive an explicit unstarted status if the total budget is exhausted. Exceptions mark the report as an error; abrupt process termination can leave an incomplete report, which is not a successful experiment.

Exact reproduction checks the plan hash, solver/CLI source fingerprint, recorded environment, and protocol, then compares all deterministic report fields and strategy bytes. Timings are excluded. Changed code/environments fail rather than silently becoming a reproduction claim. Keep full bundles under ignored `results/` or retained artifact storage; commit compact evidence and plans. Strategy exports are inference tables, **not resumable optimizer checkpoints**. In-memory chunked training preserves the same state and random stream; saved/resumed neural training is a later task.

## Next boundary

This delivers the tabular reference only. The next PR should implement the small-game neural Deep CFR baseline: correctly weighted samples, reservoir storage, regret fitting, and strategy averaging, checked against these exact references. Multiple neural seeds and reliable resume remain acceptance requirements before carrying the rewrite into six-player no-limit Hold'em. None of the two-player results gives a multiplayer equilibrium guarantee.
