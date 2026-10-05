# Roadmap

## Goal

Build a strong no-limit Texas Hold'em agent that plays by the selected table rules and sees exactly what a human in its seat would see. Six-handed play is the main target; four- and five-handed tables, changing lineups and unequal stacks are part of the design.

Playing strength is the measure of progress. A finished training run, a lower loss or a win against random opponents is useful evidence, but none of them shows the agent plays good poker.

Anything can be replaced: engine integration, algorithm, models, scripts, file formats. Keep useful regression scenarios and experiment records.

The full history of plans, protocols and results is in [docs/roadmap-history.md](docs/roadmap-history.md) and [docs/research-history.md](docs/research-history.md).

## Release milestones

Patch releases improve the current heads-up 20 BB game; minor releases move to deeper stacks and more players. The [readme](readme.md#release-plan) has the short version.

| Release | What it means |
|---|---|
| **v0.4** (released, `v0.4.0`) | Rebuilt research preview: a local heads-up 20 BB table, the hash-verified B100M policy, replayable sessions and an honest evidence index. Not a strength claim. |
| **v0.4.x** (now) | The Pluribus recipe on heads-up 20 BB: average-policy play, a native trainer, a better training procedure or card abstraction, turn/river search, then flop search, in the order the evidence supports. Each patch must beat its predecessor in a paired arena without severe scenario regressions. |
| **v0.5** | Heads-up 100 BB, with a first benchmark against an established heads-up bot. |
| **v0.6** | Three players: multiway blueprint and search. |
| **v0.7** | Four and five players: a blueprint for each table size. |
| **v0.8** | Six players at 100 BB with basic playing strength: reliable profit against the existing scripted opponent pool (no rake). Confirmed on fresh held-out deals with at least two training seeds, fixed final checkpoints, and a 95% interval above zero for each seed; evaluation size and any multiple-comparison adjustment are declared beforehand. Requires a usable train/resume/export/evaluate workflow, correct rules, legal observations and verified recovery. |
| **v0.9** | The full table: four to six players, changing lineups, unequal stacks, 20–200 BB, opponent adaptation and a decision inspector. |
| **v1.0** | Lower-end professional standard, as defined in the [readme](readme.md#what-10-means), with independent seeds and a credible professional reference. |

From v0.5 on, every milestone is confirmed on fresh held-out deals with several independent training seeds and a declared evaluation size. No release relaxes a later milestone's criteria.

### The v0.4.x path

1. **Trainer bench** ([#162](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/162)) — done. On fixed turn spots, CFR on the v1 abstraction stays near blueprint quality even with dense training; the standard opponent-sampled average does clearly better than the production average, but not enough. The [report](docs/reports/hu20-trainer-bench.md) classifies a poor v1 CFR fixed point at 3M under the frozen rule; this budget does not prove asymptotic convergence. Next, compare abstraction-aware training and better card abstraction under a prospectively fixed protocol.
2. **Native trainer** ([#164](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/164)) — done. Rust, about 120× faster than Python, and reproduces Python's runs exactly.
3. **v0.4.1: average-policy play** ([#165](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/165)) — [arena completed](docs/reports/hu20-v041-arena.md); the predeclared release rule is not met. Native pressure is inconclusive, not a measured regression. No follow-up arena is proposed or run; no release or tag.
4. **Turn search** ([#166](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/166)) — river validation and Linux pilot passed; the paid arena quote awaits owner approval.
5. **Better training procedure** — native trainer options measured on #162's bench, one at a time: discounting and regret floors, training against an unabstracted opponent (CFR-BR style), pruning.
6. **Equity buckets** ([#163](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/163)) — tables built; validate on #149's turn spots before adding them as an alternative key to the native trainer, with v1 kept as the reference.
7. **Flop search** — last, if the evidence still supports it at 20 BB.
8. **v0.5 groundwork**, in parallel: generalize the native trainer beyond 20 BB (stacks, bet menu, key schema, rules parity against the engine), compact memory and multi-core pods, and choose the external benchmark bot (owner decision).

**Proposed v0.4.x exit (owner to confirm):** each ingredient above has been adopted or rejected on recorded evidence, and the latest patch beats v0.4.0 in the paired arena.

## Ground rules

- **Information:** the agent receives an immutable observation of what its seat legitimately knows — its cards, the board, every observed action and amount, legal action bounds, revealed cards and its own history with each player. It never sees other players' hidden cards, the deck, seeds, future outcomes or evaluator data. Training may use simulated payoffs and counterfactual branches as targets, never as inputs; search samples hidden worlds from legal beliefs, never from the real deal. Details: [observations](docs/observations.md).
- **Rules:** cash-game no-limit Hold'em under the profile in [docs/rules.md](docs/rules.md). Tournament rules, ICM, straddles, rake and multiple runouts are separate scope decisions, never silent approximations.
- **Architecture:** rules engine, session manager, agent (observation in, legal action out), solver/trainer, evaluation arena, search/adaptation — small explicit boundaries. Native code where profiling justifies it.
- **Evaluation:** paired deals, rotated seats, uncertainty over independent blocks, and fresh confirmation data. Approximate best responses (LBR) are weakness probes, not exploitability certificates.
- **Strength gate:** a change of default model or a claim of stronger play needs a predeclared paired comparison whose 95% interval clears zero on the primary benchmark, with scenario regression limits written down in advance and both statistical and practical effect sizes reported. Don't re-inspect one test set until something passes; multi-seed claims include training-seed variation.
- **Experiments:** each campaign declares its hypothesis, opponents, seeds, maximum cost and runtime, evaluation schedule, stopping conditions and decision rule before the main run. Numerical failures, information leaks, broken accounting, invalid play and resource exhaustion stop a run. Weak or inconclusive results are kept as evidence, never a reason to keep extending the budget. A positive poker result is not required before longer, budgeted training.

## How we work

- **Branches and PRs:** one PR per coherent task, on a `feature/` branch with a short descriptive name. Keep `main` usable. Commit and push each completed subtask; commit messages describe the actual change.
- **Writing:** PR descriptions state the problem, the resulting behavior, the design choice and the evidence, in plain language. Comment development decisions, constraints and non-obvious invariants; don't narrate code. Remove dead paths instead of keeping compatibility scaffolding.
- **Code:** small modules, typed boundaries, explicit data ownership, deterministic examples and tests of real behavior. Keep logging out of game transitions and reports out of training logic.
- **Merging:** review the diff against its goal, run the relevant checks, resolve findings and record validation. Never bypass branch protection or failing required checks. Rules, information access and regret math deserve independent review; unresolved correctness questions block dependent training.
- **Autonomy:** routine implementation, fixes, refactors, tests, docs and short local experiments proceed without asking, and routine PRs merge when their checks pass with no open findings. Treat the owner's exploratory ideas as hypotheses to assess.
- **Ask the owner** for: game scope changes, significant algorithm or compute tradeoffs, long or paid compute (with a quote), destructive removal of valuable artifacts, and public strength or release commitments. Bring a recommendation, alternatives and evidence. Owner-approved PR comments are binding.
- **Artifacts:** index large untracked evidence in [RESULTS_INDEX.md](RESULTS_INDEX.md) and archive it to the project's Google Drive folder at cleanup, with manifests, partials and retrieval provenance. Delete local originals only after verified upload and owner authorization; never delete inside synced folders or force offloading. Never commit credentials or large checkpoints.
- **This roadmap:** update the v0.4.x path and Current position when work lands, keeping each entry short and moving detail into reports. This document doesn't schedule unattended work.

## Current position

*October 5, 2026*

- **v0.4.1 arena complete; release rule not met** ([#165](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/165), [report](docs/reports/hu20-v041-arena.md)): all 165,888 hands / 850,106 actions independently replay; all 52 contrasts independently recompute. O−R LBR improves **+30.83 [14.03, 47.63] BB/100**. Native pressure is **inconclusive, not a measured regression**: **+0.44 [−18.47, 19.34]**, near zero with an interval too wide to establish the required lower bound > −10. The severe-scenario check passes. All panels, lineage/position splits and street fallback/zero-mass counts are retained; [archives](RESULTS_INDEX.md) have confirmed uploads and originals remain.
- **Turn search ready for its arena** ([#166](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/166)): 32/32 river validations passed; the solver is bit-identical on Linux pods. Waiting for owner approval of the revised quote and RTX 3090 stock.
- **Trainer bench finished** ([#162](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/162)): all 40 roots and both 3M folds completed. The [report](docs/reports/hu20-trainer-bench.md) gives production/opponent-sampled Q=0.9514/0.7308: both meet the frozen poor-v1-fixed-point classification. Closeout and raw-evidence staging are complete; cloud upload verification remains pending in [RESULTS_INDEX](RESULTS_INDEX.md).
- **Native trainer merged** ([#164](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/164)): exact Python parity, including all three #116 lineages at 20M and 100M nodes.
- **Equity buckets built** ([#163](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/163)): K=50/200 flop, turn and river tables, awaiting validation.

**Release gate:** #165’s native-pressure safeguard is not established, so the frozen release rule remains not met. Shipping stays an owner decision; no release, tag or follow-up arena. The separate turn-search arena quote still awaits its owner decision.

## References

- Rules: [PokerStars Hold'em rules](https://www.pokerstars.com/poker/games/texas-holdem/); [Poker TDA](https://www.pokertda.com/poker-tda-rules/) is tournament-only and never defines cash procedures; [PokerKit](https://pokerkit.readthedocs.io/en/stable/) is a comparison tool, not an authority.
- Algorithms: [Deep CFR](https://proceedings.mlr.press/v97/brown19b.html), [Single Deep CFR](https://arxiv.org/abs/1901.07621), and the [Pluribus supplement](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf) for the blueprint-plus-search direction. None of their guarantees transfers automatically to six-player play.
