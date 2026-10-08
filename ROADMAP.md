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
| **v0.4.1** (released, `v0.4.1`, current) | The 1B-node linear-CFR opponent-sampled average, seed 2026100601; it beats v0.4.0 in a fresh direct match and passes the arena rule. [Release comparison](docs/reports/v0.4.1-release.md). v0.4.0 stays available. |
| **v0.4.x** (now) | The Pluribus recipe on heads-up 20 BB: average-policy play, a native trainer, a better training procedure or card abstraction, turn/river search, then flop search, in the order the evidence supports. Each patch must beat its predecessor in a paired arena without severe scenario regressions. |
| **v0.5** | Heads-up 100 BB, with a first benchmark against an established heads-up bot. **Still to decide (owner):** which external opponent, and the benchmark acceptance criteria. |
| **v0.6** | Three players: multiway blueprint and search. |
| **v0.7** | Four and five players: a blueprint for each table size. |
| **v0.8** | Six players at 100 BB with basic playing strength: reliable profit against the existing scripted opponent pool (no rake). Confirmed on fresh held-out deals with at least two training seeds, fixed final checkpoints, and a 95% interval above zero for each seed; evaluation size and any multiple-comparison adjustment are declared beforehand. Requires a usable train/resume/export/evaluate workflow, correct rules, legal observations and verified recovery. |
| **v0.9** | The full table: four to six players, changing lineups, unequal stacks, 20–200 BB, opponent adaptation and a decision inspector. |
| **v1.0** | Lower-end professional standard, as defined in the [readme](readme.md#what-10-means), with independent seeds and a credible professional reference. |

From v0.5 on, every milestone is confirmed on fresh held-out deals with several independent training seeds and a declared evaluation size. No release relaxes a later milestone's criteria.

### The v0.4.x path

Status of each ingredient, in dependency order. Details and full results are in the linked reports; superseded status is kept in [roadmap history](docs/roadmap-history.md#superseded-active-roadmap-status-october-57-2026).

1. **Native trainer and trainer bench — done.** The Rust trainer ([#164](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/164)) reproduces Python exactly and runs about 60–120× faster; the [turn/river bench](docs/reports/hu20-trainer-bench.md) ([#162](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/162), native in [#169](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/169)) measures training rules on fixed spots.
2. **Average-policy play — done, adopted.** The opponent-sampled average shipped as [v0.4.1](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/tag/v0.4.1) ([#165](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/165), [#176](docs/reports/hu20-v041-o-confirmation.md)). The optional zero-mass fallback had no detectable effect ([#179](docs/reports/hu20-zero-mass-fallback.md)).
3. **Training procedure — CFR+ rejected; budget scaling under confirmation.** The tested CFR+ floor ("0.4.0-shield") lowered weakness probes but lost directly to v0.4.0 ([report](docs/reports/hu20-cfr-plus.md), [floor control](docs/reports/hu20-floor-control.md)). Training the v0.4.1 recipe to 10B nodes beats v0.4.1 directly ([#185](docs/reports/hu20-o-10b.md)); it is the v0.4.2 candidate, pending [#188](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/188).
4. **Card abstraction — in progress.** Follow the [abstraction lessons](docs/reports/hu20-abstraction-lessons.md). Each step depends on the one before:
   1. validate [#163](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/163)'s equity-bucket tables on #149's turn roots ([#190](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/190), running);
   2. versioned bucket keys in the native trainer, card part only, with exact key parity;
   3. a trained bench comparison against v1 at matched visits per key, or each at its plateau;
   4. full-game confirmation: learning curves, then a direct match against the current release and the arena rule.
5. **Turn search — arena complete, adoption unresolved.** The [fixed-work arena](docs/reports/hu20-fixed-work-arena/attempt-2-closeout.md) ([#166](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/166)) shows large gains against bounded LBR and native pressure, and a regression against selective stackoff that needs diagnosis before any search-based candidate. No direct search-versus-no-search match yet.
6. **Flop search — conditional,** only if the evidence still supports it at 20 BB.
7. **v0.5 groundwork, in parallel (planned):** generalize the native trainer beyond 20 BB (stacks, bet menu, key schema, rules parity against the engine), and compact memory (average exports already load compactly, [#186](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/186)).

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

*Updated October 7, 2026.* Earlier entries are in [roadmap history](docs/roadmap-history.md#superseded-active-roadmap-status-october-57-2026).

- **M1 storage cleanup:** merged PR166 evidence/PR162 duplicate archive and inactive download caches removed; **7.573 GB reclaimed /31.022 GB free** at cleanup. Open #188/#190 dependencies remain protected. Confirmed Drive uploads are trusted for future cleanup without repeated downloads/hash audits. [Receipt and restoration](docs/artifacts/m1-vacuum-20261007.md).

- **M4 storage cleanup:** verified merged PR162 duplicate inputs and inactive caches removed under the owner's standing authorization; **6.951 GB reclaimed /71.419 GB free** at cleanup. Open #188/#190 roots/dependencies remain protected. [Receipt and restoration](docs/artifacts/m4-vacuum-20261007.md).

- **Local spectator ([#193](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/193)):** two pinned releases, paused playback and one-decision stepping, exact action probabilities and bot perspectives, retained replayable history. All 25 spectator hands /141 decisions audit; 102 focused tests and independent review pass. [Guide and verification](docs/spectator.md).
- **Shipped:** [v0.4.1](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/tag/v0.4.1) is the current release; v0.4.0 remains available.
- **v0.4.2 candidate (O at 10B nodes):** [#185](docs/reports/hu20-o-10b.md) passed the direct match and the other safeguards; its bounded-LBR safeguard was inconclusive. [#188](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/188) is resolving that uncertainty with a fresh, larger LBR sample. Publication requires the owner's explicit go.
- **Abstraction:** [#190](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/190) is validating #163's equity-bucket tables. Native bucket keys, bench training and full-game runs depend on its result.
- **Turn search:** the arena is complete; adoption is unresolved. The selective-stackoff regression needs diagnosis, and no direct search-versus-no-search comparison exists yet.
- **v0.5 groundwork:** [#196](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/196) prepares [native HU100 support](docs/native-hu100-preparation.md) with versioned 20/100-BB training, recovery and research loaders, with a staged future resource protocol. Preparation only; no campaign or worker use. The external benchmark opponent and its acceptance criteria are owner decisions not yet made.

**Release rule:** v0.4.1 stays the incumbent. v0.4.2 depends on #188's predeclared result ([protocol](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/188)) and on verifying its release package. Passing research checks does not authorize publication; only the owner's explicit go does.

## References

- Rules: [PokerStars Hold'em rules](https://www.pokerstars.com/poker/games/texas-holdem/); [Poker TDA](https://www.pokertda.com/poker-tda-rules/) is tournament-only and never defines cash procedures; [PokerKit](https://pokerkit.readthedocs.io/en/stable/) is a comparison tool, not an authority.
- Algorithms: [Deep CFR](https://proceedings.mlr.press/v97/brown19b.html), [Single Deep CFR](https://arxiv.org/abs/1901.07621), and the [Pluribus supplement](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf) for the blueprint-plus-search direction. None of their guarantees transfers automatically to six-player play.
