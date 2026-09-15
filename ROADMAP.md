# Roadmap

## Goal

Build a strong no-limit Texas Hold'em agent that plays by the selected table rules and has exactly the game information available to a human in its seat. Six-handed play is the main target. Four- and five-handed tables, changing lineups, and unequal stacks are part of the design from the beginning.

Playing strength is the measure of progress. A successful training run, a lower loss, or a win against random opponents is useful evidence, but none establishes that the agent plays good poker.

We are willing to replace the engine integration, learning algorithm, models, training scripts, and checkpoint formats. Backwards compatibility is not a requirement. Preserve useful regression scenarios and experiment records; retire old code as its replacement passes the relevant checks.

## Initial scope

These defaults let development proceed without waiting on every implementation decision. Changes to the target game or resource budget need an explicit decision.

| Area | Initial target |
| --- | --- |
| Game | Cash-game no-limit Texas Hold'em, table stakes, one board |
| Players | Six seats; train and evaluate four-, five-, and six-handed games separately |
| Stacks | Start with a fixed 100 BB benchmark, then add unequal stacks and 20–200 BB scenarios |
| Costs | No rake or antes in the first research benchmark; add a named, versioned rake profile before making claims about raked games |
| Seating | Arrivals, departures, sit-outs, and top-ups between hands; a hand's participants remain fixed until settlement |
| Heads-up | Correct action order and settlement when a hand becomes heads-up; full two- and three-handed table benchmarks follow the primary target |
| Observation | Structured game events available to that player, including complete observed betting history and legitimately revealed cards |
| Deployment | A headless agent API and local play/replay interface |
| Compute | Short local checks first, then rented Vast.ai hardware; an RTX 5090 is a candidate to compare after profiling |

“Official rules” needs a precise profile. Cash rooms and tournaments differ on procedures such as missed blinds and showing cards. The first implementation PR will write `docs/rules.md`, cite its sources, and specify every supported house-rule choice. Tournament payouts, ICM, straddles, multiple runouts, and live audiovisual tells are separate scope decisions. Do not silently approximate them.

## Information available to the agent

The engine needs hidden cards and a deck to simulate a hand. The playing agent must receive a separate, immutable observation, never the engine state itself.

The observation includes:

- The agent's hole cards, public board, and the street on which each board card appeared.
- Button, blinds, occupied seats, participants, public player identities, stacks, committed chips, pot breakdown, and action order.
- Every observed action with its actor, street, and exact amount, including the agent's own actions.
- Exact legal action bounds and the public information needed to derive them.
- Cards actually exposed under the selected rules, including permitted showdown disclosures; mucked and folded cards stay hidden.
- Previously observed hands for the same player, with a documented retention policy. A new occupant does not inherit the old occupant's history.

The agent must not receive other players' unrevealed cards, undealt cards, deck order, simulator seeds, future outcomes, hidden opponent model identities, or privileged evaluator data. Multiple copies at one table do not share private cards or private observations.

Training may use simulated terminal payoffs and counterfactual branches to form learning targets. These are not policy inputs. Search samples possible hidden worlds from beliefs built from legal observations; it must not inspect the actual hidden deal.

**Acceptance checks:** worlds with the same player-visible history produce identical observations and action distributions when the agent's own random stream is fixed. Changing unrevealed cards or deck order must not change its current decision. Counterfactual branches cannot mutate live opponent history. Event replays reconstruct the information available at each decision, including observed prior hands.

## Architecture

Keep the boundaries small and explicit:

1. **Rules engine:** integer chip accounting, legal actions, state transitions, cards, and settlement.
2. **Session manager:** seating, public identities, stack changes between hands, and observed hand histories.
3. **Agent:** observation in, legal action out. No dependence on simulator internals.
4. **Solver and trainer:** traversal, regret estimation, strategy averaging, replay storage, and model fitting.
5. **Evaluation arena:** frozen candidates, opponents, schedules, results, and uncertainty estimates.
6. **Search and adaptation:** optional improvements to a validated baseline, each evaluated independently.

Python is the starting point for orchestration and model training. Choose a native engine or move traversal work to Rust/C++ where correctness and profiling justify it. The existing `pokers` implementation is already native; rewriting it in another language alone is not a performance strategy.

The likely destination is an offline strategy with range-aware search during play. Deep CFR is a baseline to validate, not a constraint on every future experiment. Compare alternatives against the same game, information boundary, and resource budget.

## Delivery plan

The items below are proposed PR-sized changes, not existing PR numbers. Split an item further when engine work or algorithm review needs it. Each milestone has a condition for proceeding; elapsed training time alone never satisfies it.

### 1. Establish the rules and a trustworthy environment

**Engine foundation delivered:** [rules profile](docs/rules.md) and [fork audit](docs/engine-audit.md). The repaired Rust engine supplies integer chips, legal betting, side pots, and reference tests. The observation boundary is delivered in [PR #42](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/42); the [session lifecycle](docs/sessions.md) is implemented with a moving button, big-blind entry, and identity-owned history.

- [x] **Specify the game and observation contract.** Add `docs/rules.md`, typed public observations and actions, integer chip units, explicit raise-to semantics, and a compact decision record for the initial rule profile. Add CI for the full test suite, installation, and a headless smoke run.
- [x] **Validate betting and settlement.** Port useful regression scenarios, add property-based tests, and compare supported hands with an independent reference implementation. Audit the pinned engine before deciding whether to repair or replace it. Reproduce the raise from 2 to 10: the next full minimum raise-to is 18, not 20 or 11.
- [x] **Implement the session lifecycle.** Handle four to six occupied seats, button/blind movement, sit-outs, player replacement, unequal stacks, and top-ups between hands. Keep absent, folded, and all-in players distinct. Record replayable public events.

Cover full and short all-in raises, reopening action, cumulative short raises under the chosen profile, side pots, ties, odd chips, uncalled bets, and players leaving after a hand. Terminal rewards must reconcile with chip movement. With rake enabled, reconcile player losses with house collection rather than expecting a zero-sum player result.

**Exit condition:** deterministic scenarios and generated legal hands settle correctly, independent-engine disagreements are resolved against the rules, and observation leakage checks pass. No repair of poker state in logging code and no silent substitution of a different action in benchmark play. An invalid transition fails the run with a replayable trace.

### 2. Build the evaluation arena before serious training

- [x] **Add reproducible match schedules and reports.** Separate random streams for deals, action sampling, opponent selection, and training. Use disjoint training, validation, and final-test schedules. Save hand-level outcomes and a manifest containing the commit, rules, configuration, model hashes, seeds, and environment.
- [x] **Add opponents and benchmark suites.** Include simple legal agents with different tendencies, training-seed provenance, and frozen historical candidates. Independent training-seed campaigns remain required before model promotion. Keep a stable evaluation pool separate from any training league. Port a legacy checkpoint only if the adapter preserves legal observations and actions; label such comparisons explicitly.

Report BB/100, confidence intervals, completed hands, invalid actions, wall time, and action latency. Pair deals and rotate seat assignments; compute uncertainty over the appropriate independent deal or session blocks. Correlated replays are not independent samples. Stratify results by table size, stack depth, and opponent lineup.

**Exit condition:** the same manifest reproduces the same schedule and results on the supported deterministic execution path. The arena can distinguish an intentionally weakened agent from its baseline over a declared sample budget and reports an inconclusive result when evidence is insufficient. No failed hands disappear from the denominator without invalidating or explicitly qualifying the run.

### 3. Prove the learning algorithm on games we can solve

- [x] **Implement a tabular reference.** Use two-player Kuhn and Leduc poker with exact best-response evaluation. Check utilities, regret updates, strategy averaging, and external sampling against small exact calculations.
- [x] **Implement a faithful Deep CFR baseline.** Keep the same small-game interface. Use justified sampling and iteration weights, reservoir memories, documented network fitting, and correct average-strategy estimation. Keep regrets in a consistent utility scale. Establish fitting quality before introducing replay prioritization or other heuristics.
- [ ] **Compare neural and tabular results.** Run multiple seeds, test replay replacement and resume behavior, and measure exploitability rather than loss alone. Consider Single Deep CFR as a separately benchmarked alternative after this baseline works.

**Exit condition:** the reference approaches known game values, and the neural implementation reaches predeclared exploitability tolerances on the small games across multiple seeds. Record the numerical thresholds and budgets in benchmark configuration before the acceptance runs. Explain any difference from the reference algorithm.

These small-game checks establish algorithm correctness. They do not imply a Nash-equilibrium guarantee for six-player poker.

### 4. Train a credible no-limit baseline

- [ ] **Represent complete decisions.** Encode full public action history, card reveal stages, relative seats, player masks, amounts in BB, pot-relative amounts, and effective stacks. Test seat rotation and suit permutation. Begin with a modest model and measure whether more capacity helps.
- [ ] **Learn several bet sizes.** Use street-appropriate candidate raises, including minimum raises, overbets, and exact all-ins. Deduplicate candidates after legal bounds are applied. Record the executed action. Each candidate needs its own value/regret signal; do not train a sizing head to copy its own untested prediction.
- [ ] **Train all player roles in self-play.** Keep the strategy profile consistent during each collection phase. Use current regret-matched strategies where the algorithm requires them, and implement averaging separately. Share parameters across seats only with a correct representation and sampling scheme. Keep frozen-opponent exploitation experiments distinct from the base solver.
- [ ] **Add genuine resume and a first training report.** Save optimizer state, replay contents or a recoverable reference, reservoir counters, iteration state, configurations, and random streams. Separate collection, fitting, evaluation, and save schedules. Checkpoint writes must be atomic. Export a smaller inference artifact separately.

Start with six-handed 100 BB. Once the pipeline behaves correctly, explicitly train and evaluate four- and five-handed games and the declared unequal-stack distribution. A six-handed hand in which two players fold is not a substitute for a four-handed starting game.

**Exit condition:** all correctness checks pass; small-game checks remain healthy; and the candidate has a reproducible report against the fixed arena with multiple training seeds. Promotion to the default model requires the strength gate below. If results are inconclusive or self-play regresses, investigate before scaling the run.

### 5. Scale only after a measured pilot

- [ ] **Profile and batch collection.** Measure time in engine transitions, encoding, network inference, replay sampling, and fitting. Batch inference across independent traversals and use compact arrays. Add workers with explicit seed ownership and a fixed policy version per collection phase.
- [ ] **Run the Vast.ai campaign.** Package the same configuration for local and remote execution. Compare current rental offers, run a short hardware pilot, estimate throughput and storage, then choose the training budget. An RTX 5090 is the owner's initial candidate, not a committed hardware choice. Compare multiple seeds and one change at a time against the baseline.

**Exit condition:** parallel and serial collection agree on controlled reference cases, the speedup is measured, restart works, and the larger run has a complete report. GPU utilization is a diagnostic; it is not a measure of poker strength.

### 6. Add range-aware search

- [ ] **Track ranges from public history.** Respect card removal and update beliefs using observed actions. Validate beliefs on small games where exact enumeration is possible. Keep each player's inference separate from actual hidden cards.
- [ ] **Solve constrained late-street situations.** Start with river and small turn subgames. Evaluate counterfactual values over ranges, including the agent's possible hands, and model continuation choices beyond a search depth limit. Avoid solving sampled deals as independent perfect-information games.
- [ ] **Expand search where it helps.** Include observed opponent bet sizes even when absent from the offline action set. Add a time budget, interruption behavior, and a reliable baseline fallback. Expand to earlier streets and multiway pots only after controlled comparisons succeed.

**Exit condition:** search improves paired arena results at a stated latency budget without information leaks or severe scenario regressions. Multiway beliefs must respect card correlations, either exactly or through a documented approximation. Heads-up resolving guarantees must not be claimed for an unproved multiplayer extension.

### 7. Add opponent adaptation and make the agent inspectable

- [ ] **Learn opponent behavior from observations.** Predict actions and sizes from preceding public context. Maintain separate player histories, uncertainty, cold-start priors, and a policy for behavior changes. Do not train every opponent feature toward the same noisy hand reward.
- [ ] **Evaluate adaptation against unfamiliar opponents.** Test new players, replacements, changing styles, and opponents deliberately trying to exploit the agent. Compare against the same agent with adaptation disabled. Keep the default strategy available when evidence is weak.
- [ ] **Add a decision inspector and headless release.** Show the exact observation, candidate actions, policy probabilities, estimated values, ranges, search time, and model version. Label estimates and uncertainty. Replay explanations must use information available at the original decision.

**Exit condition:** adaptation improves held-out session results, including cold starts, without exposing private data or causing unacceptable losses against the wider pool. The released agent runs the four-to-six-player lifecycle suite and ships with a reproducible model report.

## Training and model promotion

| Run | Purpose | Resource policy |
| --- | --- | --- |
| Local smoke | Check transitions, losses, checkpoint/restart, and end-to-end execution | Short bounded runs; initially at most 15 minutes per job, one job at a time |
| Small-game benchmark | Verify learning against exact solutions over several seeds | Local runs within the agreed machine budget; longer jobs need a declared limit |
| Hardware pilot | Measure traversal, fitting, memory, storage, and evaluation cost | Short local or workstation run before choosing the campaign size |
| Baseline campaign | Establish a reproducible 100 BB result and compare seeds | Explicit runtime, storage, and resource limits; resume on interruption |
| Improvement campaign | Compare one algorithm, model, sizing, or search change | Same arena and a comparable compute budget; retain the baseline |

Vast.ai is the planned rental provider. Before renting, present a cost/performance comparison and obtain the owner's campaign budget. No specific rental or spending amount has been approved. The comparison must include:

- Measured local time spent in traversal, inference, fitting, and evaluation, and projected CPU, RAM, VRAM, and storage needs.
- An RTX 5090 offer and suitable alternatives available at the time, including their host CPU allocation, memory, storage, availability, and interruption terms.
- Total expected cost: provisioning, the pilot, training, evaluation, storage, transfers, and a stated interruption allowance. Quote current offers rather than treating today's prices as permanent.
- Cost per complete collection-and-fitting cycle and projected cost to reach a fixed validation target. Raw GPU throughput or hourly price alone is insufficient.
- A short paid pilot with its own spending cap. Use measured throughput to revise the campaign estimate before committing the remaining budget.
- A maximum campaign spend and runtime, checkpoint/export schedule, and shutdown procedure that also accounts for any storage left behind.

Obtain access through the owner's chosen account workflow when needed. Never commit credentials or large checkpoint files. Commit configurations and compact reports; store model artifacts separately with hashes and retrieval instructions.

Each campaign must declare its hypothesis, opponents, seeds, maximum cost/runtime, evaluation schedule, stopping conditions, and promotion rule before the main run. Stop on invalid states, information leakage, non-finite values, broken accounting, or resource exhaustion. A disappointing result is a result to preserve, not a reason to keep extending the budget.

**Default strength gate:** use a predeclared paired comparison whose 95% confidence interval for improvement clears zero on the primary benchmark, with scenario-specific regression limits written down in advance. Report both statistical and practical effect size. Use fresh confirmation data or a valid sequential method when repeatedly evaluating candidates; do not repeatedly inspect one fixed test set until a result happens to pass. Multi-seed claims must include training-seed variation.

Infrastructure and correctness PRs can merge without a strength improvement if they pass their own acceptance checks. Changing the default model or claiming stronger play requires a model report. Approximate best-response agents are useful weakness probes, but they do not certify exact exploitability in full multiplayer no-limit Hold'em.

## How we work

- Work on a branch and open one PR per coherent roadmap task. A task may contain several small commits. Keep `main` usable. A rewrite can proceed in small pieces behind explicit interfaces; avoid a long-lived branch that replaces everything at once.
- Start branch names with `feature/` and use short descriptive names such as `feature/poker-rules`, `feature/player-observations`, and `feature/reservoir-memory`. Commit messages describe the actual change: `Track the last full raise`, `Keep hidden cards out of agent observations`, or `Resume training with replay state`.
- Commit and push after each completed subtask so progress is backed up and reviewable. Each commit should represent a meaningful change; do not accumulate an entire task locally or create noise commits for individual edits.
- Write direct PR descriptions: the problem, resulting behavior, relevant design choice, and evidence. Use ordinary names and plain language. Add comments for development decisions, constraints, or non-obvious invariants; let clear code express routine Python operations. Do not narrate each line. Remove dead paths and duplication instead of adding compatibility scaffolding we do not need.
- Prefer small modules, typed boundaries, explicit data ownership, deterministic examples, and tests of real behavior. Keep logging separate from game transitions, and benchmark reports separate from training logic.
- Before merging, review the diff against its milestone, run the relevant checks, resolve review findings, and record validation. Never bypass branch protection or failing required checks. An independent review is particularly useful for rules, information access, and regret mathematics; unresolved correctness questions block dependent training.
- Routine implementation choices, fixes, refactors, tests, documentation, and short local experiments can proceed autonomously. Routine PRs can be merged when their acceptance checks pass and there are no unresolved findings. Summarize meaningful results and decisions rather than asking for approval at every step.
- Ask the owner for changes to game scope, significant algorithm/compute tradeoffs after a comparison, long or paid compute budgets, destructive removal of valuable artifacts, and public strength/release commitments. Bring a recommendation, alternatives, and the evidence needed to decide.
- Update this roadmap in each milestone PR. Link merged PRs and training reports, state the current blocker if any, and keep the next task unambiguous. Further work runs in active development sessions or an explicitly configured follow-up; this document alone does not schedule unattended work.

## Current position

- **Plan established:** September 2026, based on the review of `af1593a`.
- **Engine foundation completed:** [fork PR #1](https://github.com/dberweger2017/pokers/pull/1) and [bot PR #40](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/40) are merged. The bot pins the corrected engine. See the [engine audit](docs/engine-audit.md) for rule-validation evidence and limits. The session lifecycle completes the remaining environment gate in milestone 1.
- **Observation boundary delivered:** [PR #42](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/42) adds immutable observations, complete public events, owner-specific history, explicit actions, and leakage checks. Existing agent calls and human card displays use this boundary. See [the interface contract](docs/observations.md).
- **Session lifecycle delivered:** [PR #43](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/43). The [session contract](docs/sessions.md) documents changing lineups, bankrolls, physical seats, blind admission, public replay, and private-history ownership. Headless session policies support variable participant counts; existing neural inputs remain fixed-size.
- **Evaluation scheduling delivered:** [PR #44](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/44). The [arena](docs/evaluation.md) saves paired fixed-stack/session schedules, disjoint split seed ranges, source/engine/environment fingerprints, public hand records, strict failures, and block-level confidence intervals. The [predeclared control](docs/reports/arena-validation.md) checks sensitivity and exact reproduction. The following benchmark delivery adds styles and archived models; competitive trained opponents remain future work.
- **Remaining Hold'em gaps:** limited neural history encoding, fixed-size seat inputs, self-imitation sizing targets, deviations from validated CFR sampling/averaging, and incomplete training reproducibility and competitive benchmark coverage. The separate small-game solver validates the replacement learning approach before it reaches Hold'em.
- **Benchmark infrastructure delivered:** [PR #45](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/45) adds varied legal styles, separate named training/evaluation pools, four-/five-/six-player stack-depth and session plans, and frozen standard checkpoint adapters with verified bytes and public-only decisions. See the [benchmark contract](docs/benchmarks.md) and [declared validation/results](docs/reports/benchmark-validation.md). Historical training seeds are unknown; multi-seed learned-opponent campaigns remain open.
- **Tabular reference delivered:** [PR #46](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/46). The [solver contract](docs/solver-reference.md) documents Kuhn/Leduc, exact information-set best responses, independent sequence-form equilibrium checks, simultaneous CFR, and external-sampling regrets with exact averaging. The [declared validation report](docs/reports/tabular-validation.md) records convergence budgets and all training seeds. This completes the first task in milestone 3, not the neural-learning gate.
- **Small-game neural baseline delivered:** [PR #47](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/47). The [implementation contract](docs/neural-cfr.md) covers alternating Deep CFR, separate advantage/strategy reservoirs, linear-weighted losses, fresh advantage fitting, public-only features, fitting-error diagnostics, and verified inference exports. The [declared checks and pilots](docs/reports/neural-validation.md) establish implementation/fitting evidence; a completed single-seed pilot is not the convergence gate.
- **Next task:** compare neural and tabular results across predeclared seeds, add complete replay/random-state persistence and interrupted-resume tests, and set exploitability tolerances and budgets before the acceptance campaign. Use the fitting/sampling diagnostics to resolve failures before the no-limit learning rewrite or substantial training. Evaluate Single Deep CFR separately if the evidence warrants it.
- **Compute direction:** Vast.ai rental, with an RTX 5090 as an initial candidate. Compare cost/performance after profiling; select an offer and campaign budget with the owner before renting.
- **Owner input needed before large training:** pilot and campaign spending ceilings, permitted unattended runtime, storage retention, and account access method. Confirm a different game profile if the cash-game defaults above do not match the intended table.

## References

- [PokerStars Hold'em rules](https://www.pokerstars.com/poker/games/texas-holdem/) and [betting examples](https://www.pokerstars.com/poker/learn/lesson/texas-holdem-rules/) provide a starting point for the cash-game betting specification. Resolve procedural details explicitly in `docs/rules.md`.
- [Poker TDA rules](https://www.pokertda.com/poker-tda-rules/) are a tournament reference; they must not silently define cash-game procedures.
- [PokerKit documentation](https://pokerkit.readthedocs.io/en/stable/) describes a candidate independent reference implementation. It is a comparison tool, not the authority when implementations disagree.
- [Deep CFR](https://proceedings.mlr.press/v97/brown19b.html) is the initial neural solver reference. Its two-player convergence result is not a six-player guarantee.
- [Single Deep CFR](https://arxiv.org/abs/1901.07621) is an alternative to compare after the first baseline.
- [Pluribus supplementary material](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf) informs the multiplayer blueprint and online-search direction. Adopting that direction does not establish equivalent playing strength.
