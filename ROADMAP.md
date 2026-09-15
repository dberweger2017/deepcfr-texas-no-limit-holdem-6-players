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

“Official rules” needs a precise profile. Cash rooms and tournaments differ on procedures such as missed blinds and showing cards. The implemented `docs/rules.md` cites its sources and specifies the supported house-rule choices. Tournament payouts, ICM, straddles, multiple runouts, and live audiovisual tells are separate scope decisions. Do not silently approximate them.

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

### 3. Validate the learning algorithm on games we can solve

- [x] **Implement a tabular reference.** Use two-player Kuhn and Leduc poker with exact best-response evaluation. Check utilities, regret updates, strategy averaging, and external sampling against small exact calculations.
- [x] **Implement a faithful Deep CFR baseline.** Keep the same small-game interface. Use justified sampling and iteration weights, reservoir memories, documented network fitting, and correct average-strategy estimation. Keep regrets in a consistent utility scale. Establish fitting quality before introducing replay prioritization or other heuristics.
- [x] **Diagnose the fitting gap.** [PR #52](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/52) completes the 144-fit comparison. Decayed minibatches and exact-gradient controls pass all absolute limits, but the complete exploration screen selects no candidate because of a paired regression. Preserve that outcome.
- [x] **Integrate strategy learning-rate decay.** [PR #54](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/54) adds an explicit strategy-only cosine schedule, constant-mode equivalence, complete configuration/provenance, and deterministic recovery checks. Advantage fitting and collection are unchanged. The [confirmation protocol](docs/neural-readiness.md) and executable plan were frozen before the subsequent run.
- [x] **Run one fresh end-to-end confirmation.** [PR #55](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/55) records the [failed result](docs/reports/neural-readiness.md): 7/8 seeds pass in each game; milestone 3 remains open. The run executed the [frozen protocol](docs/neural-readiness.md): all eight reserved seeds in both games, final-only checks, and one paired fixed-rate control per Leduc seed, under a $2.50 / two-hour rental ceiling. Every seed must meet the original absolute limits: Kuhn exploitability/value error ≤0.03/0.03; Leduc ≤0.15/0.10. The prior exploration's 0.005 paired-regression veto is explicitly not carried into this prospective readiness test; regressions remain reported, and the old screen remains failed. Passing readiness does not establish uniform improvement or promote a Hold'em model.

- [x] **Choose and implement snapshot averaging.** The [decision and validation](docs/decisions/snapshot-average.md) select complete advantage-policy snapshots. The first implementation provides own-reach-weighted distribution queries, one-snapshot-per-hand play, correct alternating-update capture and verified inference exports. All 20 new tests and all 318 repository tests pass. This is implementation evidence, not a new readiness pass.
- [ ] **Integrate snapshot-aware training and recovery.** Persist the archive with replay/RNG/provenance, compare interrupted and uninterrupted execution, report the actual exported average, and define a fresh prospective readiness protocol before another learning campaign. The first archive API deliberately rejects resuming training from an inference export.

**Bounded follow-up:** the failed confirmation prompted the [snapshot-average decision](docs/decisions/snapshot-average.md). All final exact played averages passed, but Leduc 433 had little margin. Correct averaging does not fix inaccurate advantage learning. No automatic extra seeds, steps or widths. Independent Hold'em engineering proceeds next; substantial training remains gated.

**Exit condition:** the reference approaches known game values, and the neural implementation reaches predeclared exploitability tolerances on the small games across multiple seeds. Record the numerical thresholds and budgets in benchmark configuration before the acceptance runs. Explain any difference from the reference algorithm.

These small-game checks provide implementation and learning evidence on solvable games. Their model sizes and fitting settings need not transfer to Hold’em, and they do not imply a Nash-equilibrium guarantee for six-player poker.

### 4. Build and train a credible no-limit baseline

**Engineering can start while milestone 3 remains open.** Work on decision encoding, action generation, trainer interfaces, and recovery has independent acceptance checks. Substantial self-play and architecture campaigns require both the small-game readiness pass and a correct no-limit pipeline. Do not run a capacity sweep on the legacy trainer.

- [ ] **Represent complete decisions.** Encode full public action history, card reveal stages, relative seats, player masks, amounts in BB, pot-relative amounts, and effective stacks. Test seat rotation and suit permutation. Begin with a modest model and measure whether more capacity helps.
- [ ] **Learn several bet sizes.** Use street-appropriate candidate raises, including minimum raises, overbets, and exact all-ins. Deduplicate candidates after legal bounds are applied. Record the executed action. Each candidate needs its own value/regret signal; do not train a sizing head to copy its own untested prediction.
- [ ] **Train all player roles in self-play.** Keep the strategy profile consistent during each collection phase. Use current regret-matched strategies where the algorithm requires them, and implement averaging separately. Share parameters across seats only with a correct representation and sampling scheme. Keep frozen-opponent exploitation experiments distinct from the base solver.
- [ ] **Add genuine resume and a first training report.** Save optimizer state, replay contents or a recoverable reference, reservoir counters, iteration state, configurations, and random streams. Separate collection, fitting, evaluation, and save schedules. Checkpoint writes must be atomic. Export a smaller inference artifact separately.

Start with six-handed 100 BB. Once the pipeline behaves correctly, explicitly train and evaluate four- and five-handed games and the declared unequal-stack distribution. A six-handed hand in which two players fold is not a substitute for a four-handed starting game.

**Exit condition:** all correctness checks pass; small-game checks remain healthy; and the candidate has a reproducible report against the fixed arena with multiple training seeds. Promotion to the default model requires the strength gate below. If results are inconclusive or self-play regresses, investigate before scaling the run.

### 5. Scale only after a measured pilot

- [ ] **Profile and batch collection.** Measure time in engine transitions, encoding, network inference, replay sampling, and fitting. Batch inference across independent traversals and use compact arrays. Add workers with explicit seed ownership and a fixed policy version per collection phase.
- [ ] **Run a bounded Hold’em architecture pilot.** After profiling and the preceding gates, freeze a small capacity comparison with multiple independent seeds, common evaluation deals, and the same action representation. Compare parameter counts, equal training work, and equal cost. Exact widths and depth are experimental choices, not inherited Leduc settings. Benchmark concurrency before placing several jobs on one GPU. A short pilot measures throughput, memory, recovery, and early learning; it need not be long enough to rank strength or rule out eventual convergence.
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

Vast.ai remains a candidate for substantial training. The owner authorized a **$10 total Runpod CPU budget** on September 15, 2026, including more cores and longer small-game runs, separate from future GPU funding. The [bounded CPU pilot](docs/cpu-pilot.md) used only a few cents. The CPU pilot, capacity/fitting studies and [fresh confirmation](docs/reports/neural-readiness.md) are complete with conservative total CPU usage of $2.08, leaving $7.92 of that authorization. All rentals are terminated; the confirmation seeds are consumed and the readiness gate failed. New CPU runs still require a committed protocol and a cap within the remaining allowance. GPU funding is separate. Before a rental beyond that authorization, present a fresh cost/performance comparison and obtain the owner's campaign budget. The comparison must include:

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
- Treat the owner’s exploratory ideas as hypotheses to assess, not automatic implementation requirements. Recommend experiments for the decisions they can resolve; avoid repeated small-game tuning with little expected value for the Hold’em objective.
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
- **Training recovery and convergence checks delivered:** [PR #48](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/48) adds immutable training snapshots, complete replay/random-state restoration, fresh-process equivalence checks, and a campaign that requires every declared seed to meet final exploitability and value-error limits. The [validation report](docs/reports/neural-convergence.md) retains the results and a separate frozen-strategy fitting diagnosis. The original Leduc campaign fails its all-seed gate; milestone 3 remains open.
- **Strategy schedule and confirmation runner delivered:** [PR #54](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/54). Constant/cosine fits match the retained diagnostic optimizer; fresh-process recovery reproduces uninterrupted training; shared-replay controls match independent constant-rate runs. All 298 local tests pass. The [protocol](docs/neural-readiness.md) freezes both games, all eight reserved seeds, complete settings, final-only checks, paired reports, and rental limits. No reserved seeds or paid compute were used.
- **Fresh readiness confirmation completed, gate failed:** [PR #55](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/55), [results](docs/reports/neural-readiness.md). All sixteen jobs and eight controls completed. Kuhn 431 reaches 0.030425 against 0.03; Leduc 433 reaches 0.160619 against 0.15. Other seeds and all value errors pass. Cosine improves six of eight paired Leduc controls, with two regressions retained. All final exact played averages pass; Leduc 433's replay average already fails, so fitting alone does not explain the entire gap. Artifacts and all 24 exported policies were verified locally, the rental is terminated, and no model was promoted. Allow $0.65 for this campaign, $2.08 total conservative CPU usage and $7.92 remaining authorization.
- **Snapshot-average foundation delivered:** the [decision and tests](docs/decisions/snapshot-average.md) cover complete snapshot storage, correct alternating-update alignment, own-reach weighting, fixed-per-hand sampling, isolated recording and hash-pinned inference exports. All 318 tests pass, including 20 new tests. No rental, convergence campaign or model promotion occurred. Training/recovery integration remains open; existing runners still use the baseline.
- **Next task:** implement complete public-history and variable-seat Hold’em decision encoding, with seat rotation, suit permutation and observation-boundary checks. This begins milestone 4's independent engineering. The [snapshot-average decision](docs/decisions/snapshot-average.md) bounds the remaining small-game work; do not launch another tuning sweep or reuse consumed seeds as fresh confirmation.
- **Following tasks:** legal bet candidates with separate regret targets, consistent all-role self-play and complete recovery. Before another learning campaign, integrate snapshot-aware provenance/recovery and declare fresh readiness checks. Substantial training requires a readiness pass and the corrected no-limit pipeline. Profile before a budgeted architecture pilot.
- **CPU pilot completed:** [PR #49](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/49), [protocol](docs/cpu-pilot.md), and [results](docs/reports/cpu-pilot.md). Four workers delivered 3.63× remote throughput with exact training-result agreement. Width-128 strategy refits fell below 0.15 on all three retained Leduc seeds, with a narrow margin; these exploratory refits do not close milestone 3. The pod was terminated after verified retrieval. The $10 total CPU budget is separate from future GPU funding; only a few cents were used.
- **Strategy-capacity study completed:** [PR #50](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/50), [protocol](docs/strategy-capacity-study.md), and [results](docs/reports/strategy-capacity.md). Twelve new Leduc seeds ran 480 iterations with four strategy recipes compared at three checkpoints. All wider fits improve the final baseline; each passes 11/12 seeds. The best worst-seed result is 0.151373, above 0.15, so no recipe qualified; the eight confirmation seeds were still unused at that stage. Every final exact played average and replay average is below 0.15, directing the next task toward the remaining neural fitting gap. All artifacts were verified locally before the pod was stopped and terminated. The conservative quoted-rate estimate is about $1.18 for this rental, about $1.22 including the earlier pilot, within the $10 total CPU authorization. No model was promoted; milestone 3 remains open.
- **Next fitting study designed:** [PR #51](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/51), [protocol](docs/strategy-fitting-study.md), and [pinned plan](configs/solver/strategy-fitting-v1.json). The design declares 144 fits, paired random streams, per-situation diagnostics, final-only screening, and a $2 spending / 60-minute rental ceiling. The [input audit](docs/reports/strategy-fitting-inputs.json) verifies coverage in the saved snapshots without new training. The design PR performed no rental or additional model fitting. The runner and completed results follow in PR #52; no confirmation was performed.
- **Strategy-fitting diagnosis completed:** [PR #52](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/52), [results](docs/reports/strategy-fitting.md). All 144 fits and 432 checkpoints completed; all twelve original controls reproduced. Decayed minibatches pass the absolute limits in 36/36 fits (fixed-rate control: 31/36), but one paired worsening of 0.015791 exceeds the declared 0.005 limit, so no candidate qualifies. Exact gradients also pass all absolute checks and remain diagnostic-only. Frequently represented situations account for larger observed hybrid-policy effects than rare observed situations. The artifact archive is verified locally and the pod is terminated with no retained billable storage. Allow $0.21 for this rental, $1.43 conservative total CPU usage, and $8.57 remaining authorization. The study itself authorized no follow-on sweep or confirmation; milestone 3 remains open. The subsequent decision record defines the prospective integration and confirmation path.
- **Compute direction:** consider Vast.ai and an RTX 5090 for larger workloads after profiling. The small CPU pilot is not authorization for substantial GPU training.
- **Owner input needed before large training:** pilot and campaign spending ceilings, permitted unattended runtime, storage retention, and account access method. Confirm a different game profile if the cash-game defaults above do not match the intended table.

## References

- [PokerStars Hold'em rules](https://www.pokerstars.com/poker/games/texas-holdem/) and [betting examples](https://www.pokerstars.com/poker/learn/lesson/texas-holdem-rules/) provide a starting point for the cash-game betting specification. Resolve procedural details explicitly in `docs/rules.md`.
- [Poker TDA rules](https://www.pokertda.com/poker-tda-rules/) are a tournament reference; they must not silently define cash-game procedures.
- [PokerKit documentation](https://pokerkit.readthedocs.io/en/stable/) describes a candidate independent reference implementation. It is a comparison tool, not the authority when implementations disagree.
- [Deep CFR](https://proceedings.mlr.press/v97/brown19b.html) is the initial neural solver reference. Its two-player convergence result is not a six-player guarantee.
- [Single Deep CFR](https://arxiv.org/abs/1901.07621) is an alternative to compare after the first baseline.
- [Pluribus supplementary material](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf) informs the multiplayer blueprint and online-search direction. Adopting that direction does not establish equivalent playing strength.
