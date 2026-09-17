# DeepCFR Poker AI

Building a reproducible training system for strong **no-limit Texas Hold’em**, with six-handed play as the main target and support for four- and five-handed tables, changing lineups, and unequal stacks.

The agent must play by the documented rules and use exactly the game information available to a human in its seat. Deep CFR is our starting point; we will replace algorithms and architecture when measured results justify it.

**Status: building v0.5, a reproducible research release, on the way to v1.0. No current model has demonstrated professional-level Hold’em strength.**

[Release plan](#release-plan) · [1.0 strength standard](#what-10-means) · [Current progress](#where-we-are) · [Quick start](#quick-start) · [Roadmap](ROADMAP.md) · [Documentation](#documentation)

## Release plan

The next release is **v0.5**, followed by **v1.0**. The rebuilt system belongs to the 1.0 roadmap.

**v0.5 is a research release.** It should let someone train, interrupt and resume a run, export a playable model, and reproduce its evaluation. It ships with a documented longer training experiment: fixed settings and budgets, multiple seeds, learning curves, throughput and memory measurements, model artifacts, and every failure or weak result. Legal play and the player-information boundary remain mandatory. Professional strength, positive win rate, and completion of search or adaptation are not v0.5 requirements.

We can run longer exploratory training once the implementation, recovery and resource checks pass. Weak or inconclusive poker results are useful data; they do not block the next budgeted experiment. We will use that evidence to choose improvements for v1.0 instead of repeatedly tuning small diagnostics. A research checkpoint is clearly labelled as such; publishing it does not certify professional strength.

## What 1.0 means

**Version 1.0 is earned by demonstrated playing strength.** We will release it only when the code, paired with a documented and repeatable training recipe, produces an agent that meets our **lower-end professional cash-game standard** in the declared benchmark format.

A working trainer, completed roadmap, larger network, or long GPU run does not qualify on its own. We must actually train and evaluate qualifying models; “this should become strong with enough training” is not release evidence.

### The playing-strength standard

Our target is at least the lower end of professional no-limit cash-game play. For this project, we use a conservative, measurable release test: **demonstrably winning play against a credible reference group representing that level**, under a specified ruleset and resource budget.

“Lower-end professional” is a project target, not an established numerical poker rating. Before the qualification campaign, we must document the stakes, format, reference-group selection, and evidence that the group represents established winning professionals or comparably strong regulars. A job title, a few winning sessions, or success against our own bots is insufficient.

We will not claim “better than X% of poker players” without defining and measuring the relevant player population. Professional-level evidence applies to the tested format and opponents; it does not imply a universal ranking across cash games, tournaments, stakes, or player pools.

### Requirements for the release

| Requirement | Evidence needed for 1.0 |
| --- | --- |
| **A credible strength benchmark** | Direct evaluation against the documented human reference group, or an independently validated benchmark demonstrably calibrated to that group in the same format. Our style agents and historical checkpoints alone cannot establish professional strength. |
| **A statistically supported result** | Positive net win rate in big blinds per 100 hands (BB/100), with the lower bound of a predeclared 95% confidence interval above zero on the primary qualification benchmark. Include the declared rake and all completed hands; use an analysis that accounts for shared deals, sessions, opponents, and repeated measurements. |
| **A reliable training recipe** | At least three independently seeded training runs under the same declared recipe and compute budget, with per-seed results and all failures retained. Each must meet its predeclared qualification criteria; a lucky seed or the best intermediate checkpoint cannot rescue a failed campaign. |
| **Fresh confirmation** | Freeze the selected code, model, opponents/selection procedure, and evaluation protocol before the final test. Use held-out opponents or sessions and fresh evaluation data. Choose sample size, meaningful effect target, stopping rules, and scenario limits before looking at the results. An inconclusive result does not qualify. |
| **Correct and fair play** | Verified legal betting and settlement, consistent chip accounting, no privileged information in decisions, no sharing of private observations between agents, and no silent substitution of invalid actions. |
| **Supported table conditions** | Separate results for four-, five-, and six-handed starts, changing lineups, and declared unequal-stack scenarios. Publish predeclared regression limits for secondary scenarios; do not hide a failure in an overall average. |
| **A reproducible release package** | Versioned rules, source, dependencies, training configuration, hardware/runtime/cost record, complete resume support, model hashes and retrieval instructions, and a report sufficient to repeat training and evaluation. A fresh training reproduction must confirm the recipe's quality. |

The primary strength benchmark starts with **six-handed, 100 BB cash play**. The qualification protocol must pin the opponent population, seating, stack/reset policy, rake, decision-time limits, and any search or adaptation allowed during play. Four-/five-handed and other stack-depth results are reported separately; a professional-strength claim extends only to formats that have passed their own qualification checks.

These are **our release requirements**, not a claim that a particular win rate or hand count universally defines a professional player. Exact exploitability is useful in the small games we can solve; we do not have an exact exploitability certificate for full multiplayer no-limit Hold’em.

**The professional reference group and final qualification protocol have not yet been established. Until both the evidence and the implementation meet this standard, the project stays pre-1.0.**

## The game we are building for

| Area | Target |
| --- | --- |
| Game | Cash-game no-limit Texas Hold’em, table stakes, standard deck, one board |
| Players | Six-handed first; four and five players are first-class table configurations |
| Stacks | Begin with 100 BB; expand to unequal stacks and declared 20–200 BB scenarios |
| Table lifecycle | Arrivals, departures, sit-outs, and top-ups between hands |
| Information | Own hole cards, public board and actions, legal betting bounds, public stacks/positions, and legitimately observed opponent history |
| Current research rules | No rake or antes; the release benchmark must explicitly name its rake/rules profile. Unraked results do not establish profitability after rake. |
| Interface | A headless agent API, reproducible evaluation, and a way to inspect and replay decisions |

Hidden opponent cards, undealt cards, deck order, simulator seeds, and future outcomes never belong in the agent's input. Training may use simulated payoffs and counterfactual branches to construct learning targets; those targets do not grant extra information during play. Search must infer possible hidden worlds from legitimate observations.

The [rules profile](docs/rules.md), [observation contract](docs/observations.md), and [session contract](docs/sessions.md) define the supported behavior. Tournament payouts/ICM, straddles, multiple runouts, and live audiovisual tells are outside the initial scope.

## Where we are

The project began as an earlier Deep CFR implementation. We are rebuilding its learning and evaluation foundations, retaining useful regression cases and replacing components as their successors pass meaningful checks. Backwards compatibility is not a requirement.

| Area | Current evidence |
| --- | --- |
| Rules engine | Maintained Rust [`pokers` fork](https://github.com/dberweger2017/pokers), pinned to an audited revision; integer chips, legal raises, side pots, and settlement checks. [Engine audit](docs/engine-audit.md) |
| Player information and sessions | Immutable player observations, complete public history, identity-owned records, and changing four-to-six-player lineups. [Observation](docs/observations.md) / [session](docs/sessions.md) contracts |
| Evaluation | Reproducible schedules, separate data splits, paired reports, varied style opponents, and hash-pinned historical models. Professional-strength opponents remain an open requirement. [Benchmark guide](docs/benchmarks.md) |
| Tabular reference | Kuhn and Leduc CFR checked against exact best responses and independently solved equilibrium values. [Results](docs/reports/tabular-validation.md) |
| Neural baseline | Small-game Deep CFR with complete snapshot-average training recovery and verified inference exports. Fresh snapshot readiness passes in both games. [Contract](docs/neural-cfr.md), [snapshot design](docs/decisions/snapshot-average.md) |
| Neural convergence | Snapshot confirmation passes: 8/8 seeds in each game meet the original exploitability and value-error limits. Milestone 3 is complete. [Latest results](docs/reports/snapshot-readiness.md) |
| Hold’em decisions | Full current-hand event encoding, variable-seat masks, board reveal stages and a small sequence model. [Contract](docs/holdem-encoding.md) |
| Hold’em betting | Exact legal bet candidates, per-action regret/value heads and branch-payoff targets. [Contract](docs/holdem-betting.md) |
| Hold’em collection | External sampling for every role against isolated current policies, with exact action records and reproducible 100 BB checks. [Contract](docs/holdem-collection.md) |
| Hold’em fitting | Separate role reservoirs and whole-iteration rollback, with explicit sampled replay/root normalization and verified recovery. [Reference contract](docs/holdem-training.md), [sampled trainer](docs/holdem-sampled-training.md) |
| Hold’em baseline pipeline | Averaged play, full iteration-boundary recovery, deterministic button rotation and a multi-seed training/arena runner. [Contract](docs/holdem-baseline.md), [first report](docs/reports/holdem-baseline.md). Substantial learning and strength remain unproven. |

**The longer v0.5 experiment is complete.** All three six-handed 100 BB seeds reached 512 iterations, with 147,456 scheduled evaluation hands, zero invalid actions and verified checkpoint recovery. The [report](docs/reports/holdem-longer-training.md) retains every seed and learning curve. All 48 paired comparisons are inconclusive; the final policies lose heavily against the style pool. This is research evidence, not demonstrated strength or a release announcement.

**Full-game learning remains unproven.** The [paired policy comparison](docs/reports/holdem-policy-comparison.md) found no consistent fitting improvement, and the [persistent critic](docs/reports/holdem-persistent-critic.md) failed its cost screen. Expanding a second own decision reduced variance on [controlled frozen probes](docs/reports/holdem-collector-branching.md), but the [fresh six-run comparison](docs/reports/holdem-branching-online.md) did not meet its consistency criterion: one seed improved clearly relative to its control, while all final policies still lost heavily against styles. All six jobs reached 512 iterations with verified recovery; first-decision collection remains the default.

**The [reliable-target architecture comparison](docs/reports/holdem-representation.md) is also complete:** larger and card-separated models fit training examples better, but none qualified on unseen validation boards. **Next: separate training-board diversity from explicit card relationships**, using fresh holdouts and features derived only from visible cards, before another long campaign. Sparse postflop coverage and extreme sampled targets remain open problems. Artifacts are retained locally; public hosting and v0.5 release packaging are ahead. No model has been promoted, and the v1.0 professional standard remains unproven.

## The route to 1.0

1. **Trust the game and the measurements.** Rules, legal observations, sessions, reproducible evaluation, and independent small-game references.
2. **Validate the learning algorithm.** The snapshot-average path passes fresh predeclared small-game checks; preserve that evidence and move to Hold’em.
3. **Build the no-limit learner.** The full-decision, variable-seat pipeline and complete recovery are implemented. Establish meaningful all-role learning and bet preferences against fixed benchmarks.
4. **Scale from measured throughput.** Profile collection and fitting, batch work, and run bounded hardware pilots before larger campaigns.
5. **Improve demonstrated playing strength.** Evaluate range-aware search and opponent adaptation as separate changes against fixed baselines.
6. **Qualify the release.** Establish the professional reference benchmark, run the declared training and confirmation campaigns, and publish the complete evidence package.

The [roadmap](ROADMAP.md) contains the PR-sized work, acceptance checks, and current next task. Completing implementation milestones does not waive the release standard above.

Vast.ai remains a candidate for substantial training. Bounded Runpod CPU studies have already measured parallel throughput and tested small-game learning. Future hardware choices follow profiling and a cost/performance comparison; no GPU campaign is committed.

## Quick start

Use **Python 3.11**, Git, and a Rust toolchain. The engine is built from the pinned fork. Commands run from the repository root.

```bash
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git
cd deepcfr-texas-no-limit-holdem-6-players

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

For the CPU Torch version used by the current CI and neural reports, install `torch==2.5.1` from the [PyTorch CPU package index](https://download.pytorch.org/whl/cpu) before installing the requirements. The general requirements allow other Torch 2.x versions; exact report reproduction requires matching the recorded runtime.

Run a short neural check, pause it, and resume in a new process:

```bash
python -m scripts.check_deep_cfr \
  --plan configs/solver/neural-smoke.json \
  --stop-after 1 --out results/readme-paused

python -m scripts.check_deep_cfr \
  --resume results/readme-paused --out results/readme-resumed
```

Run and reproduce a small Hold’em arena schedule:

```bash
python -m scripts.run_arena \
  --plan configs/arena/smoke.json --out results/readme-arena

python -m scripts.run_arena \
  --reproduce results/readme-arena --out results/readme-arena-replay
```

Use fresh output directories; runners preserve existing results. These commands check execution and reproducibility. They do not train or certify a professional-strength model.

## Documentation

| Read this | For |
| --- | --- |
| [Roadmap](ROADMAP.md) | Development order, current blockers, contribution workflow, and compute policy |
| [Rules](docs/rules.md) / [engine audit](docs/engine-audit.md) | Supported poker rules and verification evidence |
| [Observations](docs/observations.md) / [sessions](docs/sessions.md) | What the agent can see and how tables change between hands |
| [Evaluation arena](docs/evaluation.md) / [benchmarks](docs/benchmarks.md) | Schedules, opponents, metrics, and reproducible comparisons |
| [Tabular solver](docs/solver-reference.md) / [neural solver](docs/neural-cfr.md) | Learning conventions, commands, snapshots, and diagnostics |
| [Snapshot readiness results](docs/reports/snapshot-readiness.md) | All seeds, artifact verification, rental cost, limitations and the next decision |
| [Repository layout](docs/repository-layout.md) | Supported commands, retained historical opponents and retired workflows |

The old trainers, desktop UI and experimental opponent model have been retired. Use the current headless commands from a source checkout; release packaging and a new play interface follow the roadmap. Historical standard checkpoints remain read-only arena opponents.

## Research foundations

- [Deep Counterfactual Regret Minimization](https://proceedings.mlr.press/v97/brown19b.html) — the original inspiration and starting neural algorithm.
- [Single Deep CFR](https://arxiv.org/abs/1901.07621) — an alternative approach to strategy averaging that we may compare separately.
- [Pluribus: Superhuman AI for multiplayer poker](https://doi.org/10.1126/science.aay2400) and [supplementary material](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf) — multiplayer self-play, search, and evaluation against professional players. This project's release criteria are our own; citing that result does not establish equivalent strength here.

## License and acknowledgments

[MIT](LICENSE.txt). Thanks to the original [`pokers` maintainers](https://github.com/Reinforcement-Poker/pokers), contributors who reported and reproduced game/training failures, and the open-source research and tooling behind this work.
