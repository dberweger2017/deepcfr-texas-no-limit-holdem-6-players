# Roadmap

## Goal

Build a strong no-limit Texas Hold'em agent that plays by the selected table rules and sees exactly what a human in its seat would see. Six-handed play is the main target; four- and five-handed tables, changing lineups and unequal stacks are part of the design.

Playing strength is the measure of progress. A finished training run, a lower loss or a win against random opponents is useful evidence, but none of them shows the agent plays good poker.

Anything can be replaced: engine integration, algorithm, models, scripts, file formats. Keep useful regression scenarios and experiment records.

The full history of plans, protocols and results is in [docs/roadmap-history.md](docs/roadmap-history.md) and [docs/research-history.md](docs/research-history.md).

## Release milestones

The v0.4.x releases improve heads-up 20 BB; v0.5 starts at 100 BB and adds 200 BB for the v0.5.5 external benchmark. Later minor releases add players. The [readme](readme.md#limits-and-release-plan) has the short version.

| Release | What it means |
|---|---|
| **v0.4** (released, `v0.4.0`) | Rebuilt research preview: a local heads-up 20 BB table, the hash-verified B100M policy, replayable sessions and an honest evidence index. Not a strength claim. |
| **v0.4.1** (released, `v0.4.1`) | The 1B-node linear-CFR opponent-sampled average, seed 2026100601; it beats v0.4.0 in a fresh direct match and passes the arena rule. [Release comparison](docs/reports/v0.4.1-release.md). v0.4.0 stays available. |
| **v0.4.2** (released, `v0.4.2`, current) | The same recipe at 10B nodes, fixed seed 2026100601. Direct gain +3.50 [+1.63, +5.37] BB/100; all four gates pass, fresh LBR narrowly. [Verified publication record](docs/releases/v0.4.2/PUBLICATION.md). v0.4.0/v0.4.1 stay available. |
| **v0.4.x** (now) | The Pluribus recipe on heads-up 20 BB: average-policy play, a native trainer, a better training procedure or card abstraction, turn/river search, then flop search, in the order the evidence supports. Each patch must beat its predecessor in a paired arena without severe scenario regressions. |
| **v0.5.0** (released, `v0.5.0`) | Heads-up 100 BB: #207's 1B-node average, seed 2026100601, with public-history translation (512/128). It beats four of five scripted opponents across three seeds; against pot pressure it's −9 [−39, 20] BB/100. Validated through internal checks and scripted evaluations only: no suitable free public 100 BB opponent has been found, so there is no external strength claim. [Model card](docs/releases/v0.5.0/MODEL_CARD.md). Later v0.5.x releases continue on 100 BB. |
| **v0.5.1** (released, `v0.5.1`) | Heads-up 100 BB: #223's 2B-node average from the same seed, with the same translation. Beats v0.5.0 head to head by +29.51 [26.82, 32.20] BB/100, and two independent seeds repeat the 2B-over-1B gain at +28.14 and +27.90 (#226). No scripted difference from v0.5.0 is significant; pot pressure still beats it. [Model card](docs/releases/v0.5.1/MODEL_CARD.md). |
| **v0.5.2** (released, `v0.5.2`) | Heads-up 100 BB: #226's 4B-node average from the same seed, with the same translation. Beats v0.5.1 head to head by +17.90 [14.98, 20.83] BB/100, and two independent seeds repeat the gain over their own 2B models at +18.67 and +19.42 (#236). No severe scripted regression from v0.5.1; pot pressure still beats it. [Model card](docs/releases/v0.5.2/MODEL_CARD.md). |
| **v0.5.3** (released, `v0.5.3`) | Heads-up 100 BB on the larger menu (min, ½ pot, pot, 1.5× pot, 2× pot, all-in) with K50 postflop cards: PR272's frozen 5B average, seed 2026100601, with the v2 translation adapter. Beats v0.5.2 head to head by +37.10 [30.16, 44.04] BB/100 with translation on for both; all three training seeds beat v0.5.2 by +41 to +47 (#296). Translation passes its no-harm safeguard; no benefit against human bet sizes is measured yet. [Publication record](docs/releases/v0.5.3/PUBLICATION.md). |
| **v0.5.5** | Heads-up 200 BB with a policy trained and validated at that depth, tested against Slumbot at its public API's 200 BB stacks. Publish results and uncertainty; these do not establish HU100 strength. **Acceptance:** the original recipe failed its [first protocol](docs/hu200-v055-acceptance-protocol.md) on one of three seeds ([#304](docs/reports/hu200-v055-slumbot-confirmation.md)). [Version 2](docs/hu200-v055-acceptance-protocol-v2.md) (owner-approved October 16, before any candidate hand): #306's three-seed blend plays 10,000 fresh hands on each of three new accounts, and the pooled 30,000-hand 95% lower bound must be above −50 BB/100; "beats Slumbot" is claimed only if it is above 0. **Confirmed October 17: pooled −19.67 [−38.57, −0.76] BB/100, qualifies ([report](docs/reports/hu200-v055-blend-confirmation.md)).** |
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
3. **Training procedure — CFR+ rejected; 10B scaling checks pass.** The tested CFR+ floor ("0.4.0-shield") lowered weakness probes but lost directly to v0.4.0 ([report](docs/reports/hu20-cfr-plus.md), [floor control](docs/reports/hu20-floor-control.md)). Training the v0.4.1 recipe to 10B nodes beats v0.4.1 directly ([#185](docs/reports/hu20-o-10b.md)); the fresh [#188 LBR confirmation](docs/reports/hu20-v042-lbr-confirmation.md) narrowly passes the unchanged safeguard. The owner-authorized [v0.4.2 publication](docs/releases/v0.4.2/PUBLICATION.md) is complete: reviewed green-check merge, exact-source tag and verified downloads/runtime.
4. **Card abstraction — in progress.** Follow the [abstraction lessons](docs/reports/hu20-abstraction-lessons.md). Each step depends on the one before:
   1. **validation complete:** [#190](docs/reports/hu20-global-bucket-validation.md) passes at K50 E 0.3917 [0.3614, 0.4234] BB, reproducing fitted50's advantage over v1. K200 is descriptively worse at 0.4233; all coverage/replay and independent audit checks pass.
   2. **bucket keys done:** [`…-equity-k50-v1`](docs/reports/native-equity-bucket-keys.md) keys postflop cards by #163's pinned K50 tables in the native trainer, bench and Python key function, with exact Rust/Python parity. It has about 1.4× v1's keys;
   3. **trained bench passes:** [#232's fresh M4 all-40 scoring](docs/reports/hu20-equity-bench-scoring-m4.md) gives matched K50 − v1 = **−0.2038 [−0.2836, −0.1274] BB**, upper bound below −0.10; 10M point contrast **−0.1853 BB** is negative. #222's exports/freeze and science are unchanged. #222/#225/#230 stops and the previously inspected nine-root analysis remain preserved; this is one fixed training seed and benchmark, not fresh confirmation;
   4. **evaluation support ready ([#235](docs/reports/k50-evaluation-support.md)):** pinned K50 checkpoint/average loaders, mixed-schema arena/direct play and LBR/native pressure pass the guarded M1 correctness smoke. Turn search and web runtime explicitly refuse K50. The shorter owner-authorized [matched 5B first pass](docs/reports/hu20-k50-5b-first-pass.md) now beats v1 +10.31 [7.66,12.96] BB/100 across three fixed seeds. Recommend the longer [learning-curve protocol](docs/hu20-k50-full-game-proposal.md); the owner decides its scope and budget. No adoption or release change.
5. **Turn search — diagnosed, adoption unresolved.** The [fixed-work arena](docs/reports/hu20-fixed-work-arena/attempt-2-closeout.md) gains against LBR/native pressure; the [stackoff diagnosis](docs/reports/hu20-search-stackoff-diagnosis.md) finds mis-modeled value callers and future folds, rather than large-bet over-calling. Its −10.35 BB/100 regression is plausible but not established after 13-panel adjustment. Fresh unchanged-search confirmation is quoted, awaiting owner approval; no direct search-versus-no-search match yet.
6. **Flop search — conditional,** only if the evidence still supports it at 20 BB.
7. **v0.5 groundwork, in parallel:** native 100-BB support and exact HU20 recovery are verified ([#197](docs/reports/native-recovery-hu100.md)). The #197 table-capacity stop was followed by [#202 export/audit memory work](docs/reports/hu100-export-audit-memory.md), which removes the serialization bottleneck. Merged [#203](docs/reports/native-hu100-growth.md) reaches 20,001,470 nodes and improves both aggressive primaries. Merged [#204](docs/reports/native-hu100-growth-50m.md) stops cleanly at 39,438,279 nodes under its fresh entry ceiling; loose gains continue, tight is inconclusive, and loose/pot still lose. The [compact native table](docs/reports/native-compact-table.md) then cuts training memory 4.5× at that endpoint, to about 90 bytes/entry growing linearly, with byte-identical checkpoints. [#207 growth to 1B](docs/reports/native-hu100-growth-1b.md) then reaches 1,000,002,065 nodes at 3.56 GiB training peak: loose-aggressive gain +214.39 [122.28,306.51] BB/100 improves; tight is inconclusive (97.5% primaries). Known late-street visits become much denser; unsupported histories and an owner-defined external benchmark remain. The [#223 direct ladder](docs/reports/hu100-3b-ladder.md) extends the exact #207 seed to a preventive 2B disk-admission terminal; 3B never started. Terminal versus 1B passes at +29.51 [26.82,32.20] BB/100; 1B versus 500M is descriptively +38.93 [35.92,41.95]. All final hands replay and reproduce; the five translated scripted contrasts are inconclusive. [#226's direct ladder](docs/reports/hu100-4b-seed-ladder.md) reaches4B and passes4B-over2B +17.90 [14.98,20.83] BB/100; two independent2B-over1B contrasts pass +28.14 [24.87,31.41] and+27.90 [24.65,31.15] (98.33% adjusted). This repeats conditional gains across three fixed seeds; training-seed population and external qualification remain. The [small scripted-opponent baseline](docs/reports/native-hu100-playing-baseline.md) is development evidence, not a strength gate. Average exports already load compactly ([#186](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/186)). Explicit [local HU100 research play](docs/hu100-local-play.md) and [external benchmark preparation](docs/hu100-external-benchmark.md) are verified; external benchmark execution follows in a new PR.

**Proposed v0.4.x exit (owner to confirm):** each ingredient above has been adopted or rejected on recorded evidence, and the latest patch beats v0.4.0 in the paired arena.

## Ground rules

- **Information:** the agent receives an immutable observation of what its seat legitimately knows — its cards, the board, every observed action and amount, legal action bounds, revealed cards and its own history with each player. It never sees other players' hidden cards, the deck, seeds, future outcomes or evaluator data. Training may use simulated payoffs and counterfactual branches as targets, never as inputs; search samples hidden worlds from legal beliefs, never from the real deal. Details: [observations](docs/observations.md).
- **Rules:** cash-game no-limit Hold'em under the profile in [docs/rules.md](docs/rules.md). Tournament rules, ICM, straddles, rake and multiple runouts are separate scope decisions, never silent approximations.
- **Architecture:** rules engine, session manager, agent (observation in, legal action out), solver/trainer, evaluation arena, search/adaptation — small explicit boundaries. Native code where profiling justifies it.
- **Evaluation:** paired deals, rotated seats, uncertainty over independent blocks, and fresh confirmation data. Approximate best responses (LBR) are weakness probes, not exploitability certificates.
- **Strength gate:** a change of default model or a claim of stronger play needs a predeclared paired comparison whose 95% interval clears zero on the primary benchmark, with scenario regression limits written down in advance and both statistical and practical effect sizes reported. Don't re-inspect one test set until something passes; multi-seed claims include training-seed variation.
- **Experiments:** each campaign declares its hypothesis, opponents, seeds, maximum cost and runtime, evaluation schedule, stopping conditions and decision rule before the main run. Numerical failures, information leaks, broken accounting, invalid play and resource exhaustion stop a run. Weak or inconclusive results are kept as evidence, never a reason to keep extending the budget. A positive poker result is not required before longer, budgeted training.

## How we work

[CONTRIBUTING.md](CONTRIBUTING.md) defines the branch, commit, PR, review and code conventions, the research-proposal process and the decisions kept by the maintainer. They apply to everyone, including the maintainer's own work.

Update the v0.4.x path and Current position when work lands, keeping each entry short and moving detail into reports. This roadmap doesn't schedule unattended work.

## Help wanted

Open tasks for contributors, grouped by the compute they need. Comment on or open an issue before starting, so work isn't duplicated; [CONTRIBUTING.md](CONTRIBUTING.md) explains setup and the research process.

**No large compute**

- **Glossary.** A `docs/glossary.md` that explains the project's terms in plain language (BB/100, HU20/HU100/HU200, nodes, card buckets and K50, betting menus, translation, LBR, paired blocks, opponent-sampled average) and is linked from the readme.
- **Script index.** A `scripts/README.md` that separates maintained entry points from frozen one-campaign launchers, which [development.md](docs/development.md#research-versus-reusable-code) says are currently mixed.
- **Quick first check.** A documented test selection a newcomer can run in a few minutes, with its measured runtime, alongside the full suite.
- **Packaging.** A `pyproject.toml` with play and development extras. `requirements.txt` must keep its exact bytes, because historical opponent freezes pin them.
- **Web table.** Usability, accessibility and hand-history review in `apps/poker-web/` and `src/play_api/` ([play guide](docs/play-web.md)), without changing what a player can see.
- **Scripted opponents.** New tested styles in `src/arena/heuristics.py` that widen the evaluation panel.
- **Native trainer speed.** Faster training in `native/hu20-trainer/` with byte-identical checkpoints and parity tests.
- **External 100 BB benchmark.** No suitable free public heads-up 100 BB opponent has been found; propose one, with its API and terms.
- **Issue triage.** Reproduce and answer open issues, such as [#33](https://github.com/dberweger2017/CFR-NLHE-6max/issues/33) on bet rounding with the `pokers` engine ([rules](docs/rules.md#amounts-and-actions)).

**Laptop-scale experiments**

- **Reproductions.** Rerun a small published check (small-game solvers, `scripts.check_game`, a short arena) on other hardware or operating systems and report any difference.

**Research proposal first**

- **Search.** Turn and flop search at 100 BB, building on the [HU20 turn-search diagnosis](docs/reports/hu20-search-stackoff-diagnosis.md).
- **Three players (v0.6).** Multiway search on top of the [Phase 2 blueprints](docs/reports/nplayer-blueprint-phase2.md).

## Current position

- **v0.5.5 confirmation diagnosis (October 17):** card-only AIVAT on all 30,000 hands gives pooled plain **−21.12 [−36.39, −5.84] BB/100**, worth 1.53 raw hands per hand; raw protocol v2 qualification stands. Across the confirmation and four earlier runs, the largest recurring flop-fold path is BB call of a 2 BB open followed by check-fold to a half-pot bet. These are descriptive targets, with no evaluated fix or policy change. [Report](docs/reports/v055-confirmation-diagnostics.md).

- **v0.5.5 confirmation passes (October 17):** under [protocol version 2](docs/hu200-v055-acceptance-protocol-v2.md), #306's three-seed blend played 3 × 10,000 fresh hands on new accounts: pooled raw **−19.67 [−38.57, −0.76] BB/100**, so the lower bound clears −50 and v0.5.5 qualifies. The upper bound below 0 means it loses to Slumbot by a small, detectable margin. All journals verify, sessions reconcile, 0 abandoned. Most of the loss comes from flop folds and the big blind. Release awaits the owner's go. [Report](docs/reports/hu200-v055-blend-confirmation.md).

One short entry per active line of work; details live in the linked reports. Storage and cleanup receipts belong in [docs/artifacts](docs/artifacts/) and [RESULTS_INDEX.md](RESULTS_INDEX.md), not here.

- **Frozen HU200 blend ([#306](https://github.com/dberweger2017/CFR-NLHE-6max/pull/306)) complete:** all fixed evaluations and 2,162,688 stored-hand replay/reproduction checks finish on the 16 GiB M4, at 12.75 GB peak family RSS. No direct gain is established; the blend still loses to 1.5× pot pressure, −29.39 [−55.24,−3.53] BB/100, retained as failed adverse evidence without retry or tuning. Drive evidence and unpacked 2 TB copies are verified. [Full results, coverage, failures and retrieval](docs/reports/hu200-v055-blend-run.md). Owner decides merge and any Slumbot use.

- **Open PRs:** [#301](https://github.com/dberweger2017/CFR-NLHE-6max/pull/301) tests whether a larger HU200 table gives a stronger 20B player.

- **v0.5.5 acceptance, version 2 (October 16):** the owner approved [a new protocol](docs/hu200-v055-acceptance-protocol-v2.md) for the three-seed blend (#306) before any of its Slumbot hands: 30,000 fresh hands across three accounts, pooled lower bound above −50 to qualify, "beats Slumbot" only above 0. Pass chances are stated in advance; the first protocol's failure stands. Exploratory blend hands are allowed after this merge and never count.

- **v0.5.3 published and verified:** [stable Latest](https://github.com/dberweger2017/CFR-NLHE-6max/releases/tag/v0.5.3) since October 16, tag at #302's merge `74f1be71`; the twelve assets (three model parts, three K50 tables) verify byte for byte from the draft and from the public URLs ([publication record](docs/releases/v0.5.3/PUBLICATION.md)).

- **v0.5.5 Slumbot confirmation ([#304](https://github.com/dberweger2017/CFR-NLHE-6max/pull/304)), recipe does not qualify:** 10,000 hands per seed, raw: 2026101422 **−7.87 [−39.39, +23.65]** and 2026101421 **−10.66 [−40.53, +19.21]** pass; the frozen 20B candidate 2026101302 **−23.08 [−56.53, +10.38]** fails the −50 lower bound. All journals verify; abandoned 33/60/108 of 200. Interpretation, not decisive: 9.3 BB/100 of 1302's gap is all-in luck, and at a true −10 to −14 the three-seed rule passes only about a quarter to a third of the time. No rerun or extension; the owner decides the next candidate and test design. [Report](docs/reports/hu200-v055-slumbot-confirmation.md).

- **Three-player Phase 2 ([#292](https://github.com/dberweger2017/CFR-NLHE-6max/pull/292)), owner authorized merge after green CI:** all three 1B-node full-average blueprints pass the declared gain over their own 200M models; all 42 fixed comparisons replay and exactly reproduce before readout. Scientific work is complete. Cloud closeout is complete at 237/237 accepted pieces with final restoration index and failure/provenance ZIP. Cleanup remains deferred because dependency inspection was denied for five owned processes; originals/shared inputs remain protected. [Results](docs/reports/nplayer-blueprint-phase2.md), [archive receipt and restoration](docs/artifacts/pr292-closeout-20261016.md). Search and release remain out of scope.

- **HU200 frozen three-seed blend ([#297](https://github.com/dberweger2017/CFR-NLHE-6max/pull/297)), capacity stop, owner-authorized merge:** all three raw 20B checkpoints verify and reproduce their released averages exactly. The unchanged sequence-form rule yields 156,208,331 union keys; the first disk gate requires 74.47 GB free versus 63.33 GB available, an 11.14 GB shortfall. No full blend, cache or evaluation ran; the normalized-export stop remains preserved. Evidence archives, resource traces, failures and cleanup receipts are accepted and indexed. M4 released October 16 at 01:19 Madrid. No strength, release or default change; continuation needs a new owner decision. [Report](docs/reports/hu200-v055-blend.md).

*Updated October 17, 2026.* Earlier entries, including storage receipts, are in [roadmap history](docs/roadmap-history.md#superseded-current-position-through-october-16-2026).
