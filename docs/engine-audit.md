# Poker engine audit

Audited on 15 September 2026. The decision is to keep the small Rust fork, replace its betting and payout core, and validate it before further training.

## Fork and upstream

| Revision at audit | Meaning |
| --- | --- |
| Upstream `6b493e3` | Latest `Reinforcement-Poker/pokers` main; July 2025 merge of this fork's existing work |
| Fork main `f11337b` | Adds the forced all-in checkdown fix |
| Previously pinned `b1a48bd2067176cb73f95286d2e8878e3a2eb3d1` | Adds the no-raise-when-call-uses-the-stack fix on top of fork main |

The upstream merge brought no newer betting or settlement implementation. Before this rewrite, the fork/main tree differed from upstream/main only by the forced-checkdown change and its test. The old engine branch in the owner's separate checkout was preserved.

The [engine PR](https://github.com/dberweger2017/pokers/pull/1) merges upstream ancestry, retains both relevant fork fixes, and replaces the core. The bot pins merged revision `5db20e3d5d6862b32a7402035c1340b622d3b005` instead of following a moving branch.

## What needed replacement

- Calls could commit more chips than a short stack held, and settlement did not allocate proper side pots.
- The engine did not retain the last full raise or each player's reopening state. Checking could be rejected even when the player owed nothing.
- All-in players could remain in the action cycle. Bot logging had accumulated a forced-showdown workaround.
- Heads-up blinds and action order were incorrect.
- Wheel straights could be ranked above higher straights.
- Floating chip arithmetic, incomplete decks, and permissive input handling weakened invariants.

The replacement has integer accounting, capped calls, contribution-based pots, uncalled-chip returns, explicit odd-chip allocation, per-player reopening, automatic all-in runouts, correct heads-up order, and complete deck validation. Python game fields are read-only. The bot takes the engine's minimum increment directly and maps sizes to whole chips.

## Evidence and limits

The engine's local acceptance run passed 37 Python tests, including 5,000 generated hands against PokerKit 0.7.5 and the 9,908 bundled Pluribus records replayed both serially and in parallel. Rust property tests exercise 2–10-player legal hands with unequal stacks, termination, and exact chip conservation. Fixed examples test rules that random hands rarely reach.

PokerKit is a useful independent reference, but it disagrees on some short-all-in reopening cases, short-blind pricing, and odd-chip allocation. Those differences are documented precisely in the fork's `RULES.md`, with expected-result regressions and explicit comparison limits. The Pluribus adapter uses finite 100 BB stacks, a complete deck in two-round deal order, and explicit checks in place of zero-cost recorded calls.

The bot also tests the real installed engine through its sizing and logging adapters across four-, five-, and six-handed games. CI checks the engine on Linux, macOS, and Windows. No finite suite proves every possible hand; preserve each newly discovered disagreement as a replayable regression.

Remaining work is substantial: player observations and leakage checks, public disclosure events, table sessions, the evaluation arena, and a validated CFR implementation. Engine throughput should be profiled before choosing a GPU rental; correctness was the priority of this change.
