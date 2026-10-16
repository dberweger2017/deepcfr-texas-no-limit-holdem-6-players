# v0.5.5 acceptance: HU200 against Slumbot

**Outcome (October 16):** the confirmation under this protocol failed on seed 2026101302 ([report](reports/hu200-v055-slumbot-confirmation.md)). New candidates follow [version 2](hu200-v055-acceptance-protocol-v2.md): the three-seed blend, 30,000 pooled hands across three accounts.

Owner-approved October 14, 2026, before the exploratory frozen 20B + translation
run reached 5,000 hands. These criteria decide whether the recipe below
qualifies for v0.5.5; publishing still needs the owner's explicit go.

## Claim

With 95% confidence, the candidate loses less than **50 BB/100** (half a big blind
per hand) to Slumbot at its public API's 200 BB stacks, and the same holds for
two further independent training seeds of the same recipe.

## Recipe and candidate

Every seed follows one recipe, trained on the 16 GiB Apple M4 under the
existing guards (60 s sustained RSS > 13 GiB, critical pressure, free memory
< 8% or swap growth > 4 GB; disk < 16 GiB stops at once):

1. HU200, #163's pinned K50 tables (flop `084e243d…`, turn `c0bc6a8a…`,
   river `70c1cb5e…`), menu v1, R=1, linear CFR, opponent-sampled lifetime
   average; trained to [#250's](reports/hu200-k50-first.md) predeclared cap of
   85,373,862 entries, ending at the first complete iteration past it.
2. Continued with [frozen-table mode](reports/frozen-table.md)
   (`--freeze-at-cap --max-entries` = that seed's terminal entries) to
   20,000,000,000 cumulative nodes, as in [#265](reports/hu200-k50-frozen.md).
3. Average exported with the every-row audit.
4. Played with [#271's action translation](reports/hu200-k50-translation.md),
   512 states and 128 events, uniform fallback for unresolved keys.

| Seed | Status | Release role |
|---|---|---|
| 2026101302 | Trained: #265's 20,000,002,032-node average, SHA256 `9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6` | **The v0.5.5 candidate** |
| 2026101421 | To train | Confirmation seed |
| 2026101422 | To train | Confirmation seed |

## Slumbot confirmation, per seed

- **Size:** the first **10,000** completed hands, then stop. No early stop,
  no extension and no other hands count. Earlier runs of these models,
  including October 14's exploratory runs, are reported separately and never
  pooled with a confirmation.
- **Policy seeds:** 20261014055101 (seed 2026101302), 20261014055201
  (2026101421) and 20261014055301 (2026101422). If a run stops before
  10,000 hands, it continues in a new run with the next policy seed
  (…02, …03), and parts are pooled in play order.
- **Client:** `scripts.play_slumbot` with `--action-translation
  --translation-max-states 512 --translation-max-events 128`, a 30 s request
  timeout, `--max-abandoned 200` and `--max-rejected 50`.
- **Validity:** every journal's hash chain verifies; every settlement
  recomputes from the public actions and revealed cards; Slumbot's session
  totals reconcile with the verified nets; abandoned hands stay at or below
  200 (2%). A run that fails validity is reported, does not pass, and
  isn't repeated.

## Rule

For each seed, compute raw BB/100 over its 10,000 hands with the 95% interval
`mean ± 1.96 × SD / √n`, as `scripts.analyze_slumbot_run` reports.

- **Pass:** the lower bound is above **−50 BB/100**.
- **v0.5.5 qualifies** only if all three seeds pass. No multiple-comparison
  adjustment: each seed must pass on its own.
- **If any seed fails,** this recipe doesn't qualify. All results are
  published as evidence, and the owner decides what comes next.

The all-in-adjusted result, results by seat and the play profile
(`scripts.profile_slumbot_runs`) are reported alongside but don't decide.

**Expected behavior:** at 10,000 hands the interval is about ±30 to ±40 BB/100.
A model whose true rate is +25 passes with about 97–99% probability, one at
break-even about 75–90%, one at −30 about 20–30%, and #250's 752M model
(−77 ± 18) almost never.

## Out of scope

This doesn't establish HU100 strength, profitability against Slumbot, or
strength against other opponents.
