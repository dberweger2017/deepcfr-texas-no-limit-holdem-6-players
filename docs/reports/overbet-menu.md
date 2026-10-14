# Overbet menu: half-pot, 1.5× and 2× pot raises

[#251](hu100-pot-pressure-diagnosis.md) found bet size to be the main reason HU100 releases lose to the scripted pot-pressure opponent: when it raises 1× pot instead of 1.5×, v0.5.2 gains +72.36 [58.58, 86.13] BB/100. The v1 menu offers min-raise, pot and a conditional all-in, so overbets never appear in v1 training. This adds a second menu with **half-pot, 1.5× pot and 2× pot** raises, for the native trainer and the Python key. The original change was engineering only; [#260's first capped K50-v2 training](hu100-k50-v2.md) now tests it. v1 stays the production menu.

## The menu

`--menu v2` (Rust) and `menu_sizes="v2"` (Python `choices`) offer fold when no check is available, check, call, and raises to:

| Name | Raise above a call, as a fraction of the pot after calling |
|---|---|
| `min` | the legal minimum |
| `half` | ½ |
| `pot` | 1 |
| `pot-1.5` | 1½ |
| `pot-2` | 2 |
| `jam` | all-in, only when at most 2× pot, as in v1 |

Each size is clamped to the legal bounds, and a target an earlier name already offers is skipped, as in v1. Two consequences:
- **Preflop, `half` usually equals the min-raise,** so the opening menu is fold, call, min, pot, pot-1.5, pot-2.
- **`jam` itself never appears under v2.** Whenever v1's rule allows the all-in, it is at most 2× pot, so `pot-2` (or `pot-1.5`) is already clamped to the same amount and takes its place. The all-in is offered exactly when v1 offers it, under the largest size it equals.

A decision has at most 8 actions instead of 5, and menu codes are 16-bit. v1's names keep their bits; the new names have new bits, and the canonical order follows size rather than bit position.

## History labels

v1's history puts each raise into one of four size buckets, so a ½-pot bet and a pot bet would share a label, as would 1.5× and 2×. v2 labels each raise by the same measure the menu uses: the raise above the current bet divided by the pot after calling. The bucket boundaries lie between the menu's sizes, at 0.35, 0.75, 1.25, 1.75 and 2.5, in exact integer arithmetic so Rust and Python cannot round differently. The labels are `raise-s`, `raise-h`, `raise-p`, `raise-o15`, `raise-o2` and `raise-x`, each with an `-all-in` variant. Off-menu raises from external opponents fall into the nearest bucket.

## Identity

Six new schemas mirror v1's and K50's, for example `hu100-native-reopening-ordered-history-card-menu-v2` and `hu100-native-reopening-ordered-history-equity-k50-menu-v2`. Checkpoints record `action_menu: hu100-min-half-pot-overbet-conditional-jam-native-reopening-v2`. Resume refuses a checkpoint whose menu differs from the one requested, and `AveragePolicy` loads a v2 average only under its v2 schema. Translation stays v1-only: its loader refuses every schema it doesn't list.

## Exactness

| Check | Result |
|---|---|
| v1 checkpoints with the new binary: HU20 30M nodes, HU100 20M nodes | Byte-identical to the previous binary (`197a1840…`, `04dea4ae…`) |
| Rust/Python parity on Python-engine hands with random off-menu sizes, v1 and v2, HU20/HU100/HU200, 1,500 hands each | 0 mismatched hands over about 59,000 decisions |
| Same in CI: v2 for HU20, HU100 and HU200, and HU100 with synthetic K50 tables; v1 still passes | 0 mismatched hands |
| A v2 run resumed from half way | Byte-identical to the uninterrupted run; resuming it as v1 is refused |
| A v2 average loaded in Python | Plays complete hands with the v2 menu; refused under the v1 schema |
| Native trainer, bench, HU200, K50 evaluation and save/stop suites | 56 pass, 1 optional skip |

## Cost

HU100, seed 11, opponent-sampled, 20M nodes on the M1:

| | v1 | v2 |
|---|---:|---:|
| Iterations | 14,555 | 4,753 |
| Nodes per iteration | about 1,400 | about 4,200 |
| Entries | 4,908,887 | 8,560,900 (1.74×) |
| Training time | 10.5 s | 9.8 s |

Speed per node is unchanged in this small comparison, but each iteration's tree is about three times larger, because the traverser explores every action at its decisions. At equal nodes in that witness, v2 runs about a third as many iterations and stores about 1.7× the keys. Those ratios are not a reliable large K50 capacity forecast: #260 reaches 97,290,578 entries at only 332,894,788 nodes under its measured fixed cap.

The first capped K50-v2 run reduces the pot-pressure contrast from #254's −85.00 to **−17.85 [−60.10,24.41] BB/100**, descriptively closing **79%** of the earlier point gap. Complete removal is unproved, and the model fails the v0.5.3 candidate rule because tight- and loose-aggressive contrasts are severe regressions. It beats both v0.5.2 and K50 v1 1B directly, while training at roughly one third of the latter's node budget. Whether v2 learns as well at matched nodes or visits remains open; this single capped model cannot answer that question. [Full audits, frozen contrasts and limits](hu100-k50-v2.md).

## HU200 at the fixed 5B budget

[#270](hu200-v2-frozen-5b.md) compares frozen v2 and v1 at 5B cumulative
nodes, the same seed/R=1/K50/linear-CFR/opponent-sampled recipe and 11 GB budget.
The paired 1.5× pot gain is **+238.59 [201.61,275.57] BB/100**, bringing v2 to
**−1.83 [−23.96,20.30]** absolute. It nearly removes that large loss, but
half-pot worsens **−32.31 [−59.14,−5.48]** and tight/loose/matched 1× pot
fail the predeclared regression limit. The Slumbot rule fails; no confirmation
preparation or live play follows. All 720,896 final hands replay/reproduce.

This answers the fixed-budget HU200 case, with substantial tradeoffs. It does
not settle a fully dense v2: stored visits/key are **2.80 versus v1's 9.73**,
with 42.55% zero visits and frequent late training misses. Matched-visit or
more densely trained HU200 v2 and general off-menu strength remain
open questions. No longer run, release or default change is scheduled.

## HU100 frozen continuation to5B

[#272](hu100-v2-frozen-5b.md) freezes #260's exact 97,290,578-key table and
continues to 5,000,006,023 nodes. The pressure fix survives: versus released
v0.5.2 translation-on, the paired contrast is **+61.76 [44.42,79.11]
BB/100**, with frozen v2 itself **+27.24 [15.27,39.21]**. Fresh paired
learning gains over the unfrozen terminal are **+66.95 [52.99,80.91]**
against tight and **+276.92 [247.26,306.59]** against loose. The severe
aggressive-bot losses are removed under the predeclared rule; smaller
release deficits remain. Matched 1× pot is **−26.57 [−40.79,−12.34]**,
passing the requested upper-bound safeguard without proving non-inferiority.
No primary upper bound is below −20 and pressure's lower bound exceeds −20:
**recommend this fixed-seed model as a v0.5.3 candidate; owner decides**.

This answers the requested HU100 frozen 5B case, with all 1,179,648 final
hands independently replayed and exactly reproduced. Visits/key rise from
0.66 to 5.27; zero visits fall to 32.70%. Played v2 river misses are 8.63%,
but full-continuation river training misses remain 64.10% traverser /
56.66% opponent. Deep support is better on reached hands than across the
counterfactual training tree; fully dense original-game convergence,
independent-seed confirmation and general off-menu strength remain open.
No release or default change.

## Use

```
hu20-trainer train --stack-bb 100 --menu v2 ... --out CHECKPOINT
hu20-trainer parity FIXTURES.jsonl --menu v2
python -m scripts.native_parity_fixtures --stack-bb 100 --menu v2 --out FIXTURES.jsonl
```

In Python, `schema_choices(view, schema)` gives the menu a native schema was trained with, v1 or v2. Harnesses that build menus with `choices` directly need `menu_sizes=menu_sizes(schema)`.

## Not yet covered

- Translation (`hu100-public-menu-translation-v1`), turn search and the web runtime are v1-only.
- Windowed extraction, subgame-bench exports and historical HU100/HU200 support-witness diagnostics remain v1-only and explicitly refuse v2. Native bench/traversal/run parity reject `--menu`; their fixture formats remain v1.

[PR #258](overbet-correctness.md) adds schema-derived Python reference training and direct/scripted/pot-pressure replay, including mixed v1/v2 matches and retained baseline reuse across schemas. Independent replay follows pinned reuse ancestry to the original baseline's schema and tables. Fixed v1 harness streams and small native HU20/HU100 v1/v2 checkpoint/export/resume files match main exactly.
