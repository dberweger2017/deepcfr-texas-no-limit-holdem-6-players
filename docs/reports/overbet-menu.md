# Overbet menu: half-pot, 1.5× and 2× pot raises

[#251](hu100-pot-pressure-diagnosis.md) found bet size to be the main reason HU100 releases lose to the scripted pot-pressure opponent: when it raises 1× pot instead of 1.5×, v0.5.2 gains +72.36 [58.58, 86.13] BB/100. Our menu offers min-raise, pot and a conditional all-in, so overbets never appear in training. This adds a second menu with **half-pot, 1.5× pot and 2× pot** raises, for the native trainer and the Python key. It is engineering only: no model is trained with it yet, and v1 stays the production menu.

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

Speed per node is unchanged, but each iteration's tree is about three times larger, because the traverser explores every action at its decisions. At equal nodes, v2 therefore runs about a third as many iterations and stores about 1.7× the keys. Whether it learns as well per node is the open question for the first v2 training run. HU100 v2 at 1B should need roughly 65–70M entries, which fits the M4; K50 with v2 reaches the M4's cap sooner.

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

[PR #258](overbet-correctness.md) adds schema-derived Python reference training and direct/scripted/pot-pressure replay, including mixed v1/v2 matches. Fixed v1 harness streams and small native HU20/HU100 v1/v2 checkpoint/export/resume files match main exactly.
