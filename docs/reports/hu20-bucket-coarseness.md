# Current postflop bucket coarseness on M1

I sampled **3,072 holdings** using the [frozen protocol](../hu20-bucket-coarseness-protocol.md):
64 independent boards per street, 16 distinct compatible own holdings per board.
Flop/turn equity uses 512 uniform compatible opponent/runout worlds per holding;
river enumerates all 990 opponent pairs exactly. Total: **2,062,336 worlds/pairs**.
No models, poker evaluation games, training or M4 computation were used.

The unchanged descriptor is `(category, top_band, flush_draw, straight_draw,
board_paired)`: category 0 high card, 1 pair, 2 two pair, 3 trips, etc.; top bands
are <8, 8–J and Q–A. Draw flags are the existing binary heuristics. Board-paired
means any repeated rank. These fields omit secondary made-hand ranks and kickers.

## Common buckets with wide equity spreads

Shares are within the street’s 1,024 uniformly sampled holdings. Equity is against
an unconditioned uniform compatible HU range, **not the learned betting range**.
I rank common buckets (n≥20) by share × (p90−p10); this is a descriptive priority
score, not a loss estimate.

| Street / bucket | n / boards | Share | Equity p10 / median / p90 | p90−p10 |
| --- | --- | ---: | --- | ---: |
| Flop `(0,2,0,0,0)` | 366 / 55 | 35.74% | .204 / .348 / .476 | .272 |
| Turn `(0,2,0,0,0)` | 150 / 41 | 14.65% | .102 / .249 / .390 | .287 |
| River `(2,1,0,0,1)` | 78 / 20 | 7.62% | .145 / .666 / .818 | .673 |
| River `(1,0,0,0,1)` | 101 / 13 | 9.86% | .045 / .198 / .429 | .384 |
| River `(1,2,0,0,1)` | 73 / 9 | 7.13% | .033 / .251 / .465 | .432 |

The broad Q–A high-card/no-draw bucket is common on flop/turn. The largest common
river spread occurs among two-pair hands on paired boards: top pair rank alone
does not distinguish playing the board from improving the lower pair.
The [complete dashboard](hu20-bucket-coarseness-artifacts/dashboard.md) gives
**all 147 observed buckets** (43 flop, 69 turn, 35 river), including frequency,
share, p10/median/p90, spread, min/max, board count and small denominators.
Unobserved descriptors are unmeasured, not impossible; zero sampled spread in a
tiny bucket is not evidence of strategic homogeneity.

## Concrete same-board collisions

These are the largest selected same-board differences, with explicit selection
bias. River equities are exact uniform-range comparisons, independent of Monte
Carlo noise. Tuples show the full best-made-hand value used by the reference.

| Street / board | Same bucket | Low holding: value; equity | High holding: value; equity |
| --- | --- | --- | --- |
| River `7h Td Ts 2s 2c` | `(2,1,0,0,1)` | `4d 5s`: `(2,10,2,7)`; **.0369** | `7s Kd`: `(2,10,7,13)`; **.7828** |
| River `Td Tc 9s Th 7s` | `(3,1,0,0,1)` | `3h 5s`: `(3,10,9,7)`; **.0652** | `Ad Ks`: `(3,10,14,13)`; **.6192** |
| River `Qh 9s Ad 3d As` | `(1,2,0,0,1)` | `2h 4c`: `(1,14,12,9,4)`; **.0051** | `Ks Th`: `(1,14,13,12,10)`; **.5020** |
| Turn `4h Jh 3s Jd` | `(1,1,0,0,1)` | `2c 7d`: `(1,11,7,4,3)`; **.0879** | `8d Ah`: `(1,11,14,8,4)`; **.5566** |
| Flop `8s 8d 8h` | `(3,1,0,0,1)` | `3h 4d`: `(3,8,4,3)`; **.2686** | `5h Ac`: `(3,8,14,5)`; **.6885** |

In the first river example the weak holding plays the board; the other upgrades
its lower pair and kicker. The trips/pair examples differ in kicker quality while
the category/top band stays fixed. Because both holdings share the exact board,
these examples isolate more than between-board variation. They establish concrete
coarseness, **not a causal explanation for Luna’s loss pattern or a policy error**.

## Future questions, without designing v2

A future abstraction study would need to distinguish own-card contribution versus
playing a paired board, secondary pair rank, kicker quality, and board-relative
pair strength. On earlier streets it should examine draw quality/nut potential
and blockers beyond binary draw flags, together with public texture. It must test
these distinctions against reached-state occupancy and conditional betting ranges,
then measure table capacity, learning and action consequences. This PR does none
of that and changes neither information keys nor training.

Uniform showdown equity is useful for finding collisions but is not betting EV:
fold equity, ranges induced by prior actions, sizing and future decisions matter.
Different equilibrated actions can be reasonable even when equities differ.
Across-board bucket spreads mix public texture with own-card variation. Flop/turn
Monte Carlo SE is at most **.0221 per holding**; selected extremes have additional
noise. Sixteen holdings share each board, so rows are clustered. The reported
p90−p10 is a between-holding spread, **not a confidence interval**. I make no
occupancy, strength or causal claim.

## Evidence and reproduction

[Full holdings](hu20-bucket-coarseness-artifacts/holdings.jsonl) retain exact boards,
cards, seeds, buckets, complete made values, board-playing flags, equities and
method/sample counts. [Summary](hu20-bucket-coarseness-artifacts/summary.json)
retains every bucket and its largest same-board collision. [Checksums](hu20-bucket-coarseness-artifacts/manifest.json)
pin all outputs, with abstraction/ranker source hashes in the summary.

Implementation source `fe0522c`, M1 / Python 3.11.15 / NumPy 1.26.4:
**7.67 seconds**, peak RSS **47.42 MiB**, one worker. The worker exited.
Five focused tests pass: independent river enumeration, royal-board ties,
reference-versus-fast flop/turn worlds, kicker collisions, deterministic replay,
card removal, quantile/share arithmetic and same-board isolation. The previously
verified #126 ranker is reused unchanged, not re-proved.

```sh
python -m pytest tests/diagnostics/test_bucket_coarseness.py -q
python -m scripts.study_hu20_bucket_coarseness \
  --out results/hu20-bucket-coarseness-reproduction
```

Use a new output directory. Root and work bounds are fixed in the protocol; there
is no model path or policy loading option.
