# HU200 frozen 5B overbet-menu comparison

[PR270](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/270)
· [Predeclared protocol](../hu200-v2-frozen-5b-protocol.md)

**Recovery verified; training and evaluation in progress.** This experiment asks
whether representing off-menu bet sizes and training the retained keys more
densely improves HU200 against those sizes. Only the menu changes from #250/#265.
The primary evidence is the paired seven-opponent panel. Different-menu direct
play is descriptive because v1 can encounter histories it never trained.

## Recipe copied from #250/#265

| Value | Fixed recipe |
|---|---|
| Game | Heads-up, 200 BB, 20,000 chips per seat |
| Blinds /rake /ante | 50/100 chips /0 /0 |
| Training seed | 2026101302 |
| Roots per seat | 1 |
| CFR weighting | Linear, no floor/DCFR/training options |
| Lifetime averaging | Opponent-sampled |
| Cards | #163 pinned K50 flop/turn/river |
| Memory budget | 11,000,000,000 bytes |
| Treatment menu | v2: min, half, pot, 1.5-pot, 2-pot, conditional jam; schema-owned v2 history |
| Freeze | Cap-reaching complete iteration; update retained keys, uniform missing decisions |
| Node target | 5B cumulative, including pre-freeze growth |
| Export fallback /evaluation translation | Uniform zero mass /off |

The [first report](hu200-k50-first.md) and its
[exact input artifact](hu200-k50-first-artifacts/inputs.json) pin these tables:

| Table | Bytes | SHA256 |
|---|---:|---|
| Flop | 12,867,944 | `084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111` |
| Turn | 139,600,524 | `c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f` |
| River | 1,231,562,564 | `70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629` |

#250's historical v1 cap was
`floor((11,000,000,000−1,400,000,000−500,000,000)/(96.9×1.1)) = 85,373,862`.
Its terminal overshot by 45 entries: 752,593,175 nodes, 493,496 iterations,
85,373,907 keys, 1.7446 visits/key, 44.29% zero visits.
[#265](hu200-k50-frozen.md) reused that exact table and recipe, reaching 5B
at 5,000,000,031 nodes /3,748,464 iterations, 9.7324 visits/key and 26.41%
zero visits. Its average is 2,756,501,598 bytes, SHA256
`43aa3b3cd3c7365c5be1b177dcc8a7c04e01c8470403c38327aa845922726512`.
That frozen 5B average is the fixed comparison, rather than the 20B final.

## Build, exact recovery and measured cap

Native source is current-main base `e2bcdc63`; research orchestration is
`bbfa6a1df1fc70f323b1335ccf2146804d0dcc30`. No native/game/regret changes.
The path-independent M4 build is 1,492,512 bytes, SHA256
`4f6abcbb954c3c744448c2a3542ab3164e0e1a015cb05ecebeda077525991cd4`,
matching #265. [Build receipt](hu200-v2-frozen-5b-artifacts/build-receipt.json).

Fresh recoverable 20M and 10M→20M are byte-identical:
**20,001,042 nodes /2,299 iterations /10,427,061 entries**, 343,463,165 bytes,
SHA256 `d0ab3813e65001cc331058cb6bdeccaf878c619fe2ffae4073ff1a468dab5b52`.
The fresh native segment took 34.75 seconds including its 29.30-second save;
resume took 32.83 including 30.00 seconds saving. Table load precedes that timer.
[Recovery and all memory measurements](hu200-v2-frozen-5b-artifacts/gates-complete.json).

Native process RSS was sampled every 50 ms for the gate, independently of the
five-second campaign guard. Fresh train peak: **2,307,260,416 bytes**; save peak:
**2,311,684,096 bytes**. As in #260, reserve the fixed table bytes separately
and retain all other overhead in the per-entry measurement:

```
B/entry = (max(2,307,260,416, 2,311,684,096) − 1,384,000,000) / 10,427,061
        = 88.96889507024079
cap = floor((11,000,000,000 − 1,384,000,000 − 250,000,000) / (88.96889507024079 × 1.1))
    = floor(9,366,000,000 / 97.86578457726488)
    = 95,702,497 entries
```

This cap is fixed before main training. RSS is a sampled planning measurement,
not a promise that all future frozen-table regret arrays fit the same footprint.

## Predeclared reading rule

Recommend Slumbot confirmation only if the paired 1.5× pot contrast's nominal
Student-t 95% lower bound exceeds zero and no panel contrast upper bound is below
−20 BB/100. Half-pot is the other key contrast. The direct match cannot decide
this rule. Cost-only pilots precede a posted canonical count plan; every final
hand independently replays and exactly reproduces before outcome readout.

The Slumbot shared loader previously forced v1 and its admission list refused
v2. This PR selects an admitted HU200 schema from the average header, then
verifies the complete hash/header/table identity through AveragePolicy. Both
play and analysis use that loader; HU100 and mismatched-card models remain
refused. A real synthetic K50 v2 model passes load, warm read-only cache,
Slumbot admission and recorded-hand analysis; corrupt hashes are rejected.
These are offline checks, with no account, credential or service calls.
