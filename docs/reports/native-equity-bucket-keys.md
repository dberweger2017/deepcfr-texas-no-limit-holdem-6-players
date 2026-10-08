# Equity-bucket information keys in the native trainer

Step 2 of the [abstraction plan](hu20-abstraction-lessons.md#planned-sequence): the native trainer and the Python key function can now key cards by #163's K=50 equity buckets, which [#190](hu20-global-bucket-validation.md) validated. This is engineering only. No bench comparison or full-game training has been run, and v1 stays the production schema.

## The schema

`hu20-native-reopening-ordered-history-equity-k50-v1`, and `hu100-…-equity-k50-v1` for heads-up 100 BB, change only the postflop card part of the key. It becomes the holding's bucket on the current street in #163's flop, turn or river K=50 table. Everything else is v1's (rule 3): preflop's 169 classes, the seat flags, the ordered size-bucketed history and the menu. Buckets depend only on cards, so both games use the same tables.

Each schema name stands for one exact table set. Production loads in both languages refuse any table whose SHA256 differs from #163's build:

| Table | Bytes | SHA256 |
|---|---:|---|
| `flop-k50.bin` | 12,867,944 | `084e243d…b572c111` |
| `turn-k50.bin` | 139,600,524 | `c0bc6a8a…ae9f168f` |
| `river-k50.bin` | 1,231,562,564 | `70c1cb5e…83977629` |

Checkpoints record `card_descriptor: equity-histogram-k50-v1` and the three table hashes in their identity. Resume refuses a checkpoint whose tables differ from the ones supplied, including a v1 resume of a bucket checkpoint. Bench exports name the schema and tables, and their groups use the metric `equity-k50`.

## Use

```
tar -xzf ~/Local/Research-Cloud/PR-163-equity-buckets/hu20-equity-buckets-20261005.tar.gz --strip-components 1 \
    hu20-equity-buckets-20261005/flop-k50.bin hu20-equity-buckets-20261005/turn-k50.bin hu20-equity-buckets-20261005/river-k50.bin
hu20-trainer train … --card-buckets DIR
hu20-trainer bench-train … --card-buckets DIR
```

Extract into an ignored, non-synced directory and check the hashes above. In Python, call `equity_buckets.register(equity_buckets.EquityCards(DIR))`, then key with `schema=HU20_EQUITY_SCHEMA`. `SubgameTrainer(…, schema=HU20_EQUITY_SCHEMA)` is the bench's Python reference.

## Exactness

| Check | Result |
|---|---|
| Rust/Python key parity with #163's tables: 2,000 HU20 hands (14,377 decisions) and 1,000 HU100 hands (7,288 decisions) from the Python engine, including off-menu sizes | 0 mismatched hands |
| Same, in CI, on synthetic tables covering the fixture hands (HU20 and HU100, 300 hands each) | 0 mismatched; every key differs from v1's, and every menu is unchanged |
| Native `bench-train` against Python `SubgameTrainer`, equity schema, two real #149 turn roots, 200 iterations | Identical node counts and all three exports |
| Fresh 4M-node equity run against one resumed from its 2M checkpoint | Identical rows |
| v1 production paths | #204's checkpoint reproduced byte for byte (`792a675c…`); all earlier native parity suites pass |
| Rust tests | 12 trainer tests, including checkpoint table identity, and a bucket-table round-trip and corruption test |

## Cost

Measured on the M1 with #163's tables, HU20 production recipe:

| | v1 | Equity K50 |
|---|---:|---:|
| Entries at 10M nodes | 896,001 | 1,248,056 (1.39×) |
| Entries at 100M nodes | 2,173,888 | 3,039,462 (1.40×) |
| Speed at 100M nodes | 1.35M nodes/s | 1.17M nodes/s |
| Bench keys after 100k iterations, 20 turn roots | 50,096 | 67,036 (1.34×) |
| Bench speed | 10.6k iterations/s | 8.8k iterations/s |

The tables add about 1.4 GB to each process. Bucket keys multiply the key count by about 1.4. Card v2 multiplied it by 7 ([#143](hu20-card-v2.md)), so matched visits per key costs only about 1.4× the nodes here.

## Not yet covered

The bench, native training, resume and export are covered. Before any full-game bucket model is played, these still need the schema (rule 6):
- the Python checkpoint and average loaders (`load_training`, `AveragePolicy`);
- the web runtime;
- turn search, which takes ranges from blueprint keys;
- LBR.

That work is only worth doing if the trained bench passes (step 3).
