# Native trainer: compact training table

HU100 training stopped at 39.4M nodes because the native table's memory per entry was large and unpredictable ([#204](native-hu100-growth-50m.md): 425 bytes/entry, up 18% on its pilot). The trainer now stores its table compactly. **Peak memory falls 4.5× at #204's endpoint (3.17 → 0.71 GB) and grows in proportion to entries.** Checkpoints are byte-identical, so every earlier result and recovery chain still holds.

## What changed

`native/hu20-trainer/src/store.rs` replaces the `HashMap<Key, Node>` shards:

- **Actual-length values.** A `Node` held fixed 5-slot regret and average arrays (96 bytes plus its 16-byte key per slot), whatever the menu's length. HU100's mean menu has 3.1 actions. Entries now keep a 40-byte record and their values at the menu's length.
- **No regrets for opponent-only keys.** Under opponent-sampled averaging, 52% of HU100's keys have only been reached as the sampled opponent. Their regrets are exactly +0.0 until a traverser visit, so they store none. Any regret whose bits differ from +0.0, including −0.0, is stored.
- **No doublings.** Records and values live in fixed-size chunks, and the index holds 32-bit positions. A hash map's doubling left up to half its slots empty and briefly held both tables. Here growth never copies existing entries.
- **Sorted saves by shard.** Shards split keys by their first byte, so a save sorts one shard at a time instead of building and sorting a reference to every entry.

Training arithmetic, update order, key and menu semantics, recovery and file formats are unchanged.

## Exactness

| Check | Result |
|---|---|
| Fresh HU100 run to #204's capacity stop (seed 2026100601, opponent-sampled, `--max-entries 7642767`) | Checkpoint SHA256 `792a675c…f8d416`: identical to #204's archived checkpoint, and to main's binary |
| HU100 to 20M nodes, then `--resume` to the same stop | Same `792a675c…f8d416` |
| HU20 v0.4.1 recipe to 1B nodes, then `export --zero-mass uniform` | Average SHA256 `571e1982…6b74d`, 142,677,367 bytes: the pinned v0.4.1 release average |
| Rust tests | 11 pass, including two new store tests: exact storage, −0.0, lazy regrets, duplicate rejection, 50,000-key growth and sorted iteration |
| Python suites that run the native binary (`test_native_*`, compact policy, streaming audit, checkpoint diagnosis) | 117 pass |

## Memory

Peak memory footprint (`/usr/bin/time -l`) of one `train` process, including its final save, on HU100 with the production recipe:

| Nodes | Entries | Main | Compact |
|---:|---:|---:|---:|
| 1,001,382 | 460,885 | 142 MB (307 B/entry) | 49 MB (105) |
| 5,001,210 | 1,777,063 | 422 MB (237) | 157 MB (88) |
| 10,001,922 | 3,023,624 | 926 MB (306) | 265 MB (87) |
| 20,001,470 | 4,937,867 | 1,686 MB (341) | 435 MB (88) |
| 39,438,279 | 7,643,261 | 3,173 MB (415) | 709 MB (93) |
| 200,000,077 | 19,242,803 | not run | 1,701 MB (88) |

Main's bytes per entry swing with each hash-table doubling, which is why #204's forecast from its pilot missed. The compact table's payload is a 40-byte record, 8 bytes per action for the average, and 8 per action for regrets once a key has any: 77 bytes/entry at #204's endpoint. It approaches about 90 as keys gain traverser visits, and the index adds roughly 5–10. **Forecasting with 110 bytes/entry plus 100 MB is conservative** at every measured size. No doubling allowance is needed.

Entries keep growing more slowly than nodes:

| Nodes | Entries |
|---:|---:|
| 50,003,237 | 8,856,580 |
| 100,001,959 | 13,291,612 |
| 200,000,077 | 19,242,803 |

The local growth exponent fell from 0.71 (10M → 20M nodes) to 0.53 (100M → 200M). Holding it at 0.53 projects about 45M entries at 1B nodes, roughly 5 GB with the compact table against about 19 GB with main's. This is an extrapolation, not a measurement. A 1B campaign should still stop on measured memory.

## Limits

- **Speed:** training speed is within 4% of main's at 39M nodes (25.6 s against 24.5 s). Throughput falls as the table outgrows the caches: 1.4M nodes/s at 50M, 0.78M nodes/s at 200M on the M1.
- **Saves:** a save writes about 40 bytes/entry of gzip-compressed JSON, 774 MB at 19.2M entries.
- **Scope:** this changes memory only. It doesn't change the recipe, the abstraction or the menu, so HU100's unsupported off-menu histories (#201) remain.
