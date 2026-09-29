# Postflop search on the finished checkpoint 0.4 blueprint

## Decision

The bounded range-aware search player materially improved the **same frozen 58,015,659-entry blueprint** on both new paired six-player validation schedules. It earned **+245.64 BB/100 paired over direct blueprint play** against random opponents and **+236.34 BB/100 paired** against the scripted pool; both 95% intervals clear zero. All 1,923 attempted postflop searches completed under the one-second decision limit, with no fallbacks or invalid actions. Search-only p95 latency was 0.201 seconds against random and 0.154 seconds against scripted opponents. The 64-GB RunPod pod sufficed: peak process RSS was 45.49 GiB, below the 50-GiB guard.

This supports keeping search as the next playing-policy candidate. It does **not** establish a scripted-pool win: searched play remained −167.46 BB/100 on that schedule, with a 95% interval spanning zero. The comparison uses validation deals, one blueprint seed, and an approximate sampled rollout, not Pluribus's multiplayer subgame solver or a professional reference benchmark.

## Paired result

Checkpoint SHA-256: `1b7d9ef0a6f111ac82d99f802cf7ac7bc5c8685ff6a2f214cb918f7ef9361bc2`; iteration 99,646. The implementation revision was `e50346f552366a87408c5c741278e726b1e83b56`, clean at execution. The two arms used identical fixed deal/seat/opponent schedules and separate candidate action RNGs. The random comparison completed 3,072 hands per arm; scripted completed 1,536 per arm.

| Opponents | Direct blueprint BB/100 (95% CI) | With search BB/100 (95% CI) | Paired search gain BB/100 (95% CI) |
| --- | ---: | ---: | ---: |
| Random | +98.18 [−121.96, +318.32] | +343.82 [+120.26, +567.38] | **+245.64 [+141.04, +350.24]** |
| Scripted pool | −403.79 [−569.58, −238.01] | −167.46 [−356.16, +21.25] | **+236.34 [+60.63, +412.04]** |

| Opponents | Search attempts/completed/fallbacks | Search-only p50/p95/max | All candidate decisions p95 | Invalid actions |
| --- | ---: | ---: | ---: | ---: |
| Random | 716 / 716 / 0 | 0.093 / 0.201 / 0.257 s | 0.112 s | 0 |
| Scripted pool | 1,207 / 1,207 / 0 | 0.083 / 0.154 / 0.240 s | 0.138 s | 0 |

Search ran on 539/135/42 flop/turn/river decisions against random, and 648/357/202 against the scripted pool. All 9,216 total arm hands finished valid. The comparison itself took about 3.5 minutes after roughly 9.5 minutes to hash and load the checkpoint. Peak process RSS was **48,842,797,056 bytes**. No wall, RSS, or pod OOM guard fired.

## Interpretation and artifacts

The direct blueprint still falls back to uniform abstract actions on most postflop histories. Search's positive paired result on both fresh validation schedules is therefore a useful playing improvement even though its sampled ranges and continuation styles are approximate. The scripted absolute result remains weak and uncertain, so a further independent evaluation or a stronger search/training method is needed before a v0.5 strength claim. This experiment does not retrain the blueprint.

The M4 retains the hash-verified archive at `~/Local/blueprint-04-backups/final/search-99646.tar.gz`, SHA-256 `7bd8e51151f57e9c4cac72d62c6422540a51607904ce6c5852d0c253720e14ab`. Its `checksums.json` verified all six comparison files, including 9,216 compact per-hand rows, both reports, the manifest, and the result. The final checkpoint and its one-time sealed random test are described in the [checkpoint-0.4 report](blueprint-checkpoint-04.md).
