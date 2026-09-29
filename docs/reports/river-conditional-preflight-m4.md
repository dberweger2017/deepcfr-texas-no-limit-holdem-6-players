# Conditional river paired-run preflight

The [frozen two-root plan](../../configs/blueprint/river-conditional-preflight-m4.json)
exercised the complete three-arm river evaluator on source `9d99996`, using
the unchanged 12M blueprint. The roots were previously examined development
cases, with one deal each, so their returns are **not** confirmation or
playing-strength evidence. The [manifest, every root/arm row, result and
hashes](river-conditional-preflight-m4/) were copied from the M4 and verified.

| Root | CFR sweeps / solve | Normal control | 8,192-world control | Play status |
| --- | ---: | ---: | ---: | --- |
| Dry check | 687 / 30.001 s | 0.038 s | 25.671 s | 3/3 complete |
| Paired flop | 1,266 / 30.001 s | 0.083 s | 19.352 s | 3/3 complete |

Both compute controls completed all 8,192 worlds without fallback. The
candidate had no off-tree delegation. All six hands were legal and completed.
The whole preflight took 160.92 seconds including one checkpoint load; peak
process RSS was 7.47 GiB under the 10.5-GiB guard, and system swap did not
increase. The report analyzer verified row completeness, pairing, and run
hashes on this output. No winner was selected from these two deals.

At the observed per-root costs, 32 roots with eight paired deals each likely
need a few hours, depending on how often a second hero decision requires
another 8,192-world rollout. The frozen confirmation has a five-hour wall
guard and retains partial rows if that limit is reached. ROADMAP.md reserves
approval of long M4 compute budgets for the owner.
