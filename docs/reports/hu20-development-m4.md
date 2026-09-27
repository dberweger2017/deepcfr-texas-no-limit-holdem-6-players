# Heads-up 20BB development and extraction decision

**Draft PR #112, September 27, 2026.** All three from-zero 20M-node K1 runs and the separate development schedule completed. This is a **development** result used only to freeze the playing extraction. The confirmation schedule has not been opened.

The [frozen plan](../../configs/blueprint/hu20-m4.json) has canonical SHA-256 `0da3cb6075d328acd748cf79a7e5f2d2422815c0574fdc3dbf7bb8bbb3908141`. M4 executed source `d718049`. Seeds `2026092801`–`03` completed 20,000,074 / 20,000,259 / 20,000,207 traversal nodes, respectively, with 65,492–67,084 outer iterations and 268,290–270,716 entries. Four checkpoints and eight late snapshots were saved for each seed. All 72 files listed in the three training checksums matched; final checkpoint and extraction-index hashes matched their manifests. Training peaks were 0.366–0.374 GiB process RSS, and system swap stayed at 761.38 MB used. There was no training failure or resource stop.

Development evaluated uniform, three saved early current policies, final current `C`, and eight-snapshot `A` for every seed against six frozen opponents. The 96 arms shared fresh, two-position paired schedules of 256 blocks per opponent: 49,152 completed hands. The machine [development summary](hu20-development-m4-artifacts/development-summary.json), SHA-256 `bcbe99d395531f902f9a999a3f841d004c44b91f6d374f62cb4d2ba2f0272dc2`, found no failed, duplicate or unpaired run. Its contrasts average the three seed differences within each independent deal/rotation block; opponent types have equal weight.

| Saved work point | Mean trained minus same-game uniform, BB/100 | 95% block interval |
| --- | ---: | ---: |
| 2M-node current | +40.13 | [+23.61, +56.66] |
| 5M-node current | +46.96 | [+29.39, +64.52] |
| 10M-node current | +49.97 | [+31.75, +68.18] |
| 20M-node final current `C` | +50.59 | [+32.50, +68.67] |
| 20M-node windowed `A` | +52.51 | [+34.27, +70.74] |

These are conditional chip-profit estimates against the declared scripts and uniform policy, **not full-game exploitability or confirmation**. The increasing point estimates are useful learning evidence; their overlapping intervals do not establish gains between adjacent checkpoints. `A−C` was +1.92 BB/100 [−6.20, +10.04] overall. `A` trailed `C` on each of the five scripted opponents (−1.27 to −6.12 BB/100) and led by +28.26 against the uniform opponent. In sampled `A` trajectories against loose-aggressive, preflop collection was used for about 55–57% of preflop decisions; the remainder explicitly fell back to final current. Almost all reached postflop keys had all eight snapshots, but some were wholly absent. The arithmetic extraction is not an exact reach-weighted CFR average.

**Frozen choice:** the [committed decision](../../configs/blueprint/hu20-extraction-decision.json) selects final current `C` for the untouched confirmation comparison and human-play candidate. Its development effect versus uniform is positive, while the small aggregate `A−C` edge is inconclusive and concentrated in the uniform opponent. The choice also avoids depending on the preflop collector fallback for nearly half the decisions. All three seeds remain in the comparison; no best seed or checkpoint was chosen. `A` and the early policies remain saved diagnostics. No model is promoted.
