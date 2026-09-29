# M4-only uncapped HU20 scaling recovery

Draft [PR #116](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/116) resumed the three retained, independently seeded uncapped HU20 checkpoints from the failed dual-Mac attempt. This was a separately authorized M4-only run with its own fixed 10-hour clock, from 2026-09-29 10:08:33 to 20:08:33 UTC. The previous failure, partial artifacts and deadline remain recorded in the [dual-Mac report](hu20-scaling-both-macs.md). The M4 source stayed frozen at `874ba641fc6110a2d0998af986608633abef8898`; the model, betting menu, card and history abstraction, K1, current-policy extraction, LBR work budget and paired schedules were unchanged. M1 performed only small-file transfers, edits, Git and status reads.

**Outcome:** all three lineages reached the fixed 100M total-node checkpoint. The two prespecified primary comparisons completed, were natively replayed, and passed their gates. The overall campaign is **incomplete**: 73,728 of 608,256 planned confirmation hands were played, and 99 diagnostic panels remain pending because their outcome-free time forecast exceeded the original deadline reserve. No model was promoted or merged.

## Realized playing results

Results are target profit in BB/100. Each contrast averages the three seed differences *within* an independent paired deal/rotation block. Intervals for the primary contrasts are two-sided 97.5% block-clustered intervals. Each LBR panel used 2,048 paired two-position blocks per seed and checkpoint; each native-pressure panel used 4,096. The target played its saved policy during the hand. LBR's checkdown estimate selected the attacker's actions and was not imposed on the target.

| Fixed attacker | 20M absolute | 100M absolute | 100M minus own 20M | Gate |
| --- | ---: | ---: | ---: | --- |
| Original-cap2 local best response | −99.08 | −73.14 | **+25.94 [7.43, 44.45]** | Lower bound above 0: pass |
| Native-legal pressure | +106.70 | +126.09 | **+19.39 [2.86, 35.93]** | Lower bound above −10: pass |

The 100M policy still loses **0.731 BB/hand**, or **3.66 20BB buy-ins/100**, against this bounded LBR. Its positive *improvement* is meaningful for the declared comparison, but it does not establish robustness, exact exploitability, or broad poker competence. Native-pressure profit is positive on average, but the seed-specific contrasts differ. The baseline controls and mid-training strength curve were not run in this attempt.

| Seed | LBR target 20M → 100M | LBR change | Pressure target 20M → 100M | Pressure change |
| --- | ---: | ---: | ---: | ---: |
| 2026093001 | −99.99 → −67.94 | +32.04 | +119.56 → +108.18 | **−11.37** |
| 2026093002 | −108.39 → −75.31 | +33.08 | +84.19 → +135.48 | +51.29 |
| 2026093003 | −88.85 → −76.16 | +12.70 | +116.35 → +134.61 | +18.26 |

The first seed's native-pressure regression is retained. For the 100M-minus-20M LBR contrast, the big-blind estimate is +28.58 [2.69, 54.47] BB/100 and the button/small-blind estimate is +23.30 [−3.13, 49.73]. The corresponding native-pressure role estimates are +35.05 [11.75, 58.35] and +3.74 [−19.61, 27.09]. These role intervals are descriptive; the prespecified gates use the role-balanced contrasts above. Full per-seed, role and attacker profits are in [summary.json](hu20-scaling-recovery-artifacts/summary.json).

## Work, coverage and limits

| Seed | Resumed from | Completed | Overshoot | Entries at 100M | Independent preflop / flop / turn / river trained coverage |
| --- | ---: | ---: | ---: | ---: | --- |
| 2026093001 | 40,000,075 | 100,000,029 | 29 | 1,496,914 | 100% / 99.9% / 90.9% / 56.2% |
| 2026093002 | 34,291,306 | 100,000,280 | 280 | 1,494,799 | 100% / 100% / 96.0% / 61.1% |
| 2026093003 | 20,000,268 | 100,000,083 | 83 | 1,499,679 | 100% / 99.8% / 93.6% / 65.6% |

The independent fixture has 4,248 observations per checkpoint. At 100M, its median river key has only 1–2 visits across the three seeds, versus 701–728 preflop and 1,643–1,719 flop visits. This is independent public-state density, not own-trajectory coverage or a value-quality certificate. The own-training recovery trajectories did reach the river: their attempted decision-node counts there were 17.56M / 19.05M / 22.93M by seed, but that work spread over many keys. New recovery keys were concentrated on the river (355,101 / 402,786 / 563,117 by seed); the corresponding turn counts were 117,658 / 134,255 / 194,964. Full per-street work and revisits remain in the retained `iterations.jsonl` files. Successful completed traversals were published; no failed unpublished traversal is recorded in these three successful recovery results, and there is no separate `discarded_nodes` field on their success path. The failed original dual-Mac attempt retains its own discarded-work accounting.

The native audit replayed all **73,728 hands**, verified legal actions and chip settlement, target RNG, concrete menu/key/visit membership and event digests, with **zero audit failures**. It does not independently recompute every internal LBR action-value estimate. Across the six LBR panels, all **52,433 attacker decisions completed** within the soft 5-second budget, with no recorded over-budget or zero-likelihood event; the largest per-decision time was 0.573 seconds. This shows that the bounded attack executed, not that it is a full best response. Fresh confirmation deals had zero overlap with 268 prior opened files. Independent arithmetic from raw chips reproduced both primary point estimates and intervals.

Target lookup and menu exposure changed with the played trajectories. Against LBR, fallback decisions fell from 241 / 28,129 at 20M to 41 / 28,549 at 100M. Against native pressure, they fell from 1,078 / 71,040 to 498 / 84,351. About 21,065 target decisions at 20M and 23,239 at 100M followed a history outside the *original cap2* menu in the native-pressure panels; the target's uncapped menu itself recorded no preceding off-menu event. Those counts do not attribute final chip profit to a single street or prove that a lookup miss caused a loss. The per-action street, trained/fallback and menu-history coordinates are retained in `audit-primary/results.json` on M4.

The sealed M4 inventory covers **215 files**. Verification checked 766 staged runtime inputs, 49 training-hash links, all three parent and milestone lineages, and the fresh-deal separation. Peak sampled owned-job RSS was **3.23 GiB** against a 10.5-GiB cap, minimum sampled free disk **38.83 GiB** against an 8-GiB floor, measured swap growth **0** against a 0.5-GiB cap, and all resource samples were on AC. The coordinator, primary evaluation, native audit, report and final seal all exited successfully by about 18:44 UTC, roughly 8 hours 35 minutes after the original clock began and before the unchanged 20:08:33 UTC cutoff. No paid host was used. The initial wrapper shell-quoting failure did zero training and remains in `wrapper-attempt-1`; it was repaired under the same deadline. The prior dual-Mac failure remains a separate, unchanged attempt.

## Retained artifacts and reproduction

The compact [summary](hu20-scaling-recovery-artifacts/summary.json) and [global manifest](hu20-scaling-recovery-artifacts/final-manifest.json) are committed. The latter records each sealed file's size and SHA-256. Large checkpoint, per-hand, audit, recovery and log files remain on M4 at:

`/Users/dberweger/Local/hu20-training-scaling-pr116/results/hu20-scaling-m4-recovery-20260929-1008`

```sh
rsync -a m4:/Users/dberweger/Local/hu20-training-scaling-pr116/results/hu20-scaling-m4-recovery-20260929-1008/ ./hu20-scaling-m4-recovery/
scp m4:/Users/dberweger/Local/hu20-training-scaling-pr116/results/hu20-scaling-m4-recovery-20260929-1008-final-manifest.json ./
```

The fixed-first-seed experimental human-play smoke and 20-hand replay passed, with bot cards hidden until legitimate disclosure. To play that same hash-pinned candidate on M4:

```sh
python -m scripts.play_hu20_native --policy /Users/dberweger/Local/hu20-training-scaling-pr116/results/hu20-scaling-m4-recovery-20260929-1008/training/B-2026093001/current-100000000.json.gz --sha256 4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf --history results/recovery-human.jsonl
python -m scripts.play_hu20_native --replay results/recovery-human.jsonl
```

The #112/#113/#115 playable artifacts and commands remain available.

## Recommendation

**Finish the already specified control and intermediate-checkpoint diagnostics before choosing a new training recipe.** The fixed 100M comparison shows a positive aggregate improvement but leaves a severe absolute LBR loss, one native-pressure seed regression and 99 panels pending. Those measurements can test whether the gain generalizes across opponents and whether 40M/80M work was useful. This is a recommendation for review and separate authorization; no follow-on run is scheduled by this report.
