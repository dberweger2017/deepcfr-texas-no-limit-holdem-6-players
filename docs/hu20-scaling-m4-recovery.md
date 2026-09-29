# Authorized M4-only recovery of #116

## Attempt and deployment

The failed [dual-Mac attempt](reports/hu20-scaling-both-macs.md) remains failed and unchanged. The owner separately authorized this recovery, including staging, tests, preflight, training, evaluation and reporting, with one ten-hour absolute clock.

Attempt `hu20-scaling-m4-recovery-20260929-1008` starts **2026-09-29 10:08:33 UTC (12:08:33 Madrid)** and stops **20:08:33 UTC (22:08:33 Madrid)**. The [frozen plan](../configs/blueprint/hu20-scaling-m4-recovery.json) resolves every input on M4. The coordinator, sequential workers, watchdogs, audits and report writer all run there. M1 is only used for small-file edits, transfers, Git and brief status reads; it need not remain awake or connected. A detached twelve-second probe completed after SSH disconnected.

All 766 runtime inventory entries were verified on M4 for local path, readability, size, SHA-256 and applicable parsing/schema. These cover retained checkpoints, original A/B20 controls, the exact independent-observation fixture, source/schedule generators and reference audit files. The versioned runtime path mapping and verified input receipts stay in the attempt root. The former M1 seed-2 directory was transferred without repacking. Original directories and historical raw discarded-work fields remain untouched.

## Resume and production checks

Training resumes in fixed seed order from the audited cumulative states:

| Seed | Retained lifetime nodes | Checkpoint SHA-256 |
| --- | ---: | --- |
| 2026093001 | 40,000,075 | `2a47a14f77cc9c30b51daaaa7bb9d10f994199db798dc597dbbeeadaf2f3e2f1` |
| 2026093002 | 34,291,306 | `bd54de118c87712b2bb36272aeffc014c6a766fec6b7fdbd9f08db2766b5adf8` |
| 2026093003 | 20,000,268 | `aa65e5633cc87c7d23bcb3be871813f3413120a256fe38ebdb2383215acf5afd` |

The target remains **100M total lifetime completed nodes per seed**, preserving regrets, cumulative iterations/linear weighting, K1, uncapped HU20 game, card/history keys, sizing, extraction and fallback. Seed 1's exact existing 40M checkpoint/export supplies a labeled recovered milestone with independent density and no new learning step. Seed 2 and seed 3 publish normal 40M/80M/100M milestones. Each publication stage has a separate retained completion receipt. A post-publication diagnostic failure discards zero completed nodes; an unpublished interrupted traversal retains its actual discarded work.

**32 focused tests passed on M4.** Missing/wrong fixture tests reject before training; tests exercise production publication, zero-step milestone recovery, native tiny evaluation/replay and report arithmetic. The deployed preflight ran the real save/export/density/record/reload path on disposable continuations of all three parents. Deterministic next-four-iteration hashes matched original and reloaded copies for every parent. These validation copies are not independent seeds or additional main work. Resource-only evaluator deals are separate from confirmation and suppress returns. Preflight took 210.35 seconds and peaked at 2.58 GiB process RSS. [Compact evidence](reports/hu20-scaling-recovery-preflight/compact-summary.json) and [forecast](reports/hu20-scaling-recovery-preflight/resource-forecast.json) are retained; the operational repair passed full GitHub CI.

## Frozen resource staging before outcomes

The M4-only forecast covers **205,708,351 remaining nodes**, artifact costs, all evaluation blocks, native replay and reporting. Conservative remaining projections are 4.99 hours training, 5.13 hours primary evaluation/audit, 0.96 hours diagnostics, plus reserves: **11.91 hours**, exceeding 9.63 hours available at freeze. It therefore does not establish that the entire campaign can finish within the authorized window.

The fixed phase order prioritizes all three final checkpoints, then the unchanged final-versus-own20M **original-cap2 LBR primary and native-pressure safeguard**, then their native audit. Other fixed controls, checkpoint curves and secondary opponents run only if their predeclared resource forecast plus audit/report reserves fits the remaining clock. Otherwise they are explicitly pending. No seed, scientific budget, block count, deal, comparison or LBR work limit is reduced. The unopened 608,256-hand schedule retains its block IDs and streams; every assigned whole block now runs on M4.

Training stops by **17:08:33 UTC**, evaluation by **19:18:33 UTC**, with a final reporting/sealing reserve before the absolute cutoff. A failed or unfinished phase retains attempts, rows, pending targets/blocks and partial checkpoints. Complete 97.5% block intervals average the three lineage contrasts inside each paired rotation block: the primary lower bound must exceed zero, and the native-pressure safeguard lower bound must exceed −10 BB/100. No gate is declared from missing required blocks. Weak or limited LBR is not evidence of robustness.

All heavy work is sequential with a 10.5-GiB process/owned-job RSS cap, 8-GiB free-disk minimum, 0.5-GiB swap-growth guard, AC requirement and unchanged three-million-entry/iteration bounds. M4 uses caffeinate; M1 sleep settings are untouched. The M4 report writer saves readable evidence and a publication patch without waiting for M1. Final inventory is sealed only after coordinator and supervisor logs close. No model promotion, merge of #116, rental or following campaign is authorized.

## Retention

M4 source: `/Users/dberweger/Local/hu20-training-scaling-pr116`.

The coordinator launched at **10:40:08 UTC** from frozen source `874ba641fc6110a2d0998af986608633abef8898` (coordinator initially PID 31928, detached wrapper 31917). An initial wrapper shell-quoting error occurred before any coordinator or training started. Its original source, owner receipt and log remain under `wrapper-attempt-1`; the repaired wrapper passed compilation before launch. This bounded operational repair used the same attempt and absolute clock. [Launch ownership](reports/hu20-scaling-recovery-preflight/master-owner.json), [validation supervisor](reports/hu20-scaling-recovery-preflight/validation-supervisor.json) and [failed-wrapper log](reports/hu20-scaling-recovery-preflight/wrapper-attempt-1.log) are retained. Subsequent documentation commits do not change the running M4 source. Full operational-repair CI passed **869 tests**.

Attempt root: `results/hu20-scaling-m4-recovery-20260929-1008`. Runtime inventory, path mapping, input verification, phase attempts, checkpoints and audits remain there. Compact reporting and the final inventory will be committed when available; large artifacts remain on M4.

```sh
rsync -a m4:/Users/dberweger/Local/hu20-training-scaling-pr116/results/hu20-scaling-m4-recovery-20260929-1008/ ./hu20-scaling-m4-recovery/
scp m4:/Users/dberweger/Local/hu20-training-scaling-pr116/results/hu20-scaling-m4-recovery-20260929-1008-final-manifest.json ./
```
