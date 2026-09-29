# Dual-Mac uncapped HU20 scaling: retained incomplete attempt

## Result

**The frozen campaign did not complete. No new poker-quality result exists.** Main training stopped at 02:33 Madrid on September 29, before confirmation evaluation. The original hard deadline remained 11:35 Madrid. There was no restart, count change, checkpoint selection, model promotion or paid host. PR #116 remains draft and dependent on unmerged #115.

The M4 reached its first 40M milestone and saved the complete checkpoint and current-policy export. It then raised `FileNotFoundError` while computing independent observation density: `results/hu20-scaling-both-macs-20260929/inputs/independent-observations.jsonl.gz` had not been deployed to the M4. The exact fixture exists on the M1, with frozen SHA-256 `3a8c9ee0c9c6364993ba4011af791c059f816ced2f1a02f449477928c5a35cd2`. This was a deployment omission, not a numerical or resource-limit failure. The coordinator caught the failed worker, sent SIGTERM to its owned M1 training child and retained that child's latest complete state. Seed 3 never started.

Cross-host preflight tested a one-million-node prefix and did not cross the first main milestone, so it did not exercise this missing dependency. CI passed **863 tests**; the **26 focused tests** passed, but those checks did not validate that every frozen external input existed on each deployed host.

## Retained work

| Host / lineage | Original completed nodes | Retained lifetime nodes | Additional completed nodes | Completed additional iterations | Entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| M4 / seed 2026093001 | 20,000,557 | 40,000,075 | 19,999,518 | 52,068 | 1,013,710 |
| M1 / seed 2026093002 | 20,000,238 | 34,291,306 | 14,291,068 | 35,759 | 945,986 |
| M4 / seed 2026093003 | Preserved original 20M | Main continuation not started | 0 | 0 | Original artifact preserved |

Total retained new complete work is **34,290,586 nodes**. Resumed checkpoints preserve original seeds, regrets, cumulative iteration numbers, K1, uncapped menus and the frozen 3M-entry safety bound. Both partial checkpoints reload with the recorded iteration, configuration and entry count. Every iteration row reconciles its cumulative nodes and sequential iteration number to the retained final checkpoint record. The M4 40M export was checked against the checkpoint: every saved action probability matches its current regret-matched probability.

The M4 checkpoint and `partial-last-completed.json.gz` are byte-identical, SHA-256 `2a47a14f77cc9c30b51daaaa7bb9d10f994199db798dc597dbbeeadaf2f3e2f1`. Its exported current policy is `12d3f1ad67d08b350363ce57665347c2de5edc46fab67e474c5f175d6a5271b7`. The M1 partial checkpoint is `bd54de118c87712b2bb36272aeffc014c6a766fec6b7fdbd9f08db2766b5adf8`.

### Explicit accounting correction

The raw M4 result reports `discarded_nodes=731` and `failed_iteration=107360`. Its failure occurred **after** iteration 107359 was published, during milestone reporting. The 731 nodes are that last *completed* iteration's work, not a discarded traversal; no next collection was attempted. Actual discarded M4 nodes are therefore **zero**. The raw result remains unchanged, and the derived audit records both numbers. The M1 SIGTERM interrupted iteration 90469, discarding **28 unpublished nodes**; its checkpoint retains complete iteration 90468. A future implementation should distinguish collection failures from post-collection reporting failures.

The M4 result's milestone list is empty because density reporting failed before publishing its milestone row. That does not erase the complete 40M checkpoint/export. The derived audit explicitly labels that boundary as serialized but diagnostically unpublished; it does not pretend the milestone or run succeeded.

## Resources and time

| Main training resource | M1 | M4 |
| --- | ---: | ---: |
| Worker self-recorded peak RSS | 0.74 GiB | 1.34 GiB |
| Maximum sampled aggregate job RSS | 0.80 GiB | 1.16 GiB |
| Minimum sampled free disk | 61.28 GiB | 40.94 GiB |
| Maximum measured swap growth | 0 | 0 |
| Worker elapsed time | 23.88 min | 23.55 min |
| Resource samples | 284 | 280 |

All main resource samples recorded AC power. Both hosts stayed inside 10.5-GiB RSS, 8-GiB disk and 0.5-GiB swap-growth guards. The worker self-recorded M4 peak exceeds the sampled supervisor peak because polling need not capture every transient; both are retained. Original preflight through worker stop consumed approximately **58.5 minutes** of the ten-hour window. Retention/audit/reporting also use the original clock. Synthetic 3M-entry preflight peaks of 4.93/6.16 GiB remain separate resource measurements, not the main trained-table peaks.

## Quality and diagnostic coverage

- Confirmation: **0 of 608,256 planned hands**. LBR improvement and native-pressure safeguard are **unmeasured**, with no interval to report.
- No cheap milestone profit, role/seed effect, fallback/action latency, conditional exploit or crossplay result was generated by this attempt.
- Independent 40M observation density was not generated. Available new/revisited-key and street/raise-count training work is retained in the machine-readable audit; those are own-training-trajectory diagnostics, not independent coverage or playing strength.
- No 100M candidate exists. No new human smoke was run and no partial model replaces a playable model. #112/#113 and the verified #115 first-seed 20M human command remain available.
- The prior exact checkpoint/current/resume parity and 32 outcome-free evaluator replay hands remain valid preflight evidence; their suppressed returns are not confirmation results or independent new training diversity.

## Audit and retention

[Compact audit](hu20-scaling-failure-artifacts/failure-audit.json) retains both stopped lineages, node accounting, all main resource measurements summarized, phase checksum checks, and own-trajectory street/raise-count counters. [Recovery lineage](hu20-scaling-failure-artifacts/recovery-lineage.json) connects each saved recovery slot to the original checkpoint, source, configuration and starting counters. [Export verification](hu20-scaling-failure-artifacts/export-verification.json) checks the serialized 40M current policy. [Final manifest](hu20-scaling-failure-artifacts/final-manifest.json) lists all retained local files/hashes; [M4 stopped inventory](hu20-scaling-failure-artifacts/m4-stopped-manifest.json) lists the 69 files in the original remote root after workers/logs stopped. All those remote files were copied and hash-verified locally. Per-phase checksum seals, including original preflight and synthetic-memory artifacts, were also verified.

Raw failures are preserved: the initial stale M4 memory-preflight module launch, the duplicate preflight launcher rejected before work, the main missing-file failure, and the coordinator-triggered M1 interruption. Original source is `eda55331298c4dd229bb15235f22060282ddda25`, frozen plan file SHA-256 `50698868306dca11a380dc5b5d80f020dfb0e09b48a02d2cf0077516801c49e0`. The manifest also retains the arena's canonical JSON digest; it is a different digest convention from hashing the serialized configuration file.

Large originals remain on M4:

```sh
rsync -a m4:/Users/dberweger/Local/hu20-training-scaling-pr116/results/hu20-scaling-both-macs-20260929/ ./hu20-scaling-m4-retained/
scp m4:/Users/dberweger/Local/hu20-training-scaling-pr116/results/hu20-scaling-both-macs-20260929-stopped-manifest.json ./
shasum -a 256 ./hu20-scaling-m4-retained/training/B-2026093001/checkpoint-40000000.json.gz
```

M1 originals and the verified M4 copy remain under `/Users/dberweger/.codex/worktrees/blueprint-averaged-extraction/deepcfr-texas-no-limit-holdem-6-players/results/hu20-scaling-both-macs-20260929`. Its sibling `hu20-scaling-both-macs-20260929-final-manifest.json` inventories that retained root. Preserve these ignored artifacts before any worktree cleanup.

## One next recommendation

**Repair and validate external-input deployment before another authorized continuation.** Hash-check every frozen runtime fixture on both hosts before any training starts, and exercise the milestone save/export/density path in deployment preflight. Correct post-collection failure accounting at the same boundary. This attempt cannot choose a poker-learning intervention: it produced no confirmation evidence. Any resumed compute requires a separate decision; the original clock is not reset and no continuation is automatically launched.
