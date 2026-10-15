# HU100 v0.5.3: terminal partial three-seed campaign

**Stopped 2026-10-15T12:00:39.357617+00:00. Both new training seeds and every-row audits completed; evaluation stopped at preventive disk admission after 9/17 comparisons. No qualification or poker outcome readout.** The owner decides merge and release.

[PR286](https://github.com/dberweger2017/CFR-NLHE-6max/pull/286) repeats the unchanged [#260 → #272 recipe](../hu100-v053-three-seeds-protocol.md), sequentially on M4. No retry, failed-hand rerun, count reduction, budget extension, release, tag, default change or Slumbot.

## Training and full audits

Seed **2026101501**: discarded updates **1,091,677,333**; discarded traverser visits **426,078,658**. All current/average rows pass `src.diagnostics.cfr_average.audit`, uniform zero-mass exports. [Audit](hu100-v053-three-seeds-artifacts/2026101501-5b.audit.json), [lineage](hu100-v053-three-seeds-artifacts/2026101501-training-verified.json).

Seed **2026101502**: discarded updates **1,096,014,039**; discarded traverser visits **427,878,763**. All current/average rows pass `src.diagnostics.cfr_average.audit`, uniform zero-mass exports. [Audit](hu100-v053-three-seeds-artifacts/2026101502-5b.audit.json), [lineage](hu100-v053-three-seeds-artifacts/2026101502-training-verified.json).

| Seed | Stage-1 nodes | Iterations | Entries | Final nodes | Iterations | Rows audited |
| --- | --- | --- | --- | --- | --- | --- |
| 2026101501 | 330748996 | 100827 | 97291138 | 5000004085 | 1794239 | 97291138 |
| 2026101502 | 330510042 | 101154 | 97291445 | 5000003034 | 1776372 | 97291445 |

Stage 1 copies the fixed **97,290,497** cap, 500M milestone and 1B target, ending at the first complete iteration beyond capacity. Stage 2 hash-pins that seed’s terminal, freezes at its actual entry count and targets 5B. Sorted key lists match exactly by streaming count/SHA256. Filenames name targets, not actual work. HU100 10,000/10,000, 50/100, no ante/rake/raise cap; linear CFR, opponent-sampled average, one root per seat, ordered history, menu v2, pinned #163 K50. Native build SHA256 `4f6abcbb954c3c744448c2a3542ab3164e0e1a015cb05ecebeda077525991cd4` matches #272.

| Seed | Save | Visits/key | Zero visits | Zero visits % | Zero average mass keys |
| --- | --- | --- | --- | --- | --- |
| 2026101501 | capacity terminal | 0.656047 | 55339095 | 56.8799% | 33012735 |
| 2026101501 | frozen 5B | 5.274088 | 32019282 | 32.9108% | 22187149 |
| 2026101502 | capacity terminal | 0.655108 | 55325041 | 56.8653% | 33004765 |
| 2026101502 | frozen 5B | 5.255096 | 31993485 | 32.8842% | 22259667 |

| Seed | Street | Traverser decisions | Missing traverser | Opponent decisions | Missing opponent |
| --- | --- | --- | --- | --- | --- |
| 2026101501 | preflop | 23229175 | 1610801 | 53176394 | 1878906 |
| 2026101501 | flop | 91059570 | 11244863 | 190517790 | 13282009 |
| 2026101501 | turn | 252201711 | 87079929 | 469052825 | 112487508 |
| 2026101501 | river | 508882679 | 326143065 | 964697444 | 546950949 |
| 2026101502 | preflop | 23215579 | 1701610 | 53115779 | 1875136 |
| 2026101502 | flop | 90938573 | 11618463 | 190462769 | 13715237 |
| 2026101502 | turn | 252297530 | 87779442 | 469422965 | 113988287 |
| 2026101502 | river | 508966527 | 326779248 | 964703100 | 547527977 |

Frozen misses describe counterfactual training roles, not played-hand coverage. The full receipts retain street shares. Stored-key density does not establish convergence.

## Evaluation retained without readout

All 17 cost-only pilots completed. Full **16,384 panel /131,072 direct blocks** were declared before final hands; [canonical plan](hu100-v053-three-seeds-artifacts/canonical-plan.json) and [exact PR-description readback](hu100-v053-three-seeds-artifacts/plan-posted.json). Whole prospective cost **9.433 hours**: elapsed 15341.309s + 1.25 × (627.773s fixed + 11520.596s per-block + 1305.168s future cache setup) + 1800s closeout. Only per-block work scales. The original deadline remains October 15, 20:34:12 UTC; unused budget is not permission to retry.

| Opponent | 2026100601 (#272), paired BB/100 [95%] | 2026101501 | 2026101502 |
| --- | --- | --- | --- |
| random | +25.39 [+5.24, +45.54] | withheld (barrier incomplete) | not run |
| check_call | +143.90 [+128.09, +159.71] | withheld (barrier incomplete) | not run |
| tight_aggressive | -12.95 [-22.90, -3.01] | withheld (barrier incomplete) | not run |
| loose_aggressive | -17.75 [-39.21, +3.71] | withheld (barrier incomplete) | not run |
| pot_pressure | +61.76 [+44.42, +79.11] | withheld (barrier incomplete) | not run |
| half_pot_pressure | -14.59 [-29.77, +0.59] | withheld (barrier incomplete) | not run |
| matched_pot_pressure | -26.57 [-40.79, -12.34] | withheld (barrier incomplete) | not run |

The new primary panel uses frozen v2 translation off versus v0.5.2 translation on (512/128). #272’s historical rule met its declared limits; neither new seed has a reportable rule outcome. This is a historical comparison, not an improvement-release gate.

| Fresh direct seed | Progress | Poker result / reached support |
| --- | --- | --- |
| 2026100601 | replayed-and-reproduced | withheld / no global readout |
| 2026101501 | replayed-and-reproduced | withheld / no global readout |
| 2026101502 | not-completed | withheld / no global readout |

**983,040 original final hands** in nine comparisons independently replayed and exactly reproduced. All seven seed-2026101501 panels and direct matches for 2026100601/2026101501 completed; seed-2026101502’s eight final comparisons never started. Selected-key coverage verification receipts exist for all nine, including both-side support for the two direct matches. Native and effective coverage counts remain sealed with the raw traces. The required all-17 barrier is absent; no means, intervals, new historical-rule outcomes or coverage readout is produced. Cost-only pilots remain disjoint and unpooled.

## Failure, resources and storage

The next seed’s cache admission refused **25,183,547,392 free bytes − 8,076,238,511 additional bytes**, which would leave **17,107,308,881 bytes**, below the **17,179,869,184-byte (16 GiB)** floor. This was a preventive refusal before cache construction or that seed’s final hands. The controller exited 1; the outer guard retained the failure and stopped. No resource breach triggered it. [Terminal receipts](hu100-v053-three-seeds-artifacts/partial-status.json).

Both shared locks and detached tmux on M4; all **3,884** five-second readings retained. Sampled peak family RSS **9.450 GiB**, minimum free memory **41%**, minimum free disk **17.967 GiB**, maximum original-baseline swap growth **0.524 GB**, pressure levels **[1, 2]**. Original guards/baseline are unchanged for archival preservation. [Resources](hu100-v053-three-seeds-artifacts/resource-summary.json).

- Remote git fetch used a limited refspec; origin/feature/hu100-v053-three-seeds was absent. Worktree add failed before any build/training. Corrected to fetched FETCH_HEAD.

- The previous setup left an empty planning directory and venv symlink, so the second worktree add refused its existing destination. Removed only those newly created setup entries, then used the already-created local branch.

- Detached launch PATH omitted /usr/sbin; guard could not find sysctl before creating guard baseline or starting build. Corrected admission used the original budget/deadline and preserved all logs.

- Coordinator used system Python3.9 for root reservation; dataclass slots import failed before scan or hands. Switched to supported research venv.

- A new synthetic direct test initially expected coverage grouped only by model; actual keys are model/street. Corrected assertion; no scientific final hand ran.

- Closeout process enumeration initially encountered macOS AccessDenied on PID 1; corrected to permission-tolerant metadata enumeration. No process was signalled.


Both new model sets remain in their unsynced campaign root. Only this campaign’s inactive regenerable release cache is eligible for removal, with manifest/input/rebuild provenance; models, unique evidence and synced files stay. Separate average-only ZIPs and raw trace/shared/lifecycle archives are sealed where headroom permits, with manifests and full local readback. Full checkpoint/current bundles remain pending if they cannot fit above 16 GiB. Exact archive/member hashes, Drive locators, upload status and restore commands are in [RESULTS_INDEX](../../RESULTS_INDEX.md); no archive or upload completion is implied by this partial report.

The campaign’s 30-minute schedule is disabled on this terminal stop. The owner decides merge and release.
