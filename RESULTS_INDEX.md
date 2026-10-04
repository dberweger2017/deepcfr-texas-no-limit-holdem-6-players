# Research results index

## October 4 organization and native Drive uploads

[Canonical repository index](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/main/RESULTS_INDEX.md). Repository-relative report links below resolve there; the Drive folder links work independently.

This is the existing research index. Its September inventory and October 2 receipts remain below; the current locations and upload states in this section supersede those earlier snapshots.

Both Macs use Google Drive desktop in streaming mode, with `~/Local/Research-Cloud` pointing to the same [research folder](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s). Whole archives and experiment folders go through the desktop app. The connector's observed 536,870,912-byte ingestion ceiling does not apply to this route; no research payload was split for the native uploads.

- **Organization verified:** 207 existing entries moved into [Historical-experiments](https://drive.google.com/drive/folders/1AIsc7LBHc7ziwrpOuwMuCPnvZpCJS1Zr), and five cleanup records into [Archive-receipts](https://drive.google.com/drive/folders/10cG2BTBZLcE7lO10TgBZ0Ud32quaf_BX). IDs and existing links are preserved. The [relocation receipt](docs/artifacts/native-drive-relocations-20261004.json) records every original parent, destination and verified item ID.
- **M4 upload queue:** nine closed scopes, **28,756 files / 96,459,366,205 logical bytes (89.83 GiB)**, hashed before relocation into the streamed folder. Includes closed training and flop/turn evidence. Local payloads remain under Drive management while uploading.
- **M1 upload queue:** four closed scopes, **24,637 files / 37,983,420,181 logical bytes (35.37 GiB)**, likewise hashed and staged. The first attempt stopped before moving any file because newly created cloud folders had not appeared locally; after those folders synchronized, staging completed. The failed attempt is retained.
- **PR136 upload queue:** the intact `hu20-500m-campaign-20261001-archive-20261002.tar.gz`, **23,509,934,825 bytes**, is uploading from M1. SHA256: `c0dbb3879eeae5f89be888ca3b29a6c571f6adb214c9d23c6095d0cab007636d`. The separate M4 originals remain until upload confirmation.
- **Already confirmed:** the 14,041,643,443-byte branching archive, the overnight archive, PR144's six archives and 18 earlier M4 archives. Their previous cloud IDs remain valid; the branching archive now lives under `Historical-experiments/`.
- **Pending means pending:** native staging is not cloud-upload confirmation. An hourly archival follow-up checks completion and remaining duplicate/input dependencies. The [compact staging receipt](docs/artifacts/native-drive-staging-20261004.json) records exact source paths, destination paths, counts and manifest names. Logical totals include separately retained evidence and do not measure unique or physical disk use.

### Current folder layout

Paths are relative to the research Drive folder. Upload status is a snapshot on October 4, not a promise of completion. Each newly staged experiment contains `ARCHIVE-MANIFEST-20261004.json` with member sizes/hashes and `RESTORE-README.txt`.

| Folder | Evidence / location within it | State |
| --- | --- | --- |
| [PR-113-TP20](https://drive.google.com/drive/folders/1bvceeB1paom93UQyGbbR29gXxco5_H3M) | `M4-results/` | Native upload queued |
| [PR-132-observation-reuse](https://drive.google.com/drive/folders/1l8T_GUwa2uJ8LfqmplrMglQx_3hDn1lC) | `M4-results/`, `M4-CI-repair-results/` | Native upload queued |
| [PR-133-mature-CPU](https://drive.google.com/drive/folders/1xH2hQ8budf3n9tOhCxcMj3A4d5e9pllL) | `M4-initial-results/`, `M4-six-lineage-results/` | Native upload queued |
| [PR-136-HU20-500M](https://drive.google.com/drive/folders/1Jjg9yvbPQ25nws_wW1IupyMYCfnNVFc_) | Whole 23.51-GB campaign archive and recovery manifest | Whole archive uploading |
| [PR-144-history-compression](https://drive.google.com/drive/folders/1EnCmKftt50pebTWVtu1MvTUQEaw_5hCV) | Six complete archives; `archive-manifest.json`, `SHA256SUMS` | Confirmed uploaded |
| [PR-145-exact-flop](https://drive.google.com/drive/folders/1ciOOpSaLHqvCSI8wzhWDQORtizfeCrxZ) | `M4-results/`, including retained flop attempts and exact-turn follow-up | Native upload queued |
| [PR-148-turn-calibration](https://drive.google.com/drive/folders/1pmu8GZww8SHQBkVgWM5a-Rn5txeERxC5) | `M4-closed-research/`, `M1-closed-research/`, `M1-exact-turn-evidence/` | Native upload queued |
| [M4-closed-training](https://drive.google.com/drive/folders/1Nm2vyxf2GbEk9t30--vPluJ-IFrk-TbQ) | `results/` from `deepcfr-training` | Native upload queued |
| [M4-local-CFR-diagnostic](https://drive.google.com/drive/folders/1i9HAiFFXKTgS0FLzGUtwsrEeNLaCiQsT) | `results/` | Native upload queued |
| [M1-board-pooling](https://drive.google.com/drive/folders/13SUnP_d1ZVtcyJAp-oqRecg5Wv3SYIfy) | `closed-evidence/` | Native upload queued |
| [M1-alias-audit](https://drive.google.com/drive/folders/18cCPHR6EfXLUBGbwmNukxttySSUUj0Rf) | `closed-evidence/` | Native upload queued |
| [Historical-experiments](https://drive.google.com/drive/folders/1AIsc7LBHc7ziwrpOuwMuCPnvZpCJS1Zr) | Earlier inventory entries, retaining names and IDs | Organization verified |
| [Archive-receipts](https://drive.google.com/drive/folders/10cG2BTBZLcE7lO10TgBZ0Ud32quaf_BX) | Cleanup and restoration receipts | Organization verified |

Other existing PR/M4 archive folders remain at the research root. Original experiment paths are now directory or member symlinks where staged; they are restoration conveniences, not independent backups. Private credentials, Git internals and reproducible environments/caches were excluded and retained locally. Ordinary coding checkouts and fixtures remain available.

Let streaming reclaim cache space gradually, as requested by the owner. No immediate forced offloading is needed. **Never delete within the synced folder to free local space:** that deletes the cloud copy too. Separately retained inactive originals may be removed only after the accepted desktop completion check and dependency review. Restore research into a local working directory and verify the manifest before future computation.

## Historical October 2 archive and local cleanup

The owner accepted Google Drive desktop's **Successfully uploaded / Synced** status as the upload acceptance check. Inactive copies have now been removed: **8,722 files / 41,249,100,807 logical bytes (38.42 GiB)** across the primary checkout, PR144 evidence and temporary archive staging. The full removal/restore receipt is `LOCAL-CLEANUP-20261002.json` in the research Drive folder; the [compact cleanup record](docs/artifacts/local-cleanup-20261002.json) preserves totals, archive hashes, paths and exclusions.

- **PR144:** six complete archives in [PR-144-history-compression](https://drive.google.com/drive/folders/1EnCmKftt50pebTWVtu1MvTUQEaw_5hCV). Download `archive-manifest.json` and `SHA256SUMS` from that folder; verify the selected archive, then extract its member path under a fresh restoration directory. The manifest maps every archive prefix to its original working root. Failed/partial attempts and all closed raw evidence remain in these archives. The original local retrieval paths in the [scientific report](docs/reports/dr2x2-ac-comparison.md) now require this restoration step.
- **Preserved locally:** original A inputs, six final C checkpoint/current files, six A/C average exports referenced by symlinks, working blueprint trees, tracked reports, private credentials and other agents' work.
- **Still uploading:** `branching-comparison.tar.gz` (14,041,643,443 bytes). Its original remains local until the desktop confirms completion.
- **M4:** read-only serial archival transfer started for 27 closed research result roots, including PR136, into separate PR/root folders. Uploads are **pending**, and every M4 original remains intact. Active exact-flop/turn experiment and its input root are excluded. Durable local progress: `/Users/dberweger/Local/research-archive-preparation-20261002/m4-archive-status.json`.
- **Git housekeeping:** 411 abandoned, unindexed temporary Git files were removed after checking age and open handles. All four repository HEAD trees remained readable. Free disk was about **120 GiB** immediately afterward; ongoing archive staging changes that figure. No committed Git pack, branch or scientific source was removed.


Large untracked training/evaluation artifacts are indexed here so a PR can link to their location without checking them into Git. Working files may live wherever convenient. During cleanup, archive them to the owner-designated [Google Drive folder](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s).

## Current archive state

The September 30 catalog remains the original inventory: **8,117 files / 208 top-level entries / 48,465,009,999 logical bytes**. October 2 cleanup removed completed inactive entries after owner acceptance of Drive upload completion. The [catalog](docs/artifacts/results-catalog.json) now marks each removed or retained top-level entry. The October 4 section records subsequent completed branching upload and native staging; retained input dependencies still require review before duplicate cleanup.

- Original root: `/Users/dberweger/Local/deepcfr-texas-no-limit-holdem-6-players/results`.
- Streamed destination: `/Users/dberweger/Library/CloudStorage/GoogleDrive-dberweger2017@gmail.com/My Drive/deepcfr-research-results`.
- The original inventory names are retained. Current legacy paths gain the `Historical-experiments/` prefix where recorded in the October 4 relocation receipt; existing PR folders remain at the root.
- [Machine-readable catalog](docs/artifacts/results-catalog.json) records exact bytes/files and report references. Logical sizes include duplicate retained archives/extracted copies; they do not measure physical APFS storage.

## Cleanup and retrieval policy

1. Keep active checkpoints, recovery slots and other agents’ input dependencies in their working location. Coordinate before cleanup.
2. Preserve every attempt, failed/partial output, source/config/model identity, manifest/hash and exact retrieval command. Add the PR/report and archive relative path to this index. Never replace scientific evidence with a success-only archive.
3. Let Drive upload. Under the owner's October 2 decision, accept the desktop's Successfully uploaded/Synced status, check exact cloud names/sizes and preserve archive member hashes and restoration evidence. A placeholder or filename alone is insufficient. Remove only owner-authorized inactive local copies; keep pending uploads and active dependencies.
4. Before analysis, restore the needed immutable files into a working directory, verify their hashes and source/model identities, and keep restored heavy computation on the authorized host. Do not train or audit inside a streamed folder.

## Recent work outside the initial batch

These are the original working roots, outside the initial primary-checkout batch. October 4 staging places #132/#133 evidence in the PR folders above and preserves old paths as cloud links. #134 input dependencies remain local pending review.

| PR | Retained location / evidence |
| --- | --- |
| [#132](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/132) | M4 `/Users/dberweger/Local/trainer-observation-reuse-20260930/results/observation-reuse-m4-20260930`; manifest/retrieval in `docs/reports/trainer-observation-reuse-m4.md` on that PR. |
| [#133](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/133) | M4 `/Users/dberweger/Local/runpod-mature-cpu-six-20260930/results`; complete and failed/partial mature CPU attempts; manifests/report on that PR. Preserve active extra-control dependencies. |
| [#134](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/134) | M1 `/Users/dberweger/.codex/worktrees/main-isolated/deepcfr-texas-no-limit-holdem-6-players/results/hu20-stackoff-20260930` (~43 MiB) and `results/hu20-stackoff-inputs` (1,017,051,279 bytes); report/compact evidence on that PR. New Guy reports all owned workers closed and inputs retained. |

## Initial batch table of contents

**Historical inventory; current removed/retained states are in the machine-readable catalog and October 2 cleanup record.** Report links identify associated research. Associated PRs identify the latest report-changing PR found in main history; they do not assert which run originally produced a file. Legacy/unmapped entries are retained for later identification.

| Entry in original inventory | Logical bytes | Files | Report / associated PR |
| --- | ---: | ---: | --- |
| `.DS_Store` | 32,772 | 1 | Unmapped legacy/local evidence |
| `analyze-fitting.py` | 1,348 | 1 | Unmapped legacy/local evidence |
| `arena-dev-1` | 578,287 | 6 | Unmapped legacy/local evidence |
| `arena-final-control` | 4,320,079 | 6 | [arena-validation](docs/reports/arena-validation.md); [#44](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/44) |
| `arena-final-control-replay` | 4,319,658 | 6 | [arena-validation](docs/reports/arena-validation.md); [#44](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/44) |
| `arena-final-smoke` | 684,241 | 6 | [arena-validation](docs/reports/arena-validation.md); [#44](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/44) |
| `arena-final-smoke-replay` | 684,214 | 6 | [arena-validation](docs/reports/arena-validation.md); [#44](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/44) |
| `arena-sensitivity-20260915` | 4,323,714 | 6 | Unmapped legacy/local evidence |
| `arena-sensitivity-replay-20260915` | 4,323,594 | 6 | Unmapped legacy/local evidence |
| `audit-readiness.py` | 6,455 | 1 | Unmapped legacy/local evidence |
| `audit-representation.py` | 1,355 | 1 | Unmapped legacy/local evidence |
| `audit_snapshot_readiness.py` | 7,300 | 1 | Unmapped legacy/local evidence |
| `blueprint-04-ops` | 13,368 | 2 | Unmapped legacy/local evidence |
| `blueprint-history-fixed-coverage-m4` | 168,896,207 | 10 | Unmapped legacy/local evidence |
| `blueprint-history-summary-learning-m4` | 520,841,454 | 22 | Unmapped legacy/local evidence |
| `blueprint-history-summary-v1-m4` | 684,819,417 | 19 | Unmapped legacy/local evidence |
| `blueprint-m4-slice-v1-continued-m4` | 256,135,764 | 5 | [blueprint-m4-slice-v1](docs/reports/blueprint-m4-slice-v1.md); [#95](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/95) |
| `blueprint-m4-slice-v1-m4` | 76,866,163 | 5 | [blueprint-m4-slice-v1](docs/reports/blueprint-m4-slice-v1.md); [#95](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/95) |
| `blueprint-pilot-v1-m4` | 198,609 | 14 | [blueprint-pilot-v1](docs/reports/blueprint-pilot-v1.md); [#91](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/91) |
| `blueprint-runpod-check-v1` | 1,839,316,262 | 53 | [blueprint-runpod-sizing-v1](docs/reports/blueprint-runpod-sizing-v1.md); [#97](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/97) |
| `blueprint-search-m4-smoke-20260924` | 137,076 | 7 | Unmapped legacy/local evidence |
| `blueprint-search-m4-smoke-final-20260924` | 137,667 | 7 | [blueprint-postflop-search-m4-smoke](docs/reports/blueprint-postflop-search-m4-smoke.md); [#103](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/103) |
| `blueprint-search-m4-smoke-pinned-20260924` | 137,081 | 7 | Unmapped legacy/local evidence |
| `branching-comparison-provisional.json` | 225,760 | 1 | Unmapped legacy/local evidence |
| `branching-comparison.tar.gz` | 14,041,643,443 | 1 | [holdem-branching-online](docs/reports/holdem-branching-online.md); [#79](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/79) |
| `branching-handoff-history.md` | 9,691 | 1 | Unmapped legacy/local evidence |
| `branching-handoff.md` | 1,251 | 1 | Unmapped legacy/local evidence |
| `branching-host-gate.json` | 1,869 | 1 | Unmapped legacy/local evidence |
| `branching-operations-local` | 72,238 | 24 | Unmapped legacy/local evidence |
| `branching-pilot-first` | 59,322,226 | 20 | Unmapped legacy/local evidence |
| `branching-pilot-first-resumed` | 38,757,526 | 16 | Unmapped legacy/local evidence |
| `branching-pilot-second` | 76,793,883 | 20 | Unmapped legacy/local evidence |
| `branching-pilot-second-resumed` | 45,502,176 | 16 | Unmapped legacy/local evidence |
| `branching-pilot-verification.json` | 5,136 | 1 | [holdem-branching-pilot](docs/reports/holdem-branching-pilot.md); [#78](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/78) |
| `branching-rental.json` | 1,297 | 1 | Unmapped legacy/local evidence |
| `branching-retrieved` | 1,502,864,548 | 507 | Unmapped legacy/local evidence |
| `build-fitting-report.py` | 3,137 | 1 | Unmapped legacy/local evidence |
| `build_strategy_report.py` | 6,776 | 1 | Unmapped legacy/local evidence |
| `card-diversity` | 1,368,388 | 3 | [holdem-multistreet-preliminary](docs/reports/holdem-multistreet-preliminary.md); [holdem-card-diversity-admission](docs/reports/holdem-card-diversity-admission.md); [holdem-next-training-analysis](docs/reports/holdem-next-training-analysis.md); [holdem-card-diversity-runtime-amendment](docs/reports/holdem-card-diversity-runtime-amendment.md); [#81](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/81); [#82](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/82); [#85](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/85); [#87](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/87) |
| `card-diversity-campaign` | 127,485,011 | 29 | [holdem-card-diversity](docs/reports/holdem-card-diversity.md); [#82](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/82) |
| `card-diversity-idle-calibration` | 1,350,148 | 2 | Unmapped legacy/local evidence |
| `card-diversity-retry` | 1,368,390 | 3 | Unmapped legacy/local evidence |
| `check-fitting.py` | 796 | 1 | Unmapped legacy/local evidence |
| `collect-readiness.py` | 3,347 | 1 | Unmapped legacy/local evidence |
| `collection-performance` | 288,750 | 32 | [holdem-collection-performance](docs/reports/holdem-collection-performance.md); [holdem-sampling-comparison](docs/reports/holdem-sampling-comparison.md); [#66](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/66); [#67](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/67) |
| `collector-branching` | 16,955,607 | 4 | [holdem-collector-branching](docs/reports/holdem-collector-branching.md); [#77](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/77) |
| `core-v1-baseline` | 19,364,403 | 6 | Unmapped legacy/local evidence |
| `cpu-pilot-local-v1` | 5,827,614 | 35 | [cpu-pilot](docs/reports/cpu-pilot.md); [#49](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/49) |
| `cpu-pilot-remote-v1` | 14,166,605 | 130 | [cpu-pilot](docs/reports/cpu-pilot.md); [#49](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/49) |
| `cpu-pilot-remote-v1.tar.gz` | 2,062,422 | 1 | [cpu-pilot](docs/reports/cpu-pilot.md); [#49](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/49) |
| `cpu-pilot-rental.json` | 793 | 1 | Unmapped legacy/local evidence |
| `cpu-pilot-replays` | 3,113,433 | 6 | [cpu-pilot](docs/reports/cpu-pilot.md); [#49](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/49) |
| `cpu-pilot-replays.tar.gz` | 3,057,126 | 1 | [cpu-pilot](docs/reports/cpu-pilot.md); [#49](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/49) |
| `decision-errors` | 18,943,875 | 2 | Unmapped legacy/local evidence |
| `frozen-fitting` | 1,440,341,426 | 86 | [holdem-persistent-critic](docs/reports/holdem-persistent-critic.md); [holdem-frozen-fitting](docs/reports/holdem-frozen-fitting.md); [holdem-next-training-analysis](docs/reports/holdem-next-training-analysis.md); [holdem-policy-comparison](docs/reports/holdem-policy-comparison.md); [#72](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/72); [#74](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/74); [#76](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/76); [#87](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/87) |
| `frozen-fitting-supervisor.log` | 0 | 1 | Unmapped legacy/local evidence |
| `historical-validation` | 3,734,237 | 8 | Unmapped legacy/local evidence |
| `historical-validation-replay` | 3,734,238 | 8 | Unmapped legacy/local evidence |
| `holdem-baseline-resumed-v1` | 67,894,936 | 73 | [holdem-baseline](docs/reports/holdem-baseline.md); [#61](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/61) |
| `holdem-baseline-v1` | 94,276,325 | 93 | [holdem-collection-performance](docs/reports/holdem-collection-performance.md); [holdem-variance](docs/reports/holdem-variance.md); [holdem-sampling-comparison](docs/reports/holdem-sampling-comparison.md); [holdem-baseline](docs/reports/holdem-baseline.md); [#61](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/61); [#66](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/66); [#67](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/67); [#68](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/68) |
| `launch-fitting.sh` | 1,229 | 1 | Unmapped legacy/local evidence |
| `launch-readiness.sh` | 926 | 1 | Unmapped legacy/local evidence |
| `local-training-proposal-20260918` | 133,431,692 | 23 | Unmapped legacy/local evidence |
| `longer-05-handoff.md` | 24,044 | 1 | Unmapped legacy/local evidence |
| `longer-05-integrity-second.log` | 318 | 1 | Unmapped legacy/local evidence |
| `longer-05-integrity.json` | 310 | 1 | Unmapped legacy/local evidence |
| `longer-05-integrity.log` | 369 | 1 | Unmapped legacy/local evidence |
| `longer-05-rental.json` | 732 | 1 | Unmapped legacy/local evidence |
| `longer-05-retrieved` | 992,906,569 | 231 | Unmapped legacy/local evidence |
| `longer-05-summary.json` | 82,825 | 1 | Unmapped legacy/local evidence |
| `longer-05-verification.tar.gz` | 10,578,536 | 1 | [holdem-longer-training](docs/reports/holdem-longer-training.md); [#73](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/73) |
| `longer-05.tar.gz` | 6,623,622,300 | 1 | [holdem-longer-training](docs/reports/holdem-longer-training.md); [#73](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/73) |
| `longer-05.tar.gz.sha256` | 94 | 1 | Unmapped legacy/local evidence |
| `longer-calibration` | 49,494,973 | 25 | [holdem-longer-calibration](docs/reports/holdem-longer-calibration.md); [#70](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/70) |
| `longer-calibration-tests.log` | 672 | 1 | Unmapped legacy/local evidence |
| `longer-calibration-verification.json` | 1,570 | 1 | Unmapped legacy/local evidence |
| `longer-calibration-verification.log` | 49 | 1 | Unmapped legacy/local evidence |
| `longer-calibration-verified` | 49,493,734 | 24 | Unmapped legacy/local evidence |
| `longer-calibration.log` | 826 | 1 | Unmapped legacy/local evidence |
| `longer-calibration.tar.gz` | 16,997,428 | 1 | [holdem-longer-calibration](docs/reports/holdem-longer-calibration.md); [#70](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/70) |
| `m4-overnight-20260919` | 1,014,177,619 | 459 | [holdem-continuous-m4](docs/reports/holdem-continuous-m4.md); [#88](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/88) |
| `m4-overnight-20260919.tar.zst` | 9,399,210,638 | 1 | [holdem-continuous-m4](docs/reports/holdem-continuous-m4.md); [#88](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/88) |
| `m4-overnight-20260919.tar.zst.sha256` | 144 | 1 | Unmapped legacy/local evidence |
| `m4-review` | 3,676 | 1 | Unmapped legacy/local evidence |
| `multistreet-calibration-optimized` | 422,056 | 3 | Unmapped legacy/local evidence |
| `multistreet-calibration-original` | 421,467 | 2 | Unmapped legacy/local evidence |
| `multistreet-campaign.tar.gz` | 20,793,652 | 1 | [holdem-multistreet-campaign](docs/reports/holdem-multistreet-campaign.md); [#85](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/85) |
| `multistreet-handoff.md` | 14,085 | 1 | Unmapped legacy/local evidence |
| `multistreet-integration` | 1,303,943 | 11 | [holdem-multistreet-pilot](docs/reports/holdem-multistreet-pilot.md); [#83](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/83) |
| `multistreet-integration-plan.json` | 6,322 | 1 | [holdem-multistreet-pilot](docs/reports/holdem-multistreet-pilot.md); [#83](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/83) |
| `multistreet-integration.log` | 52,967 | 1 | Unmapped legacy/local evidence |
| `multistreet-operations-local` | 46,882 | 23 | Unmapped legacy/local evidence |
| `multistreet-pilot` | 9,796 | 2 | [holdem-multistreet-pilot](docs/reports/holdem-multistreet-pilot.md); [#83](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/83) |
| `multistreet-process-smoke-1` | 19,758 | 12 | Unmapped legacy/local evidence |
| `multistreet-process-smoke-1.json` | 486 | 1 | Unmapped legacy/local evidence |
| `multistreet-process-smoke-4` | 19,754 | 12 | Unmapped legacy/local evidence |
| `multistreet-process-smoke-4.json` | 486 | 1 | Unmapped legacy/local evidence |
| `multistreet-reference-stability.json` | 378,831 | 1 | [holdem-multistreet-campaign](docs/reports/holdem-multistreet-campaign.md); [#85](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/85) |
| `multistreet-retrieved` | 87,765,231 | 2713 | [holdem-multistreet-campaign](docs/reports/holdem-multistreet-campaign.md); [#85](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/85) |
| `multistreet-source-310258a.tar.gz` | 2,321,096 | 1 | [holdem-multistreet-pilot](docs/reports/holdem-multistreet-pilot.md); [#83](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/83) |
| `neural-convergence-kuhn-11` | 33,130,869 | 11 | Unmapped legacy/local evidence |
| `neural-convergence-kuhn-29` | 33,205,100 | 11 | Unmapped legacy/local evidence |
| `neural-convergence-kuhn-47` | 33,201,917 | 11 | Unmapped legacy/local evidence |
| `neural-convergence-kuhn-summary` | 36,210 | 2 | Unmapped legacy/local evidence |
| `neural-convergence-leduc-11` | 50,966,099 | 9 | Unmapped legacy/local evidence |
| `neural-convergence-leduc-11-paused` | 41,587,271 | 8 | Unmapped legacy/local evidence |
| `neural-convergence-leduc-29` | 92,894,296 | 12 | Unmapped legacy/local evidence |
| `neural-convergence-leduc-47` | 93,137,412 | 12 | Unmapped legacy/local evidence |
| `neural-convergence-leduc-summary` | 36,387 | 2 | Unmapped legacy/local evidence |
| `neural-fitting-v1` | 4,121 | 2 | Unmapped legacy/local evidence |
| `neural-kuhn-v1` | 68,573 | 3 | [neural-validation](docs/reports/neural-validation.md); [#47](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/47) |
| `neural-leduc-refit-v1` | 4,771 | 2 | Unmapped legacy/local evidence |
| `neural-leduc-v1` | 60,300 | 3 | [neural-validation](docs/reports/neural-validation.md); [#47](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/47) |
| `neural-readiness-handoff.md` | 1,598 | 1 | Unmapped legacy/local evidence |
| `neural-readiness-retrieved` | 818,445,367 | 312 | [neural-readiness](docs/reports/neural-readiness.md); [#55](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/55) |
| `neural-readiness.tar.gz` | 138,850,141 | 1 | [neural-readiness](docs/reports/neural-readiness.md); [#55](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/55) |
| `neural-strategy-refit-v1` | 9,785 | 2 | [neural-convergence](docs/reports/neural-convergence.md); [#48](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/48) |
| `opponents-validation` | 850,470 | 6 | Unmapped legacy/local evidence |
| `opponents-validation-replay` | 850,531 | 6 | Unmapped legacy/local evidence |
| `persistent-critic` | 515,168,017 | 29 | [holdem-persistent-critic](docs/reports/holdem-persistent-critic.md); [holdem-next-training-analysis](docs/reports/holdem-next-training-analysis.md); [holdem-river-reference](docs/reports/holdem-river-reference.md); [#75](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/75); [#76](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/76); [#87](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/87) |
| `plot_strategy_report.py` | 3,400 | 1 | Unmapped legacy/local evidence |
| `policy-comparison` | 45,668,305 | 46 | [holdem-multistreet-preliminary](docs/reports/holdem-multistreet-preliminary.md); [holdem-next-training-analysis](docs/reports/holdem-next-training-analysis.md); [holdem-policy-comparison](docs/reports/holdem-policy-comparison.md); [#74](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/74); [#85](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/85); [#87](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/87) |
| `policy-comparison-supervisor.log` | 0 | 1 | Unmapped legacy/local evidence |
| `readiness-audit.json` | 9,881 | 1 | Unmapped legacy/local evidence |
| `readiness-cleanup.json` | 470 | 1 | Unmapped legacy/local evidence |
| `readiness-collector.json` | 342 | 1 | Unmapped legacy/local evidence |
| `readiness-collector.log` | 0 | 1 | Unmapped legacy/local evidence |
| `readiness-collector.pid` | 6 | 1 | Unmapped legacy/local evidence |
| `readiness-launch-manifest.json` | 9,712 | 1 | Unmapped legacy/local evidence |
| `readiness-provider-stop.py` | 2,062 | 1 | Unmapped legacy/local evidence |
| `readiness-runtime.json` | 3,813 | 1 | Unmapped legacy/local evidence |
| `representation` | 49,446,619 | 23 | [hu20-native-reopening-preliminary](docs/reports/hu20-native-reopening-preliminary.md); [holdem-multistreet-preliminary](docs/reports/holdem-multistreet-preliminary.md); [holdem-longer-training](docs/reports/holdem-longer-training.md); [strategy-fitting](docs/reports/strategy-fitting.md); [#52](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/52); [#73](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/73); [#85](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/85); [#115](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/115) |
| `representation-run.log` | 833 | 1 | Unmapped legacy/local evidence |
| `river-reference` | 17,154,184 | 12 | [holdem-river-reference](docs/reports/holdem-river-reference.md); [#75](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/75) |
| `river-reference.log` | 3,306 | 1 | Unmapped legacy/local evidence |
| `sampled-full-tests.log` | 672 | 1 | Unmapped legacy/local evidence |
| `sampled-pilot` | 310,756,128 | 159 | [holdem-sampled-pilot](docs/reports/holdem-sampled-pilot.md); [#69](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/69) |
| `sampled-pilot-run.log` | 827 | 1 | Unmapped legacy/local evidence |
| `sampled-pilot-verification-initial` | 3,515,806 | 5 | Unmapped legacy/local evidence |
| `sampled-pilot-verification-initial.log` | 240 | 1 | Unmapped legacy/local evidence |
| `sampled-pilot-verification.log` | 68 | 1 | Unmapped legacy/local evidence |
| `sampled-pilot-verified` | 60,962,178 | 49 | Unmapped legacy/local evidence |
| `sampled-pilot.tar.gz` | 138,909,769 | 1 | [holdem-sampled-pilot](docs/reports/holdem-sampled-pilot.md); [#69](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/69) |
| `sampling-comparison` | 909,657 | 18 | [holdem-sampling-comparison](docs/reports/holdem-sampling-comparison.md); [#67](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/67) |
| `sampling-comparison.tar.gz` | 73,982 | 1 | [holdem-sampling-comparison](docs/reports/holdem-sampling-comparison.md); [#67](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/67) |
| `sampling-variance` | 2,527,826 | 33 | [holdem-variance](docs/reports/holdem-variance.md); [#68](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/68) |
| `sampling-variance.tar.gz` | 450,685 | 1 | [holdem-variance](docs/reports/holdem-variance.md); [#68](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/68) |
| `selfplay_from_3000_eval_3500_to_6000.csv` | 1,927 | 1 | Unmapped legacy/local evidence |
| `selfplay_from_3000_eval_3500_to_6000.json` | 4,504 | 1 | Unmapped legacy/local evidence |
| `setup-fitting.sh` | 590 | 1 | Unmapped legacy/local evidence |
| `snapshot-operations-local` | 2,316 | 5 | Unmapped legacy/local evidence |
| `snapshot-preview-manifest.json` | 10,444 | 1 | Unmapped legacy/local evidence |
| `snapshot-preview-report.json` | 265,099 | 1 | Unmapped legacy/local evidence |
| `snapshot-readiness-audit.json` | 30,993 | 1 | Unmapped legacy/local evidence |
| `snapshot-readiness-handoff.md` | 1,563 | 1 | Unmapped legacy/local evidence |
| `snapshot-readiness-retrieved` | 3,202,663,274 | 360 | [snapshot-readiness](docs/reports/snapshot-readiness.md); [#65](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/65) |
| `snapshot-readiness.sha256` | 103 | 1 | Unmapped legacy/local evidence |
| `snapshot-readiness.tar.gz` | 1,869,868,238 | 1 | [snapshot-readiness](docs/reports/snapshot-readiness.md); [#65](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/65) |
| `snapshot-smoke-v1` | 1,218,915 | 85 | Unmapped legacy/local evidence |
| `standard_phase1.csv` | 2,575 | 1 | Unmapped legacy/local evidence |
| `standard_phase1.json` | 7,180 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_1500_4000.csv` | 1,794 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_1500_4000.json` | 4,371 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_2500_3000.csv` | 1,787 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_2500_3000.json` | 4,364 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_2500_5000.csv` | 1,794 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_2500_5000.json` | 4,371 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_early.csv` | 1,350 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_early.json` | 2,913 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_top.csv` | 1,545 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_top.json` | 3,615 | 1 | Unmapped legacy/local evidence |
| `strategy-capacity-calibration-copy` | 370,183 | 2 | Unmapped legacy/local evidence |
| `strategy-capacity-handoff.md` | 8,389 | 1 | Unmapped legacy/local evidence |
| `strategy-capacity-remote-v1` | 947,223,673 | 584 | [strategy-capacity](docs/reports/strategy-capacity.md) |
| `strategy-capacity-remote-v1.tar.gz` | 180,879,752 | 1 | [strategy-capacity](docs/reports/strategy-capacity.md) |
| `strategy-capacity-rental.json` | 1,027 | 1 | Unmapped legacy/local evidence |
| `strategy-capacity-summary.json` | 11,356 | 1 | Unmapped legacy/local evidence |
| `strategy-capacity.png` | 169,765 | 1 | Unmapped legacy/local evidence |
| `strategy-fitting-handoff.md` | 5,328 | 1 | Unmapped legacy/local evidence |
| `strategy-fitting-input-audit.json` | 5,398 | 1 | Unmapped legacy/local evidence |
| `strategy-fitting-inputs` | 13,644,908 | 37 | [strategy-fitting](docs/reports/strategy-fitting.md); [#52](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/52) |
| `strategy-fitting-inputs.tar.gz` | 13,361,710 | 1 | Unmapped legacy/local evidence |
| `strategy-fitting-remote.tar.gz` | 41,128,008 | 1 | [strategy-fitting](docs/reports/strategy-fitting.md); [#52](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/52) |
| `strategy-fitting-rental.json` | 1,413 | 1 | Unmapped legacy/local evidence |
| `strategy-fitting-retrieved` | 123,573,053 | 1194 | Unmapped legacy/local evidence |
| `strategy-fitting-smoke` | 1,333,925 | 27 | Unmapped legacy/local evidence |
| `summarize-representation.py` | 2,435 | 1 | Unmapped legacy/local evidence |
| `summarize-sampled-pilot.py` | 3,165 | 1 | Unmapped legacy/local evidence |
| `tabular-reference-v1` | 596,913 | 10 | [tabular-validation](docs/reports/tabular-validation.md); [#46](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/46) |
| `tabular-reference-v1-replay` | 596,916 | 10 | [tabular-validation](docs/reports/tabular-validation.md); [#46](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/46) |
| `tensorboard-calibration-check` | 8,676 | 1 | Unmapped legacy/local evidence |
| `tournament_phase1_20k_1500_to_4000_step500` | 2,446,589 | 6 | Unmapped legacy/local evidence |
| `tournament_phase1_20k_2500_3000` | 2,209,823 | 6 | Unmapped legacy/local evidence |
| `tournament_phase1_20k_2500_to_5000_step500` | 2,265,302 | 6 | Unmapped legacy/local evidence |
| `tournament_phase1_20k_2500_to_5000_step500_10k` | 5,925,211 | 6 | Unmapped legacy/local evidence |
| `tournament_phase1_20k_500_to_3000_step500` | 2,227,387 | 6 | Unmapped legacy/local evidence |
| `tournament_phase1_20k_early` | 2,305,659 | 6 | Unmapped legacy/local evidence |
| `tournament_selfplay_from_3000_3500_to_6000_100` | 1,218,995 | 6 | Unmapped legacy/local evidence |
| `tournament_selfplay_from_3000_3500_to_6000_10k` | 5,460,669 | 6 | Unmapped legacy/local evidence |
| `update-readiness-docs.py` | 5,786 | 1 | Unmapped legacy/local evidence |
| `update_snapshot_docs.py` | 6,681 | 1 | Unmapped legacy/local evidence |
| `verify-longer-archive.py` | 2,795 | 1 | Unmapped legacy/local evidence |
| `verify-sampled-pilot-initial.py` | 2,520 | 1 | Unmapped legacy/local evidence |
| `verify-sampled-pilot.py` | 2,544 | 1 | Unmapped legacy/local evidence |
| `write-readiness-report.py` | 11,093 | 1 | Unmapped legacy/local evidence |
| `write-representation-report.py` | 5,622 | 1 | Unmapped legacy/local evidence |
| `write_snapshot_report.py` | 11,092 | 1 | Unmapped legacy/local evidence |
| `write_strategy_summary.py` | 11,610 | 1 | Unmapped legacy/local evidence |

## HU20 turn-search development (PR #148)

- External AGPL play/quality harness: `/Users/dberweger/Local/hu20-turn-search-tool`; upstream is a symlink to the unchanged pinned `/Users/dberweger/Local/hu20-exact-flop-tool/upstream`. Restore with commit 9d1509fe5077d019825f833eed04b16d342dfda1. Source/binary fingerprints: `docs/reports/hu20-turn-search-artifacts/external-inventory.json`. Never vendor these sources into MIT.
- M1 references, requests, responses, logs, profiles and receipts: `/Users/dberweger/Local/hu20-turn-search-20261002/m1-parity-reference-01` through `-08`; `-04` retains the failed inserted-wager fixture. Current reference is `-08`; thread comparisons are `m1-thread-parity-t1`, `-t4`, `-t6` under the same parent. Compact counts/hashes: `docs/reports/hu20-turn-search-artifacts/m1-parity.json`. Full profiles, the parent `status.md`, old references and both build logs remain retained. The M4 phase is indexed below; no rental has started.
- Inputs remain the preserved six exports in `/Users/dberweger/Local/hu20-m4-archive-20261002/hu20-exact-flop-check-inputs`; they remain other agents' dependencies.
- Keep all originals and failed builds/requests. Archive to the existing designated Drive destination with complete manifests and symlink retrieval provenance at cleanup. No local deletion is authorized.

- HU20 #148 M4 admission/pilot and future Part A: `/Users/dberweger/Local/hu20-turn-search-20261003` on M4; isolated MIT checkout `repo`, source 75cbda7 for the 156-hand pilot. Includes immutable admission, six input hashes, published #145 report copy, cumulative `budget.json`, initial helper startup failure, pilot log/launch, complete `pilot-01` hand archives/manifests and independent `pilot-01-verification.json`. Local transfer-verified copies: `/Users/dberweger/Local/hu20-turn-search-20261002/m4-pilot-01` and `m4-pilot-metadata-01`; helper/report provenance is retained in the same parent. Compact prospective count/power proof: `docs/reports/hu20-turn-search-artifacts/part-a-timing-freeze.json`. No artifact deletion or paid allocation is authorized by this entry.

- HU20 #148 final Part A: M4 `/Users/dberweger/Local/hu20-turn-search-20261003/part-a-01`, pushed source 5e4de63, 77,052 complete hands. Local full transfer-verified copy `/Users/dberweger/Local/hu20-turn-search-20261002/m4-part-a-01` includes original summary/manifest, replay, post-replay console failure and selection verification. Compact publication: `docs/reports/hu20-turn-search-part-a.md` and `hu20-turn-search-artifacts/part-a-result.json`. Preserve the original journal and attempts. External M4 harness uses a dedicated pinned upstream clone; build setup symlink failure/relocation and build/parity evidence remain at the same M4 root. No cleanup, paid allocation or original #145 binary replacement.

- HU20 #148 reference-bound calibration: full immutable ranges `/Users/dberweger/Local/hu20-turn-search-20261003/calibration-references-01.json` (19,002,241 bytes), proof `reference-freeze-proof-01.json`, external native sources/build logs/failures and `m4-parity-01` remain under the same M4 root. Compact index/settings are committed; all 144 selected-base references and original report/range/result hashes remain retrievable. Original #145 bytes/binary remain unchanged. No evidence eviction is authorized.

- HU20 #148 stopped calibration-01: M4 `/Users/dberweger/Local/hu20-turn-search-20261003/calibration-01` and full M1 copy `/Users/dberweger/Local/hu20-turn-search-20261002/m4-calibration-01`. All 94 manifest members (46,619,321 bytes) independently size/SHA verified. Guard/admission/launch/clock snapshots, 17 timeout rows, interrupted eighteenth request, logs and TensorBoard remain retained. Publication: [stopped curve](docs/reports/hu20-turn-search-calibration.md) and `docs/reports/hu20-turn-search-artifacts/calibration-01-stopped.json`. Family RSS guard stopped at 4.341 GiB against 4.25 GiB; processes exited, no swap growth, cumulative journal 5740.47 seconds. Recurring continuation is paused; [proposed readmission](docs/hu20-turn-search-resource-readmission.md) requires owner restart approval. No qualified setting, evidence eviction, automatic restart or rental.

- HU20 #148 owner-approved calibration-02: prospective settings `configs/diagnostics/hu20-turn-search-calibration-readmission-02.json`, source 2d761ca, fresh admission `docs/reports/hu20-turn-search-artifacts/calibration-02-admission.json`. Raw/preliminary admission and launch helpers retained at M4 `/Users/dberweger/Local/hu20-turn-search-20261003`; no new outcomes precede this freeze. Original stopped calibration-01 is retained separately, with no timing stitching or budget reset.

- HU20 calibration-02 is running from tested source 4105419, worker 18889/sidecar 18890. Final fresh admission snapshot was pushed at a5f4d48 before outcomes; preliminary snapshot is retained. Compact source/settings/admission/PID/clock provenance: [launch evidence](docs/reports/hu20-turn-search-artifacts/calibration-02-launch.json). Complete requests/responses/curve/status/TensorBoard/clock remain at M4 root `calibration-02`; local launch/helper/admission/validation evidence in `/Users/dberweger/Local/hu20-turn-search-20261002`. Original calibration-01 and #145 evidence stay untouched; no cleanup or paid allocation.

- HU20 #148 owner-approved clean stop of calibration-02: M4 `/Users/dberweger/Local/hu20-turn-search-20261003/calibration-02`, complete M1 copy `/Users/dberweger/Local/hu20-turn-search-20261002/m4-calibration-02-owner-stop`. All 1,752 manifest members / 1,614,849,743 bytes independently size/SHA verified. The full 238-row curve, requests/responses/receipts/manifests, TensorBoard, original incomplete summary, SIGTERM reason, owner stop request and append-preserving journal are retained. [Budget report](docs/reports/hu20-turn-search-calibration-02-budget.md), complete committed screen curve, stop/verification and timing-only forecast identify zero final rows, 18 unattempted reduced screen rows and no selected setting. Processes exited, swap unchanged, cumulative 12,053.465 seconds charged. The forecast-blocked settings proposal cannot launch; heartbeat PAUSED pending owner budget/scope decision. No calibration-01 stitching, evidence eviction or paid allocation. Earlier running/launch entries are historical.

- HU20 #148 approved retained-screen final preparation: `configs/diagnostics/hu20-turn-search-calibration-final-30-only-03.json` and `docs/reports/hu20-turn-search-artifacts/calibration-03-approved-forecast.json` prospectively bind the original 238-row screen hash, all six timing-selected configurations and 1,728 final coordinates. Three-hour river reserve enforced; 120-second continuation requires separate approval. Original calibration-02 remains immutable; new final evidence will use M4 `calibration-03`. No launch or cleanup is implied by this preparation entry.

- HU20 #148 calibration-03 resumed final: M4 `/Users/dberweger/Local/hu20-turn-search-20261003/calibration-03`, actual source 93d55eb (full CI passed), worker 87830/sidecar 87831. Fresh admission 70e445d preceded final outcomes. [Launch provenance](docs/reports/hu20-turn-search-artifacts/calibration-03-launch.json) pins settings/binary/source/admission/clock; full launch/helpers, charged preflight, CI proof and local status retained at `/Users/dberweger/Local/hu20-turn-search-20261002`. All 238 prior screen rows are copied with provenance; no screen rerun, no calibration-01 stitching or evidence cleanup. Expected 1,728 final coordinates, three-hour river reserve enforced. Preserve full raw requests/responses/shared receipts/curve/TensorBoard and original append-preserving journal. No paid allocation or automatic 120-second fallback.

- HU20 #148 completed calibration-03: all 1,728 final coordinates, no strict/relaxed thirty-second qualifier, all workers exited without guard failure. [Complete report](docs/reports/hu20-turn-search-calibration-03.md), [all 1,966 final/retained-screen rows](docs/reports/hu20-turn-search-artifacts/calibration-03-full-curve.jsonl) and [result/verification/journal](docs/reports/hu20-turn-search-artifacts/calibration-03-result.json) preserve every coordinate, law, receipt, timeout and exclusion. Raw M4 `/Users/dberweger/Local/hu20-turn-search-20261003/calibration-03`: 15,292 manifest members / 29,486,803,777 bytes independently size/SHA verified. Complete M1 retrieval `/Users/dberweger/Local/hu20-turn-search-20261002/m4-calibration-03-complete`, all 15,292 members independently size/SHA verified on M1; [retrieval proof](docs/reports/hu20-turn-search-artifacts/calibration-03-retrieval-verification.json) records final cumulative 42,294.351 seconds and retained TensorBoard hashes. The local publication/verifier helpers, admission, original journal snapshots and TensorBoard remain indexed there. Heartbeat PAUSED for owner decision; separate 120-second proposal/quote remains unapproved. Preserve originals and prior attempts; no eviction, rental, automatic restart or budget reset. Earlier running entries are historical.
