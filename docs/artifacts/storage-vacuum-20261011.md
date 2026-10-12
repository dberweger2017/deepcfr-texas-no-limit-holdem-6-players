# Storage cleanup for the new M1/M4 tasks — October 11, 2026

The owner requested storage for M1 K50 evaluation support ([#235](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/235)) and M4 HU100 4B replication ([#236](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/236)). Both task roots and dependencies were reviewed. M1 already had enough space; cleanup ran only on M4 under standing owner authorization.

| Machine | Uploaded native cache released | Measured original-removal gain | Free at verification |
| --- | ---: | ---: | ---: |
| M1 | 0 | 0 | 86.501 GB |
| M4 | 42.403 GB | 27.749 GB | 97.866 GB |

Sizes are decimal GB. Cache release measures the decrease in allocated file blocks; original removal uses adjacent `df` measurements. Concurrent work and VM allocation change free space. M4 verification exceeds the new task's provisional **83.287 GB /77.567 GiB** requirement by **14.579 GB**. That estimate includes its 16 GiB floor; the research task still needs fresh admission and pilot refinement before science.

## Removed originals and restoration

**49 unchanged original files** from merged #226 were removed under `~/Local/hu100-4b-seed-ladder-20261010/results/hu100-4b-seed-ladder/`. These are sealed traces, logs, timing output and model files that the new task does not use. Existing manifest source paths, sizes and exact mtimes were checked; ctime must predate the completed archive receipt. Omitted historical model copies were checked against the executed campaign's byte-exact size/hash receipts, their checked source paths and modification history. No payload hashes were repeated.

Current live merged states and native/cloud upload acceptance were checked for #226 and its #207/#215/#223 restoration archives. Every selected inode was singly linked; there were no external links in the removal selection. Changed `preparation/T3-HANDOFF.md` and `preparation/scheduler-state.json` were retained.

[Every removed path and exact restoration archive/member/hash](storage-vacuum-20261011-locators.md).
[Full cleanup receipt](https://drive.google.com/file/d/1lj6nZVoI8vSlsxY2OYluTdnmPDPCmxp9/view): **444,294 bytes**, SHA256 `bc6666cd7ad25c0d9f3a073772462523d97f7214de42a718c2126db2719d4b0b`, confirmed cloud ID/name/size/Research-Cloud parent. It includes the existing manifest, current archive acceptance, source stats, preflight plan, fsynced removal/release journals, protected models and verification.

Restore by downloading the specified accepted ZIP into a fresh ignored nonsynced location. Verify its indexed whole SHA256, extract the named member, verify its member SHA256, then place it at the recorded path. The model reference archive can differ from the local owning PR because #226 omitted historical duplicate bytes.

## Native cache and failure preservation

Four uploaded, unpinned, unused files used supported `FileManager.evictUbiquitousItem`. All four same-ID/same-size placeholders remain with zero allocated blocks, and independent cloud readback confirms unchanged IDs/names/sizes/parents. No synced file was deleted, force-offloaded or unpinned. Opening the existing cloud file retrieves its contents.

The accepted #226 recovery ZIP, #222 partial-evidence ZIP and #232 scoring ZIP caches were released. Their existing embedded manifests were saved before release. A separate acceptance check also confirmed that Drive had fully uploaded the **19,314,987,621-byte incomplete #226 original ZIP** ([failure file](https://drive.google.com/file/d/1UlBGKIUnjZGnQh8INUpkHUO-aMNuCvTj/view)). Its native cache was released, while the failure file and placeholder are preserved. It still has no ZIP end-directory and remains **unaccepted as a scientific archive**. No original removal relies on that incomplete file. #226's scientific backup remains its separately accepted recovery archive.

The incomplete #207 original ZIP and its local cache remain untouched. Historical descriptions of all #226 originals being retained are superseded only for the 49 paths listed in this receipt; the incomplete #226 file is preserved online with its placeholder.

## Protected work

The two new task roots, K50 tables and v0.4.2 inputs, canonical #207 originals, source/build/environments/shared Git, changed or unarchived files, personal data, CrossOver and Xcode were retained. No apps were stopped, research launched/retried or unattended cleanup scheduled. No remote archive bytes were downloaded or audited for cleanup.

Six #226 model files remain unchanged, checked by exact size/inode/device/mtime/ctime after cleanup:

| Seed | Milestone | Retained file | Purpose |
| --- | --- | --- | --- |
| 2026100601 | 2B | average.gz | Scripted comparator |
| 2026100601 | 4B | average.gz | Candidate |
| 2026100901 | 1B and 2B | checkpoint.gz at each milestone | Exactness reference gates |
| 2026100902 | 1B and 2B | checkpoint.gz at each milestone | Exactness reference gates |

They remain under the original #226 `training/SEED/MILESTONE/` paths. The new task verifies the indexed hashes before use and can compare checkpoint rows if its header cap differs. Its current/average export gates use Git-pinned hashes against freshly generated outputs. Previous direct hand traces are historical evidence restored through the locators when needed; compact frozen schedules/specifications remain locally available for freshness checks.
