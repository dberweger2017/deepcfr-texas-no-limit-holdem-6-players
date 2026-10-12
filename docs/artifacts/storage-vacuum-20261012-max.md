# M1/M4 extended cleanup — October 12, 2026

The owner requested as much reclamation as possible under the existing preservation rules, without a fixed target. The pass retired **109.818 GB of existing allocated copies**, including one idle committed Git worktree. Active K50 continuation #242 and native-build #243 work, #238's input archive/retained binary, tables, current released-model caches and canonical #207 originals remain protected.

| Machine | Existing native cache released | Unique original allocation removed | Worktree measured gain | Free at verification |
| --- | ---: | ---: | ---: | ---: |
| M1 | 38.774 GB | 2.297 GB | 0.259 GB | 95.980 GB |
| M4 | 3.355 GB | 65.134 GB | 0 | 102.388 GB |

GB is decimal. The newly uploaded calibration copy's 14.318 GB cache was also released; it is **excluded from the existing-copy total** because this cleanup first created that cache. The calibration original/cache releases share one `df` window; the receipt uses unique original allocation and separate cache journals instead of counting that combined window as original-only gain. Free storage changes with other work, retrievals, VM allocation and cache behavior; no other agent's reclamation is attributed to this pass.

## Removed research originals

**192 exact paths** were removed: 18 inactive #165 M1 training/policy copies; 167 #236 M4 recorded payload/dependency copies; six #226 M4 candidate/reference copies; and #236's separately preserved calibration ZIP. [Every path and exact restoration locator](storage-vacuum-20261012-max-locators.md). Current merged status and independent native/cloud archive acceptance were checked before removal. Existing manifests, byte-exact dependency/restoration receipts and prior protected identity records establish the paths and hashes; exact sizes/mtimes and sealed ctimes or unchanged prior identity establish modification history. Every hard link in a selected inode group was accounted for; external/uncertain links would be retained.

[Full private cleanup receipt](https://drive.google.com/file/d/1iDEI_hToPPf-D9xOXvX_PbijpmWD20F4/view?usp=drivesdk): **5,735,125 bytes**, SHA256 `5a884ece439f6da732debfc6cbaa5c72754b61e927746c6be63710844f46b956`, confirmed cloud ID/name/size/Research-Cloud parent. It includes native/cloud acceptance, source stats, manifests, fsynced mutation journals, retained items, dependency review, worktree audit and final verification. Local receipts remain under `~/Local/storage-vacuum-20261012-max/` on both Macs.

Restore an indexed member by downloading its accepted archive into a fresh ignored nonsynced directory, verifying the indexed whole SHA256, extracting only the named member and verifying its member SHA256 before use. Recreate hard-link aliases from the canonical restored member where appropriate. The calibration row restores the whole ZIP, not an inner member. Source/build/environments outside the retired committed worktree were retained. Compact reports, schedules/specifications, current release downloads and changed/unarchived files remain.

## Calibration archive preserved before removal

The **14,318,147,951-byte** `results/hu100-4b-seed-replication/archive-calibrations/initial.zip` had been locally verified by #236 but never uploaded. Rather than remove an unarchived bundle, cleanup copied this existing sealed ZIP into the existing #236 folder as **hu100-4b-seed-replication-calibration-initial-20261011.zip**.

[Accepted calibration file](https://drive.google.com/file/d/1UwytNKE5M0GxAPJyicpSE81nX3yD7Ozu/view?usp=drivesdk), ID `1UwytNKE5M0GxAPJyicpSE81nX3yD7Ozu`, parent `1OUNJcDqZLFRXsffKrtjDfliumj_DmdaI`. Whole SHA256 `2f0be37f6583c8ab11ece4a910d3ad421f5fc247d94ebe0377433f6045f6324e`; embedded `ARCHIVE-MANIFEST.json` SHA256 `149d31b89e4e137527aff3e89d61ac3acaeca9f44a21fc15eef6a2f98c172738`. The existing receipt records all 542 member sizes/hashes and whole-archive verification. No new packing, remote download or repeated payload/member audit occurred; unchanged source identity/history, successful local copy and matching logical byte counts were checked, then native uploaded/no-pending/no-conflicts and independent cloud ID/name/size/parent acceptance passed. Only then were the unchanged original and unpinned native cache removed/released. This preserves the actual packing-calibration container and provenance, including its historical `cloud_upload_accepted=false` snapshot alongside current acceptance.

## Cache and failure preservation

**819 uploaded, unpinned, unused cache items** used supported `FileManager.evictUbiquitousItem`: 817 M1 files and two M4 files. All 819 same-ID/same-size local placeholders remain with zero allocated blocks at verification. Independent post-action cloud metadata agrees on IDs/names/sizes/parents. No synced file was deleted, force-offloaded or unpinned. Open the existing cloud item to restore its cache.

The M4 items are the newly accepted calibration ZIP and the **3,354,661,317-byte incomplete #207 original ZIP** ([preserved failure file](https://drive.google.com/file/d/17ns90mF3b00uoWc_nMzXLWS0dvmKNZ05/view)). Drive separately confirms this failure file is fully uploaded. Its contents, failure status and placeholder remain preserved online; it is still **unaccepted as a scientific archive**, and no research-original removal relies on it. Canonical #207 originals and its accepted complete retry archive remain. Earlier descriptions of its locally retained cache are superseded only for this cache release.

M1 cache records cover merged #148/#190/#218/#222/#225/#226/#235 items. The full receipt names every exact cache path and restoration URL; none of those cloud files was deleted. Existing local ZIP manifests were saved before release where present.

## Idle committed worktree

One clean committed worktree older than 24 hours, with no untracked/ignored evidence, open handles or maintained source/config/test references, was retired with `git worktree remove`: `/Users/dberweger/.t3/worktrees/deepcfr-texas-no-limit-holdem-6-players/t3-0ad1f6fd`. Source `7e51349ecb886a69b2c4efafdafad0998884a6d1` is reachable from current main; retained branch `refs/heads/t3/hu100-1b-growth` and shared Git objects are unchanged. Restore with:

```sh
git worktree add /Users/dberweger/.t3/worktrees/deepcfr-texas-no-limit-holdem-6-players/t3-0ad1f6fd t3/hu100-1b-growth
```

The audit retained 23 other worktrees for primary/nested canonical use, open dependencies, recent activity or dirty/ignored/unarchived files. Personal projects/data, CrossOver, Xcode, active work roots and other agents' inputs remain protected. No research was launched/retried and no app, schedule, budget or guard was changed.
