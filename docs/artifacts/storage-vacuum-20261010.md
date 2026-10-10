# M1/M4 storage cleanup — October 10, 2026

The owner requested additional storage for M1 K50 scoring (#225) and the fresh M4 HU100 4B/2B seed ladder (#226), and clarified there are two tasks. Current PR states and the active runners were checked before cleanup. No research was launched or retried.

| Machine | Native uploaded cache released | Measured original-removal gain | Free at verification |
| --- | ---: | ---: | ---: |
| M1 | 18.497 GB | 0 | 41.924 GB |
| M4 | 32.978 GB | 37.051 GB | 96.163 GB |

Sizes use decimal GB. Native cache numbers are the decrease in allocated file blocks; original-removal gains are adjacent `df` measurements. Free storage fluctuates with other work and VM allocation. M4's historical 87.5 GiB full-campaign estimate is 93.952 GB, pending pilot refinement; verification exceeds it by 2.211 GB. The estimate is not a guarantee that later admission checks pass.

## Removed originals and restoration

**819 unchanged manifested local paths** were removed on M4. The 739-path first batch released 32.529 GB; the 80-path second batch released 4.522 GB. Selected inodes with multiple hard links were removed only when every link belonged to the eligible manifested selection. The remaining 112 candidate links have external links and were retained.

The copies belong to merged #215/#223, archived #202 results, and #176/#185 immutable copies. Some #223 reference models restore from accepted #207, and #185 copied references restore from accepted #176. Canonical #207 originals remain available for maintained v0.5 retrieval paths. Current cloud metadata and native uploaded/no-pending/no-conflicts/size checks agreed before removal.

[Full cleanup receipt](https://drive.google.com/file/d/1cOGjCi1ZPSYGiO-3X22h0GO6RDcWHPBR/view?usp=drivesdk) (970,734 bytes, SHA256 `c25312817f3d4393f77005781200fd5ca0f683cd2795cda996d8edb4cab00c40`; confirmed cloud ID/name/size/Research-Cloud parent).

[Every removed path, byte size, restoration archive/member and member SHA256](storage-vacuum-20261010-locators.md). The full receipt also records source size/inode/device/mtime/ctime/link counts, archive SHA256s, native/cloud acceptance, removed journals, retained paths and verification.

To restore, download the recorded accepted archive into a fresh ignored **nonsynced** working directory. Verify the indexed archive SHA256, extract the named member, and verify its indexed SHA256 before research use. Put it at the recorded local path; recreate hard-link aliases from their canonical member where appropriate. The table names the actual source archive even when a newer bundle omitted duplicate reference models.

## Native cache release

Ten M1 and six M4 files used the supported `FileManager.evictUbiquitousItem` API after uploaded/unpinned/inactive checks. No synced file was deleted or unpinned. The placeholders remain, all 16 logical sizes and Drive IDs are unchanged, and allocated blocks are zero. Independent cloud readback confirms unchanged IDs/names/sizes/parents for all relevant archives and the receipt. Open/download the existing cloud item when a local copy is needed again.

M4 #215/#223 ZIP caches and the accepted complete #207 retry ZIP are unused by #226's fresh training runner, which checks Git-pinned gate hashes against its own outputs. Their original-model eligibility was separately reviewed. The incomplete original #207 synced ZIP was retained.

## Protected work and limits

Open #225/#226 work roots, K50 inputs/adapters and #149/#163/#190/#222 archives, source/build/environments/shared Git, canonical #207 model originals, the incomplete #207 archive, external hard links, changed/unarchived files and personal data remain protected. CrossOver and Xcode were retained. Existing manifests and restoration provenance were used without remote archive downloads or repeated payload/member hash audits.

M1's scoring admission also depends on AC power, total swap and pressure guards; this storage cleanup changes none of those conditions. No apps were stopped and no unattended maintenance was scheduled.
