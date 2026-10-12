# M4 uploaded archive cache release — October 12, 2026

The owner requested space for open PRs. Only [#238](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/238), the M4 HU20 K50 5B first pass, was open. Its prospective reserve is 35.719 GiB, rounded to a 38 GiB entry target including the 16 GiB floor. M4 had about 23.7 GB free; M1 had about 80 GB and needed no cleanup.

| Machine | Allocated native cache released | Free at verification |
| --- | ---: | ---: |
| M1 | 0 | 80.039 GB |
| M4 | 19.778 GB | 43.364 GB /40.386 GiB |

Sizes marked GB are decimal. M4 verification exceeds the original 38 GiB entry target by 2.562 GB. Free space changes with other work, VM allocation, retrievals and new outputs. The owning task still measures resources before each stage; cleanup changes no research guard, budget, timer or start authorization.

## Single cache item and restoration

The only released item was merged #236's main scientific ZIP cache:

- Local native path: `/Users/dberweger/Library/CloudStorage/GoogleDrive-dberweger2017@gmail.com/My Drive/deepcfr-research-results/PR-236-hu100-4b-seed-replication/hu100-4b-seed-replication-M4-20261011.zip`.
- [Same preserved Drive file](https://drive.google.com/file/d/1SsCcSHSsXNPKW0q9rq7rqlXv8jzXPGK4/view), ID `1SsCcSHSsXNPKW0q9rq7rqlXv8jzXPGK4`, logical size **19,777,801,324 bytes**, parent `1OUNJcDqZLFRXsffKrtjDfliumj_DmdaI`.
- Before allocated blocks: **19,784,589,312 bytes**; after: **6,787,072 bytes**. The same-size/same-ID dataless placeholder remains. No claim of zero residual allocation is made.
- Whole archive SHA256 `7cce272f16d292c62c62cdb0ac6b27e007fb51ef1460bb8e04fd9934f686ecb5`; existing embedded `ARCHIVE-MANIFEST.json` SHA256 `3bfe24115e1ee069dc0f146812bbc3ac93d2f72c97dc21ffe05bbf096c45400c`. The already-local manifest was saved before release; no archive payload was downloaded or re-audited.

Use the existing cloud item to restore its local cache. For research model retrieval, follow [#236's exact model/member index](../reports/hu100-4b-seed-replication-artifacts/model-index.json): retrieve into a fresh ignored nonsynced directory, verify whole archive SHA256, extract the required member and verify its indexed SHA256 before use.

## Acceptance and protection

Current GitHub status confirmed #236 merged at 04:29:05 UTC on October 12 before the action. Native checks confirmed uploaded, unpinned, no pending upload/download, unresolved conflicts, sync pause, trash or exclusion; cloud ID/name/size/parent matched. The unused HU100 ZIP is not an active HU20 #238 input. Open-handle review and exact size/inode/device/mtime/ctime checks preceded supported `FileManager.evictUbiquitousItem`; the action was fsynced to a journal. Post-action native and independent cloud readbacks preserved the same item and logical size.

**No research originals were removed.** Both #236's entire results root and #238's work root remain; #238's HU20 v1/K50 models, tables, gate references, controls and v0.4.2 inputs are protected. Other archives, incomplete failure files, source/build/environments/shared Git, personal files, CrossOver and Xcode were retained. No synced deletion, force offload, unpinning, app shutdown, science launch/retry or schedule change occurred in this cleanup.

[Full cleanup receipt](https://drive.google.com/file/d/19s6rartMoLUjyiCQZv3Iq8ydh2ynWjEP/view): **481,394 bytes**, SHA256 `922ad752d9874785e98035550de499f139cb6fdeaa6f5bc7404394ac253b0b8d`, independently confirmed cloud ID/name/size/Research-Cloud parent. It includes preflight/current PR status, native/cloud metadata, preserved existing manifest, journal, exact allocation measurements, active root and verification. The receipt and metadata are also retained at `~/Local/storage-vacuum-20261012/` on both Macs.
