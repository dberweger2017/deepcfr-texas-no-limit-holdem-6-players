# Storage cleanup after overnight closeout — October 14, 2026

The owner requested another M1/M4 storage pass after the overnight PRs finished.
Current GitHub status confirms every selected owner merged, including #265 and
#268; no open PR was returned. The cleanup released **62.996 GB**
from eight uploaded archive caches and 17 unchanged archived originals.
M1 released **4.373 GB**; M4 **58.623 GB**.
Verified free space: **M1 59.304 GB /M4 83.146 GB**.
The per-file accounting is distinct from filesystem free-space changes.

## Acceptance and dependencies

The previously pending #265 full ZIP is now confirmed uploaded: native
uploaded1/uploading0/conflicts0, independent Drive ID/name/size/direct-parent
acceptance. File ID `1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI`, 26,698,069,587 bytes,
parent `1qrozNU9QvCgN5QTF1OKyQZxhmUHlb6dK`. Its existing whole SHA256 is
`5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`;
embedded `ARCHIVE-MANIFEST.json` SHA256
`26760f63abd5da6ffd3c42b18f93cba27707a842b7d9963f781c8b3d891663ce`.
This supersedes the earlier pending snapshot for acceptance, preserving that
historical receipt. The scientific campaign itself performed no further cleanup.

For original retirement, existing sealing receipts prove local member readback;
fresh cloud acceptance precedes removal. Every selected original matches its
recorded exact source path, byte count and nanosecond mtime; ctime predates the
sealing receipt. Fresh process/lsof review finds no active train/export or open
selected file. PR268's config points directly to the retained #265 5B/final
averages and tables. The #248 200M average, #250 752M baseline, prepared Slumbot
server model/cache/launcher, local model/table inputs, #268 unarchived policy
caches, source, environments, executables, shared Git and uncertain files remain.
All 17 removed paths were checked absent; all three protected averages retain
their exact sizes and mtimes. No worktree was removed.

Cache release used only supported per-file `FileManager.evictUbiquitousItem`
after uploaded/no-pending/no-conflicts/unpinned/not-downloading/not-paused/
not-excluded/not-trashed acceptance, cloud identity/parent and fresh open-file
checks. Same-ID/same-size Drive metadata and local dataless placeholders remain
afterward. Counted deltas exclude remaining provider bookkeeping.
No synced deletion, forced offload, unpinning, remote archive download, repeated
whole/member payload hash audit, personal cleanup, app shutdown or scheduling.
Only the small existing PR265 manifest metadata digest was checked.

## Cache restoration

Opening/downloading each unchanged Drive file restores its archive cache.
Research use still requires the existing whole/member retrieval verification
into fresh ignored nonsynced storage.

| Mac | Native cache path | Unchanged Drive file | Bytes released |
| --- | --- | --- | ---: |
| M1 | `~/Local/Research-Cloud/PR-262-frozen-table/frozen-table-evidence-M1-20261013.zip` | [frozen-table-evidence-M1-20261013.zip](https://drive.google.com/file/d/1cOtEtzdpo7EP25ocJJAzOYpDJE5avmpS/view?usp=drivesdk) | 669,196,288 |
| M1 | `~/Local/Research-Cloud/PR-267-hu200-missing-key-fallback/hu200-missing-key-fallback-PR267-20261013.zip` | [hu200-missing-key-fallback-PR267-20261013.zip](https://drive.google.com/file/d/1gDOgSgoEbAdWjUplAK2hgwCd4KwuJeUB/view?usp=drivesdk) | 375,513,088 |
| M4 | `~/Local/Research-Cloud/PR-265-hu200-k50-frozen/hu200-k50-frozen-full-M4-20261013.zip` | [hu200-k50-frozen-full-M4-20261013.zip](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view?usp=drivesdk) | 26,698,072,064 |
| M4 | `~/Local/Research-Cloud/PR-265-hu200-k50-frozen/hu200-k50-frozen-averages-M4-20261013.zip` | [hu200-k50-frozen-averages-M4-20261013.zip](https://drive.google.com/file/d/1iVND4K_trvJ8S6gnDnyDc4tj5fNR2mDh/view?usp=drivesdk) | 5,921,071,104 |
| M4 | `~/Local/Research-Cloud/PR-250-hu200-k50-first/hu200-k50-first-averages-M4-20261013.zip` | [hu200-k50-first-averages-M4-20261013.zip](https://drive.google.com/file/d/1DMz8kmY4DIQfO-NSIdEiOu9k-tl1hzrB/view?usp=drivesdk) | 2,221,375,488 |
| M4 | `~/Local/Research-Cloud/PR-163-equity-buckets/hu20-equity-buckets-20261005.tar.gz` | [hu20-equity-buckets-20261005.tar.gz](https://drive.google.com/file/d/1Gl1JYWN0F7KFJbwoKtcA0zB_rUsWWZ6T/view?usp=drivesdk) | 1,991,950,336 |
| M4 | `~/Local/Research-Cloud/PR-268-hu200-frozen-evaluation/hu200-frozen-evaluation-M4-20261014.zip` | [hu200-frozen-evaluation-M4-20261014.zip](https://drive.google.com/file/d/1EwsMAG3ZdX7SVW1SIigpf8dgr5Vgd02_/view?usp=drivesdk) | 971,530,240 |
| M4 | `~/Local/Research-Cloud/PR-265-hu200-k50-frozen/hu200-k50-frozen-closeout-M4-20261014.zip` | [hu200-k50-frozen-closeout-M4-20261014.zip](https://drive.google.com/file/d/1WuM9CHSwQUcJCfwKpsc0es6HZBLzaWuf/view?usp=drivesdk) | 6,000,640 |

## Removed originals and exact restoration

Download the linked archive into a new ignored nonsynced restore directory,
verify the listed whole SHA256, extract the exact member and verify its listed
size/SHA256 before reuse. Keep the embedded manifest and check its digest against
the manifest pin below. These commands restore evidence; they launch no science.

- **M1 `~/Local/hu200-200m-model-20261013/results/hu200-200m-20261013/HU200-100000000.gz`** — 759,275,650 bytes, 759,279,616 allocated bytes retired. [Archive `HU200-200M-M1-evidence-20261013.zip`](https://drive.google.com/file/d/1z0vFrM0ChRs1TIErlnxip467IIWaUfqM/view?usp=drivesdk), ID `1z0vFrM0ChRs1TIErlnxip467IIWaUfqM`, 4,132,696,783 bytes; whole SHA256 `f3f169bd93e67d6b72d195747bff5989789ebd4a0eaafdaf4984af0481aef794`. Member `HU200-100000000.gz`, 759,275,650 bytes, SHA256 `f340b44cbaad923be4851cd7628f3513a937d900d7e5039dbabfea105ecfc2de`; `ARCHIVE-MANIFEST.json` SHA256 `6da6a8b48e65f838d5cdcdeb61e1f556d4a89a843f3dae8dd013a31469a0ab12`.

- **M1 `~/Local/hu200-200m-model-20261013/results/hu200-200m-20261013/HU200-200000000.gz`** — 1,206,916,078 bytes, 1,206,919,168 allocated bytes retired. [Archive `HU200-200M-M1-evidence-20261013.zip`](https://drive.google.com/file/d/1z0vFrM0ChRs1TIErlnxip467IIWaUfqM/view?usp=drivesdk), ID `1z0vFrM0ChRs1TIErlnxip467IIWaUfqM`, 4,132,696,783 bytes; whole SHA256 `f3f169bd93e67d6b72d195747bff5989789ebd4a0eaafdaf4984af0481aef794`. Member `HU200-200000000.gz`, 1,206,916,078 bytes, SHA256 `bfe4d7295a88489683d96f7b9881e316b5cb6cb0ab0f3086ee6a78bfce665d0a`; `ARCHIVE-MANIFEST.json` SHA256 `6da6a8b48e65f838d5cdcdeb61e1f556d4a89a843f3dae8dd013a31469a0ab12`.

- **M1 `~/Local/hu200-200m-model-20261013/results/hu200-200m-20261013/HU200-200000000.gz.current.gz`** — 777,323,215 bytes, 777,326,592 allocated bytes retired. [Archive `HU200-200M-M1-evidence-20261013.zip`](https://drive.google.com/file/d/1z0vFrM0ChRs1TIErlnxip467IIWaUfqM/view?usp=drivesdk), ID `1z0vFrM0ChRs1TIErlnxip467IIWaUfqM`, 4,132,696,783 bytes; whole SHA256 `f3f169bd93e67d6b72d195747bff5989789ebd4a0eaafdaf4984af0481aef794`. Member `HU200-200000000.gz.current.gz`, 777,323,215 bytes, SHA256 `b25b02eb855e07b23d86c83b23036b0c0eaa30d7ceb7246ce025257f56fe7503`; `ARCHIVE-MANIFEST.json` SHA256 `6da6a8b48e65f838d5cdcdeb61e1f556d4a89a843f3dae8dd013a31469a0ab12`.

- **M1 `~/Local/hu200-200m-model-20261013/results/hu200-200m-20261013/recovery-full-20m.gz`** — 227,890,921 bytes, 227,893,248 allocated bytes retired. [Archive `HU200-200M-M1-evidence-20261013.zip`](https://drive.google.com/file/d/1z0vFrM0ChRs1TIErlnxip467IIWaUfqM/view?usp=drivesdk), ID `1z0vFrM0ChRs1TIErlnxip467IIWaUfqM`, 4,132,696,783 bytes; whole SHA256 `f3f169bd93e67d6b72d195747bff5989789ebd4a0eaafdaf4984af0481aef794`. Member `recovery-full-20m.gz`, 227,890,921 bytes, SHA256 `c60977f8cff235886914377a2c8483a25af1bb894d731aaded4e60ab11c475f5`; `ARCHIVE-MANIFEST.json` SHA256 `6da6a8b48e65f838d5cdcdeb61e1f556d4a89a843f3dae8dd013a31469a0ab12`.

- **M1 `~/Local/hu200-200m-model-20261013/results/hu200-200m-20261013/recovery-part-10m.gz`** — 128,886,941 bytes, 128,888,832 allocated bytes retired. [Archive `HU200-200M-M1-evidence-20261013.zip`](https://drive.google.com/file/d/1z0vFrM0ChRs1TIErlnxip467IIWaUfqM/view?usp=drivesdk), ID `1z0vFrM0ChRs1TIErlnxip467IIWaUfqM`, 4,132,696,783 bytes; whole SHA256 `f3f169bd93e67d6b72d195747bff5989789ebd4a0eaafdaf4984af0481aef794`. Member `recovery-part-10m.gz`, 128,886,941 bytes, SHA256 `a39746c376656f3e5e2b6b6199e3054692ce1d07e073f8568c291e774d3347e1`; `ARCHIVE-MANIFEST.json` SHA256 `6da6a8b48e65f838d5cdcdeb61e1f556d4a89a843f3dae8dd013a31469a0ab12`.

- **M1 `~/Local/hu200-200m-model-20261013/results/hu200-200m-20261013/recovery-resumed-20m.gz`** — 227,890,921 bytes, 227,893,248 allocated bytes retired. [Archive `HU200-200M-M1-evidence-20261013.zip`](https://drive.google.com/file/d/1z0vFrM0ChRs1TIErlnxip467IIWaUfqM/view?usp=drivesdk), ID `1z0vFrM0ChRs1TIErlnxip467IIWaUfqM`, 4,132,696,783 bytes; whole SHA256 `f3f169bd93e67d6b72d195747bff5989789ebd4a0eaafdaf4984af0481aef794`. Member `recovery-resumed-20m.gz`, 227,890,921 bytes, SHA256 `c60977f8cff235886914377a2c8483a25af1bb894d731aaded4e60ab11c475f5`; `ARCHIVE-MANIFEST.json` SHA256 `6da6a8b48e65f838d5cdcdeb61e1f556d4a89a843f3dae8dd013a31469a0ab12`.

- **M4 `~/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/archive-inputs/stage2-science-resources.jsonl`** — 3,854,054 bytes, 3,854,336 allocated bytes retired. [Archive `hu200-k50-frozen-full-M4-20261013.zip`](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view?usp=drivesdk), ID `1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI`, 26,698,069,587 bytes; whole SHA256 `5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`. Member `research/archive-inputs/stage2-science-resources.jsonl`, 3,854,054 bytes, SHA256 `6946d1c9b36aa0970be7f69c4543a8a20621b47f4f3405ac5797a294e2b8124f`; `ARCHIVE-MANIFEST.json` SHA256 `26760f63abd5da6ffd3c42b18f93cba27707a842b7d9963f781c8b3d891663ce`.

- **M4 `~/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/frozen/20000000000.current.gz`** — 2,301,966,028 bytes, 2,301,968,384 allocated bytes retired. [Archive `hu200-k50-frozen-full-M4-20261013.zip`](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view?usp=drivesdk), ID `1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI`, 26,698,069,587 bytes; whole SHA256 `5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`. Member `research/frozen/20000000000.current.gz`, 2,301,966,028 bytes, SHA256 `352f233a86e422702877a2b6d14eb47c02806f3f2d8ab6a68f688a30ca353688`; `ARCHIVE-MANIFEST.json` SHA256 `26760f63abd5da6ffd3c42b18f93cba27707a842b7d9963f781c8b3d891663ce`.

- **M4 `~/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/frozen/20000000000.gz`** — 4,960,844,586 bytes, 4,966,363,136 allocated bytes retired. [Archive `hu200-k50-frozen-full-M4-20261013.zip`](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view?usp=drivesdk), ID `1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI`, 26,698,069,587 bytes; whole SHA256 `5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`. Member `research/frozen/20000000000.gz`, 4,960,844,586 bytes, SHA256 `c5929bda17457911347e47cd77a83dd8b34618a5542731bed96ece1fb4288b6f`; `ARCHIVE-MANIFEST.json` SHA256 `26760f63abd5da6ffd3c42b18f93cba27707a842b7d9963f781c8b3d891663ce`.

- **M4 `~/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/frozen/5000000000.current.gz`** — 2,289,968,280 bytes, 2,298,490,880 allocated bytes retired. [Archive `hu200-k50-frozen-full-M4-20261013.zip`](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view?usp=drivesdk), ID `1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI`, 26,698,069,587 bytes; whole SHA256 `5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`. Member `research/frozen/5000000000.current.gz`, 2,289,968,280 bytes, SHA256 `b9c4961b6de9b214950d4d272a220312b5b8445a433d4046e1b6ee37b786ebf7`; `ARCHIVE-MANIFEST.json` SHA256 `26760f63abd5da6ffd3c42b18f93cba27707a842b7d9963f781c8b3d891663ce`.

- **M4 `~/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/frozen/5000000000.gz`** — 4,357,232,664 bytes, 4,357,234,688 allocated bytes retired. [Archive `hu200-k50-frozen-full-M4-20261013.zip`](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view?usp=drivesdk), ID `1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI`, 26,698,069,587 bytes; whole SHA256 `5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`. Member `research/frozen/5000000000.gz`, 4,357,232,664 bytes, SHA256 `56244a81d90575eb4c7f73992ccdf36a304d778923076850cadc1531afe21721`; `ARCHIVE-MANIFEST.json` SHA256 `26760f63abd5da6ffd3c42b18f93cba27707a842b7d9963f781c8b3d891663ce`.

- **M4 `~/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/gates/fresh20m.gz`** — 270,143,830 bytes, 286,089,216 allocated bytes retired. [Archive `hu200-k50-frozen-full-M4-20261013.zip`](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view?usp=drivesdk), ID `1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI`, 26,698,069,587 bytes; whole SHA256 `5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`. Member `research/gates/fresh20m.gz`, 270,143,830 bytes, SHA256 `4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b`; `ARCHIVE-MANIFEST.json` SHA256 `26760f63abd5da6ffd3c42b18f93cba27707a842b7d9963f781c8b3d891663ce`.

- **M4 `~/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/gates/prefix10m.gz`** — 148,448,690 bytes, 148,451,328 allocated bytes retired. [Archive `hu200-k50-frozen-full-M4-20261013.zip`](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view?usp=drivesdk), ID `1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI`, 26,698,069,587 bytes; whole SHA256 `5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`. Member `research/gates/prefix10m.gz`, 148,448,690 bytes, SHA256 `fc102fc232fa4376627bbcd10af5f77a0b8fdb25f8a62cf81d224de5fde9e5db`; `ARCHIVE-MANIFEST.json` SHA256 `26760f63abd5da6ffd3c42b18f93cba27707a842b7d9963f781c8b3d891663ce`.

- **M4 `~/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/gates/resumed20m.gz`** — 270,143,830 bytes, 270,147,584 allocated bytes retired. [Archive `hu200-k50-frozen-full-M4-20261013.zip`](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view?usp=drivesdk), ID `1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI`, 26,698,069,587 bytes; whole SHA256 `5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`. Member `research/gates/resumed20m.gz`, 270,143,830 bytes, SHA256 `4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b`; `ARCHIVE-MANIFEST.json` SHA256 `26760f63abd5da6ffd3c42b18f93cba27707a842b7d9963f781c8b3d891663ce`.

- **M4 `~/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/retention-resources.jsonl`** — 1,996,116 bytes, 2,101,248 allocated bytes retired. [Archive `hu200-k50-frozen-full-M4-20261013.zip`](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view?usp=drivesdk), ID `1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI`, 26,698,069,587 bytes; whole SHA256 `5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`. Member `research/retention-resources.jsonl`, 1,996,116 bytes, SHA256 `2d558a12c793b6ad4d6f6715f6f4703f477c2f7b433ecfc1cdc90fc13b08d8de`; `ARCHIVE-MANIFEST.json` SHA256 `26760f63abd5da6ffd3c42b18f93cba27707a842b7d9963f781c8b3d891663ce`.

- **M4 `~/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/stage1/1000000000.gz`** — 3,442,723,146 bytes, 3,442,724,864 allocated bytes retired. [Archive `hu200-k50-frozen-full-M4-20261013.zip`](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view?usp=drivesdk), ID `1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI`, 26,698,069,587 bytes; whole SHA256 `5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`. Member `research/stage1/1000000000.gz`, 3,442,723,146 bytes, SHA256 `4b25e8029ed0ebfbeeb42ac02dee3179303e7f2300c23c21d32d27382f390da4`; `ARCHIVE-MANIFEST.json` SHA256 `26760f63abd5da6ffd3c42b18f93cba27707a842b7d9963f781c8b3d891663ce`.

- **M4 `~/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/stage1/500000000.gz`** — 2,727,516,803 bytes, 2,735,321,088 allocated bytes retired. [Archive `hu200-k50-frozen-full-M4-20261013.zip`](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view?usp=drivesdk), ID `1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI`, 26,698,069,587 bytes; whole SHA256 `5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`. Member `research/stage1/500000000.gz`, 2,727,516,803 bytes, SHA256 `2c4c1484b7e4ebefca2bb364223d6cd12e886f98b20784b51ff6ac6678f4312d`; `ARCHIVE-MANIFEST.json` SHA256 `26760f63abd5da6ffd3c42b18f93cba27707a842b7d9963f781c8b3d891663ce`.

```sh
# ARCHIVE.zip is the pinned Drive file downloaded above.
shasum -a 256 ARCHIVE.zip
python -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extract(sys.argv[2],sys.argv[3])' ARCHIVE.zip EXACT_MEMBER NEW_RESTORE_ROOT
shasum -a 256 NEW_RESTORE_ROOT/EXACT_MEMBER
```

[Full private receipt](https://drive.google.com/file/d/1LYG6QcCFqP0d1gBEHIzbgJzlLnU3y4w4/view?usp=drivesdk), ID `1LYG6QcCFqP0d1gBEHIzbgJzlLnU3y4w4`, **110,507 bytes**, SHA256 `a479d640cd8061f076238e8f84695dcbb01c19bc4a95b48086eee6dbcc836f4b`. Independent metadata readback confirms this exact name/size and Research-Cloud parent `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. It preserves per-file acceptance/stat/journal/cloud records, dependency review and verification.
