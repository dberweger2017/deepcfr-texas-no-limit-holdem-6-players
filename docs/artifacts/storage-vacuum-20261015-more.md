# More storage on M1 and M4 — October 15, 2026

Owner requested another cleanup on both Macs. This pass retires **8,370,188,288
allocated bytes /8.370 GB on M1** and **7,675,281,408 bytes /7.675 GB on M4**.
Verified free space is **37,297,278,976 bytes /37.297 GB on M1** at 17:55:42 UTC
and **80,231,067,648 bytes /80.231 GB on M4** at 17:54:09 UTC. Allocated blocks
are counted once per inode; concurrent system/research writes can change net
volume free space. No earlier pass is included in these totals.

## Retired archived originals

[All 516 exact paths, member locators, logical sizes and SHA256 pins](storage-vacuum-20261015-more-locators.csv)
are recorded separately. The private receipt holds original stats, source pins,
archive acceptance, durable journals and all retained/changed-path decisions.

| Machine /owner | Removed paths | Allocated bytes | Scope |
|---|---:|---:|---|
| M4 /merged PR149 | 309 (267 unique inodes) | 5,283,700,736 | Selected unchanged bulky board-pooling campaign payloads; source checkouts and small records stay |
| M4 /merged PR185 | 8 | 2,391,580,672 | Seeds 2026100602/603, each 1B/2B/5B/10B checkpoint; exact eight accepted milestone ZIPs |
| M1 /merged PR149 | 156 | 3,135,807,488 | Remaining unchanged old retrieval and engineering payloads, mapped to canonical/supplement members |
| M1 /merged PR164 | 42 | 1,336,123,392 | Archived native-gate checkpoint/reference payloads, including original pre-fix evidence |
| M1 /merged PR169 | 1 | 777,871,360 | Verified unused local review ZIP copy; restores as the whole accepted cloud object |

Current GitHub states for all four owners are merged. Every selected original has
an exact recorded source path, size and SHA256. M1 PR149/164 files retain their
recorded modification times and ctimes precede their accepted manifest sealing.
M4 PR149 paths/sizes/hashes match the previously fully verified canonical campaign
manifest; current mtime/ctime precede that immutable source snapshot. PR185 sizes
and hashes match the original verified milestone records; source modification
history precedes those milestone records and the existing ZIP member readback
receipts cover the exact checkpoint names. The PR169 local ZIP matches its
published-path verification receipt, size and unchanged pre-verification history.
No selected research original/member/archive is rehashed or downloaded.

All selected inode links are accounted for within the removal set. Fresh open-file
checks and exact path/size/mtime/ctime/device/inode/link guards precede removal;
per-group durable journals and per-path identity/decreasing-link checks surround
unlink. Changed, missing, uncertain and externally linked candidates are skipped.
Earlier originals-retained snapshots are superseded only for these exact paths.
Source checkouts, source code, tests, shared Git and worktrees are retained.

## Accepted restoration archives

Native FileProvider uploaded/not-uploading/no-conflicts/no-download/no-pause/
no-trash/no-exclusion/exact-size checks pass. Independent cloud ID/name/size/direct
parent checks agree both before and after removal. Confirmed Drive upload is
trusted with existing local readback/manifests; no remote payload audit is repeated.
The archive/member restoration pins are:

| Archive | Bytes | Whole SHA256 | Embedded manifest SHA256 |
|---|---:|---|---|
| [PR164-M1-native-gate-20261005.tar.gz](https://drive.google.com/file/d/13aotv71LxwK4QQ0x3XIpP4sz4m42K9B_/view?usp=drivesdk) | 1,336,346,201 | `dfabd1bfedfa0c276b803d4144005ebb831e3bd5f801f4c307e2879fdd27ff57` | `35cedb6744eee9e84f3bcacbddb78ecac866b99434f9631b5d059a23335a0974` |
| [O-2026100602-1000000000-milestone.zip](https://drive.google.com/file/d/16z0GMlOoxvK5-cFTNOtFebozcdK8OiRi/view?usp=drivesdk) | 215,176,734 | `7c51abc6963d5a415ffa3a8a130245ee63a81d1357db11e92e685280afb9613b` | `f4048a0514380046c3189d662fec81259fad8b6b26b8819b631bf20fa648f16f` |
| [PR149-M1-retrieval-history-20261005.tar.gz](https://drive.google.com/file/d/17woiEOQSDwHvyq6eNzLhm5U2cl0nCQan/view?usp=drivesdk) | 310,667,830 | `939bb2ee7d88832cd480251f2c922794f33ec631f7086c1c9979390b1371f619` | `155730ea8fa20ea29dc18058d6225c076e4cd5669e8bd4d439899a82372fb793` |
| [pr169-review-13616df-20261005.zip](https://drive.google.com/file/d/1H2Pmzq7H29686iJtmfoW1Lb89p7sRUfw/view?usp=drivesdk) | 777,870,083 | `7b3d0c6a839366a59561ae8eda1bdfbe400855c46c6df7b5b68ad26a1d4e7829` | `94c326847853c6a308fd3b24c63a5935f9b45aa2e4da5716a95f0927c289490f` |
| [O-2026100603-10000000000-milestone.zip](https://drive.google.com/file/d/1J-YuySbo-b8Zazo9NPzX8AlvxyKd7mb9/view?usp=drivesdk) | 387,875,447 | `e259d84cb5aaad7e2aec33d797fa2e5c6275c04ef7757ff21c19f5d5f34245da` | `a2f8352bb8657ddaddc6434d0d41847cb32877a85ec16f3817b221c4ec56cc70` |
| [O-2026100603-5000000000-milestone.zip](https://drive.google.com/file/d/1N-L3v5rvEqkEn2YvQZMww5vw37XHhOvf/view?usp=drivesdk) | 331,384,377 | `f360a8dcd5a1d39ca22dcab07de6a3529ee6eb68d73e5a561007372653c5a443` | `9f45dfaa77b6622d6e250a662ebb27eb11d60e462c849efdc7658e6f18ec3240` |
| [hu20-board-pooling-m4-complete-20261005.tar.gz](https://drive.google.com/file/d/1NNSkCO811USN6U9L2p6lBbti1-6Q9r2X/view?usp=drivesdk) | 16,802,192,861 | `20ad0f67df71464e7c06bdb1cf63451c243944c14e732b83650861f8f9bbcaed` | `d5608ba75a785dfc5633f96dc27246d1b0c7781e478e67bc31972163565fca65` |
| [O-2026100603-1000000000-milestone.zip](https://drive.google.com/file/d/1PgJDMu6jfTBzW5tkhXMqa5D1dBvi-J52/view?usp=drivesdk) | 213,677,835 | `2ec299e229e4fa7bc57f98ec20c76a0d06a9efcc426a3a465cc5a18be55d74eb` | `bdbe95192595cb4ac4d02aaedadad44bf49e773659c38cb3c927af32f17582b2` |
| [O-2026100602-5000000000-milestone.zip](https://drive.google.com/file/d/1YuB1dU0j68QVgYiA3oeqCajwd9vHDcgJ/view?usp=drivesdk) | 333,414,365 | `dfe132d93873203d17f980db4fcba6d7bfbcd70cec481293677171093356689a` | `1133cb9a0a06331d0fbdc1045fa8cdf12990b17eb1eec22b51d9859f8eb29d25` |
| [O-2026100602-10000000000-milestone.zip](https://drive.google.com/file/d/1_gyi_b9YdCkm5FOSj3kig2EBIkylLkSI/view?usp=drivesdk) | 387,641,010 | `122c4970149f1def9dc965d65ef4a8bacaba49101dfe9dddd60b663238348b06` | `7c7acbbb550aaaa3687e44979cc2b54c5a3963c29083210fdc5c85f79910af9c` |
| [O-2026100602-2000000000-milestone.zip](https://drive.google.com/file/d/1vI35SSrniXuT6VOgsBSfWlVgcN8F6-aP/view?usp=drivesdk) | 263,093,158 | `1be8bed66ec43f8c697eaecad0c68140a2447a41bed8285bb81864ead53ebeea` | `f9ec046ae1cc8a274fdb1f69f37078378a1211d8e99920df1e5efa7888c81ebf` |
| [O-2026100603-2000000000-milestone.zip](https://drive.google.com/file/d/1xXjQYNa1NTSu8VYN5ZXCAs4m-6N88JEC/view?usp=drivesdk) | 259,351,250 | `f7578b2ada73167c42917493b9bb088d2b0204ee93f9dcc6a32116259abf3442` | `61cc3cb7726a1d88d449225fc6e6128105316bab041cd4f527578aa995738753` |

Download the locator's archive into a fresh ignored unsynced directory and require
its whole SHA256 above before restoring. For a ZIP member:

```sh
shasum -a 256 ARCHIVE.zip
python3 -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extract(sys.argv[2],sys.argv[3])' ARCHIVE.zip EXACT_MEMBER FRESH_ROOT
shasum -a 256 FRESH_ROOT/EXACT_MEMBER
```

For a tar member, `tarfile.open(archive, 'r:gz').extractfile(member)` resolves tar
hard links; copy to a fresh destination and require the locator's member size and
SHA256. Preserve source inode groups from the private receipt when restoring
multiple aliases to avoid unnecessary duplicate storage. PR149's accepted M1
supplement also includes its existing hash-checking `restore_snapshot.py`;
`--snapshot EXTRACTED_SUPPLEMENT --canonical-dir EXTRACTED_CANONICAL --out FRESH_ROOT`
restores the complete historical mappings and hard links. For `@whole-archive`,
restore the accepted ZIP itself without extraction and require its whole hash.
Never overwrite active inputs or automatically relaunch old research.

## M1 inactive caches

Another **3,120,386,048 bytes** are disposable local caches:

- **2,071,158,784 bytes** from unused `~/.cache/codex-runtimes/codex-runtime-install-BBpV7e/`
  and `codex-runtime-install-9wKUB8/` install staging. Each contained a downloaded
  runtime archive and old partial payload. No open handles exist; no symlink from
  the current primary runtime targets either staging tree. The current
  `codex-primary-runtime` tree remains. A normal runtime installer recreates staging.
- **1,049,227,264 bytes /63,419 files** of ordinary Chrome network-resource cache
  older than 24 hours. Profile, cookies, sessions, settings, indexes, open files
  and recently modified files stay. Exact unchanged single-link files are removed;
  Chrome can refetch resources. The browser is not quit.

Each cache file's path and metadata plus regeneration method are in the private
receipt. Open/changed/linked files are retained. No current PR input or installed
runtime is removed. A separate supported `docker builder prune --filter until=24h`
prunes **4.26 MB of old unused Docker build cache**; images, containers, volumes and
local unpushed images remain. Docker's logical reclamation is excluded from the
host-byte totals because VM sparse-disk reclamation is separate.

## Protected work and durable receipt

Open PR292, PR297 and PR298 work/inputs stay, including M1's new HU100 release
candidate/reference/tables and M4's HU200 blend inputs. Stable selected retained
metadata checks pass for **138 M1 /85 M4 files**. Other agents' live generated
outputs are outside this cleanup. The failed PR278 seed1 ZIP, incomplete upload,
Lost and Found, personal data, CrossOver, Xcode, installed apps and shared Git
remain. No synced file is deleted or force-offloaded; no new PR comment, scientific
retry, release decision or unattended cleanup is triggered.

[Accepted private exact-path receipt](https://drive.google.com/file/d/1OyTa15dy7wUsLUyw2rByRMXfT1mcsU-g/view?usp=drivesdk),
ID `1OyTa15dy7wUsLUyw2rByRMXfT1mcsU-g`, **4,525,074 bytes**, SHA256
`d11197619fabaeb909971d2ce8fcd2ea1043717d6a2457bf6ab644afcc337274`, parent
`188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. Gzip expands to a JSON receipt; independent
cloud readback confirms exact name/size/direct parent. The full accepted receipt
is copied to both Macs under `~/Local/storage-vacuum-20261015-more/`.
