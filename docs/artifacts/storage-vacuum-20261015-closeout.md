# M4 space for PR286 evidence closeout — October 15, 2026

Owner-requested cleanup reclaims **18,234,642,432 bytes /18.235 GB**:
17.343 GB of six unchanged, fully archived inactive HU200 model copies and
0.891 GB of six unused Rust debug caches plus the pip download cache.
At 2026-10-15T12:33:12.838848+00:00, M4 has **40,041,111,552 bytes free
(40.041 GB /37.291 GiB)**. Archive writes can change this snapshot.
No source checkout, installed package/environment or open PR input is removed.

## Exact retired models and restoration

Paths below are under `~/Local/`. Each owning PR and each backup PR is currently
merged. Current native uploaded1/uploading0/no-conflicts/no-download/no-pause/no-
trash/no-exclusion/exact-size checks and independent cloud ID/name/size/direct-
parent checks pass for all five restoration archives, before and after cleanup.
Existing full local archive readback and indexed member/source pins are trusted;
no remote payload download or repeated archive/member hash audit is performed.

| Removed Local path | Owner | Restore archive ID | Exact member | Bytes /SHA256 |
|---|---|---|---|---|
| `hu200-v055-confirmation-20261014/results/hu200-v055-confirmation/seed-2026101421/frozen/20000000000.average.gz` | #278 | [`1OAKKAxH91_flAKK9eWa4peCy_jUImhDl`](https://drive.google.com/file/d/1OAKKAxH91_flAKK9eWa4peCy_jUImhDl/view?usp=drivesdk) | `averages/seed-2026101421/final.average.gz` | 3,162,671,080 /`c08d86a9615022df7015d34e5dc3de2705bbb8346ed6976bd72284e71d91f444` |
| `hu200-v055-confirmation-20261014/results/hu200-v055-confirmation/seed-2026101422/frozen/20000000000.average.gz` | #278 | [`1nOhwFtDzJakpkB7MnHdk5JgY75CRnQjT`](https://drive.google.com/file/d/1nOhwFtDzJakpkB7MnHdk5JgY75CRnQjT/view?usp=drivesdk) | `averages/seed-2026101422/final.average.gz` | 3,162,064,602 /`b191c5ed8b943d138e6076fb4a1b0117c98964d92dd2d6253e157f696f691f6f` |
| `hu200-k50-frozen-20261013/results/hu200-k50-frozen/frozen/20000000000.average.gz` | #265 | [`1iVND4K_trvJ8S6gnDnyDc4tj5fNR2mDh`](https://drive.google.com/file/d/1iVND4K_trvJ8S6gnDnyDc4tj5fNR2mDh/view?usp=drivesdk) | `averages/20000000000.average.gz` | 3,164,563,975 /`9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6` |
| `hu200-k50-frozen-20261013/results/hu200-k50-frozen/frozen/5000000000.average.gz` | #265 | [`1iVND4K_trvJ8S6gnDnyDc4tj5fNR2mDh`](https://drive.google.com/file/d/1iVND4K_trvJ8S6gnDnyDc4tj5fNR2mDh/view?usp=drivesdk) | `averages/5000000000.average.gz` | 2,756,501,598 /`43aa3b3cd3c7365c5be1b177dcc8a7c04e01c8470403c38327aa845922726512` |
| `hu200-v2-frozen-5b-20261014/results/hu200-v2-frozen-5b/training/5000000000.average.gz` | #270 | [`1XwE9T6K84_tZPv00soy_d8VtQAHSUEGD`](https://drive.google.com/file/d/1XwE9T6K84_tZPv00soy_d8VtQAHSUEGD/view?usp=drivesdk) | `averages/5000000000.average.gz` | 2,876,030,727 /`ba4811a4e2bdb82fca382758a3713bca6701bfe2c13c661131abb3105e28b6b3` |
| `hu200-frozen-evaluation-20261014/models/752m.average.gz` | #268 | [`1DMz8kmY4DIQfO-NSIdEiOu9k-tl1hzrB`](https://drive.google.com/file/d/1DMz8kmY4DIQfO-NSIdEiOu9k-tl1hzrB/view?usp=drivesdk) | `research/training/1000000000.average.gz` | 2,221,370,011 /`60bd37c979e018f11f136370f468cc41b8e1435550531707e89d45f28709ec7c` |

Exact sizes/recorded paths and modification history were checked. PR278 and
PR265 copies match their recorded source mtime; ctime/mtime precede their relevant
verification receipts. PR270's source ctime/mtime precede the complete source-
hash/pack/readback interval. PR268's restored copy predates its specific verified
input receipt. All six have no open handles, single links and identical identity/
size/mtime/ctime at the final pre-unlink check. Changed/uncertain files stay.
These six paths supersede earlier local-retention snapshots; their cloud
restoration sources and separate prepared server models/caches remain.

| Archive | Logical bytes | Whole SHA256 | Embedded ARCHIVE-MANIFEST.json SHA256 |
|---|---:|---|---|
| [hu200-v055-seed-2026101421-final-average-20261014.zip](https://drive.google.com/file/d/1OAKKAxH91_flAKK9eWa4peCy_jUImhDl/view?usp=drivesdk) | 3,162,672,661 | `27790a65a55f4e95e7818511118ac62b09387c41f432727a102522f53611f85f` | `678205c4f4894d643f75e4b08d064649e2e983bc0f5e7901505d640997de8f25` |
| [hu200-v055-seed-2026101422-final-average-20261014.zip](https://drive.google.com/file/d/1nOhwFtDzJakpkB7MnHdk5JgY75CRnQjT/view?usp=drivesdk) | 3,162,066,289 | `76a82e8c1f757f295a7f2498b429047eff23befb29fe3f7d48580f4cdb84332e` | `14ec89c65d748a7fd2204546793108a152515d24933aded0753404ec29539089` |
| [hu200-k50-frozen-averages-M4-20261013.zip](https://drive.google.com/file/d/1iVND4K_trvJ8S6gnDnyDc4tj5fNR2mDh/view?usp=drivesdk) | 5,921,067,848 | `fe3465bea7591e82a762943e197d88d89fa36c8002b4143e6ebf70a887b2d943` | `d349217a651c7d28ac536124be40609778c8b55827d4b000e6840e49544eddc5` |
| [hu200-v2-frozen-5b-average-M4-20261014.zip](https://drive.google.com/file/d/1XwE9T6K84_tZPv00soy_d8VtQAHSUEGD/view?usp=drivesdk) | 2,876,032,489 | `bbd1abc9abdf550504ca2c3b5a6ffc72b162c8b8fea8aadc04364a9718ed623a` | `fb32c4da340cdcf91821e437b4ed5851c718b98915e82c4f30f540504ad965e5` |
| [hu200-k50-first-averages-M4-20261013.zip](https://drive.google.com/file/d/1DMz8kmY4DIQfO-NSIdEiOu9k-tl1hzrB/view?usp=drivesdk) | 2,221,371,477 | `4e6fc9f34b93f8a4e08628e31ba72ebe0b13f08605c02adc62d6354127c33165` | `78b893820f2fcfb18af69e3da99ee68798d373ac90a543dc8c9205edb27e10db` |

Restore by downloading the appropriate archive to a fresh ignored unsynced path,
checking its whole SHA256, then extracting the named member into a fresh restore
root. Verify the manifest SHA256 and restored member size/SHA256 before use:

```sh
shasum -a 256 ARCHIVE.zip
python3 -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extract(sys.argv[2],sys.argv[3])' ARCHIVE.zip EXACT_MEMBER FRESH_RESTORE_ROOT
shasum -a 256 FRESH_RESTORE_ROOT/EXACT_MEMBER
```

Never overwrite active inputs. The original model indices in the PR278, PR265,
PR270 and PR250 reports retain lineage/retrieval commands and input pins.

## Rebuildable cache removal

All source workspaces and Cargo pins remain unchanged. No selected cache has an
open handle or active PR286/PR292 dependency; exact prior cache inventories
match before deletion. External hardlinks or changed caches are retained;
internal hardlinks are counted once. The private receipt records every file.

| Removed Local debug path | Retained source commit | Regeneration |
|---|---|---|
| `pb-devchk/target/debug` | `af76476099637d582cf1f864a2e695699c1f398c` | `cargo build --manifest-path /Users/dberweger/Local/pb-devchk/Cargo.toml --locked` |
| `pb-l011-m4/target/debug` | `5c4aaca4294ef9de2f024cbb5f3af2e3d7d874a7` | `cargo build --manifest-path /Users/dberweger/Local/pb-l011-m4/Cargo.toml --locked` |
| `pb-l011-m4/spikes/l01-throughput/native/target/debug` | `5c4aaca4294ef9de2f024cbb5f3af2e3d7d874a7` | `cargo build --manifest-path /Users/dberweger/Local/pb-l011-m4/spikes/l01-throughput/native/Cargo.toml --locked` |
| `pb-r20-m4/target/debug` | `7341fa5b57bf83f16e0dd1b180150ca7d11e72e4` | `cargo build --manifest-path /Users/dberweger/Local/pb-r20-m4/Cargo.toml --locked` |
| `pb-ledger-rev6-m4/target/debug` | `32c26881b8fcc183531cbdb4c8f1d16284bb4842` | `cargo build --manifest-path /Users/dberweger/Local/pb-ledger-rev6-m4/Cargo.toml --locked` |
| `pb-r21-m4/target/debug` | `e6853d7ef51785f162c84c897393e3be97a00b9d` | `cargo build --manifest-path /Users/dberweger/Local/pb-r21-m4/Cargo.toml --locked` |

The disposable `~/Library/Caches/pip` downloads are removed after no-open-handle/
exact-inventory checks. Installed packages, environments and dependency/source
files remain; normal package installation refills this cache. The Homebrew
cache candidate is retained because its structure failed the strict eligibility
check. No personal documents/apps or CrossOver data are removed.

## Protected work, upload gap and owner-requested comment

PR286's canonical plan and evaluation inputs explicitly name its two new HU100
models, original-seed HU100 v2 input, v0.5.2 model under
`hu100-k50-vs-v052-20261013/`, and shared K50 tables. All remain, together with its
full worktree/evidence/policy caches. All 12 protected input/retained seed1
checkpoint paths match size/mtime/inode after retirement. PR292 uses bb128;
its root and M1 coordination are untouched. No process is signaled or restarted.

The PR278 seed1 full retry is absent from its expected synced path; a Drive name
search returns no match. It is not accepted or used as a full restoration source.
Failed canonical full ZIP, Lost and Found, and every unaccepted seed1 checkpoint/
current export remain. Only its independently accepted separate average ZIP
supports retirement of that final average. No upload repair/retry, synced
folder deletion, forced offload or unattended maintenance is performed.

The owner explicitly requested a [Vacuum Cleaner Bot comment on PR286](https://github.com/dberweger2017/CFR-NLHE-6max/pull/286#issuecomment-6192153333).
The posted/read-back comment gives the verified headroom and protected paths;
it is for evidence closeout and does not change the terminal scientific stop or
retry decision. This cleanup reports no poker outcome or qualification claim.

[Accepted private per-path receipt](https://drive.google.com/file/d/19BFM2fOtr4K5jQ6axgTTyxUjPzM8_D2M/view?usp=drivesdk),
ID `19BFM2fOtr4K5jQ6axgTTyxUjPzM8_D2M`, **6,617,574 bytes**,
SHA256 `e22e8d7d9d6e0e9592e2121e66d215c8f599300aff3a438a6287a365daa66831`; parent `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`.
Independent Drive metadata readback confirms exact name/size/direct parent.
Local records are under `~/Local/storage-vacuum-20261015-closeout/` on both Macs.
