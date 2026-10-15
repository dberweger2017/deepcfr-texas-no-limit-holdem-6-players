# M4 and M1 storage cleanup — October 15 evening

Owner requested more M4 space, then about 25 GB on M1. This pass reclaims
**17,074,864,128 bytes (17.075 GB) on M4** and **25,208,426,496 bytes
(25.208 GB) on M1**. Verified free space is **73,784,881,152 bytes /73.785 GB**
on M4 at 16:47:09 UTC and **38,577,963,008 bytes /38.578 GB** on M1 at
16:56:29 UTC. Subsequent writes may change free space. Hard links are counted once.

## M4 removals and restoration

PR296 is currently merged (16:03:12 UTC, head
`eeb47f476d014b6a0c185d5517a426392c189f1e`); PR286 is merged too.

| Scope | Reclaimed bytes | Restoration or regeneration |
|---|---:|---|
| PR296 restored `~/Local/hu100-v053-complete-20261015/inputs/2026101502.average.gz` | 3,187,449,856 | [PR286 seed1502 average ZIP](https://drive.google.com/file/d/1thfF0zPFYQT5QxAa0pav_RWZpYE66p0u/view?usp=drivesdk), member `averages/2026101502/5000000000.average.gz` |
| Two PR296 policy-cache directories, source hashes `a476afa7…` and `0ed7f951…` | 10,064,023,552 | Rebuild with pinned seed1502/released v0.5.2 averages and retained canonical K50 tables; source `ed090caecde29569a5c7cd44e49f6dc434145e9c`, `src.blueprint.average.AveragePolicy` |
| PR286 seed1502-average and PR296 completion-evidence uploaded ZIP caches | 3,823,390,720 | Normal native `FileManager.evictUbiquitousItem`; same-size dataless placeholders and Drive IDs remain |

The removed restored model is **3,187,447,872 logical bytes**, SHA256
`a476afa7334968087e5dfdb268d3897f9c2afba13905694c054bca2db4a59c19`.
Its exact path/size/hash matches PR296's published verified
[input restoration](../reports/hu100-v053-completion-artifacts/input-restoration.json),
and its mtime/ctime precede that unchanged restoration receipt. No payload hash is
repeated. Average ZIP: **3,187,451,673 bytes**, whole SHA256
`275f7ed7ca143f05ca64a27d93163a08e829c2e7ca2c9d71b1a2a9791fc6f0df`, manifest
`e3f56f9b972009dd96c83e5f4d66842514c2bbd7bd421e5b611c080baffd027b`.

The [PR296 evidence ZIP](https://drive.google.com/file/d/12l23aygPaBinIZNDtp_K0shR_wx6zqf3/view?usp=drivesdk)
is **635,934,911 bytes**, whole SHA256
`0603c5b0975aa3ca8e70d1665fdf59a56b7a9521ba1e453d73b8594fb5e79417`,
manifest `83c35dbffaa47e0db72dcce2467fe3d334217d84683da05ea88370989eae0dfa`.
Native uploaded/no-pending/no-conflicts/no-download/no-pause/no-trash/no-exclusion,
unpinned status and independent cloud ID/name/size/direct-parent checks pass for
both. Both have zero resident blocks after ordinary eviction. No synced file is
raw-deleted, forced offline or unpinned. Original/final evidence stays locally.

Restore the model into a fresh ignored unsynced directory: download its ZIP by the
link above, require the whole SHA256, extract the exact member with
`python3 -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extract(sys.argv[2],sys.argv[3])' ARCHIVE.zip averages/2026101502/5000000000.average.gz FRESH_ROOT`,
then require the restored member SHA256. Download the evidence ZIP normally to
restore its residency. Do not overwrite active inputs or automatically rerun work.
Rebuild either cache with `AveragePolicy(path, expected_sha256,
expected_schema=abstraction, card_buckets=canonical_tables, sorted_rows=True,
cache=NEW_CACHE)` using the exact model/schema/table pins in the private receipt.

## M1 archived retrieval copies

Merged PR149's three old M1 retrieval copies `hu20-board-pooling-m4-retrieval-03-20261004`,
`-04-20261004` and `-05-20261004` contain **1,353 removed paths /391 unique inodes**,
reclaiming **8,918,282,240 bytes**. Only the selected unchanged bulky files are
removed; source checkouts, small metadata, partial/failure records and all other
files remain. Their exact original paths, byte counts, SHA256s and modification
times match the existing accepted supplement manifest; ctime precedes its sealing.
All links to each selected inode are within the reviewed removal set. No open
handles exist. Identity/size/mtime/ctime/link counts are rechecked before each
group; per-path identity and decreasing link count are checked before unlink.

Every selected file maps to a `campaign/...` member in the
[canonical M4 campaign tar](https://drive.google.com/file/d/1NNSkCO811USN6U9L2p6lBbti1-6Q9r2X/view),
**16,802,192,861 bytes**, whole SHA256
`20ad0f67df71464e7c06bdb1cf63451c243944c14e732b83650861f8f9bbcaed`,
manifest `d5608ba75a785dfc5633f96dc27246d1b0c7781e478e67bc31972163565fca65`.
The [M1 history supplement](https://drive.google.com/file/d/17woiEOQSDwHvyq6eNzLhm5U2cl0nCQan/view)
is **310,667,830 bytes**, whole SHA256
`939bb2ee7d88832cd480251f2c922794f33ec631f7086c1c9979390b1371f619`,
manifest `155730ea8fa20ea29dc18058d6225c076e4cd5669e8bd4d439899a82372fb793`.
Both pass current native upload status and independent cloud exact name/size/parent
checks, before and after removal. Only the existing manifest and restore helper
are read; neither archive payload is downloaded or rehashed for cleanup.

For restoration, download and require the archive hashes above. Extract the
canonical tar and supplement into fresh separate directories on a host with enough
space. The supplement's existing `restore_snapshot.py` accepts
`--snapshot EXTRACTED_SUPPLEMENT --canonical-dir EXTRACTED_CANONICAL --out FRESH_ROOT`;
it checks each restored member size/SHA256 and preserves original hard links.
For only a selected file, the private receipt's `source_manifest_entry.canonical.member`
is its exact tar member; `tarfile.open(archive, 'r:gz').extractfile(member)` resolves
tar hard links. Verify the recorded member SHA256 before use. Full historical
source mappings remain in the accepted supplement manifest; no automatic relaunch.

## M1 rebuildable caches

Another **16,290,144,256 bytes /42,013 regular cache or installer files** are
removed under the owner's standing inactive-cache permission. The private receipt
lists every removed path, metadata, generating inputs and regeneration method:

- Two unused `~/Local/slumbot-k50-20261013/cache/` policy caches, sources
  `b7c15eb642f2edec82df04983b28a8f80b8c3db310f3d6ea53e21aab6d120402`
  and `60bd37c979e018f11f136370f468cc41b8e1435550531707e89d45f28709ec7c`.
  Original models, K50 tables, server copies, all journals and account data stay.
  Rebuild with the same `AveragePolicy` API and retained pinned inputs.
- Old downloaded Flutter installer and two unused SDK `bin/cache/` trees under
  MediaServer/flutter-tvos. SDK source, version/engine pins and project work stay;
  `bin/flutter precache` bootstraps the disposable runtime/engine downloads again.
  The retained release index pins MediaServer Flutter 3.44.1 source
  `924134a44c189315be2148659913dda1671cbe99`, engine
  `c416acfeb8126e097f758c664aaa3da929e27da0`, installer SHA256
  `586220859c83f3725ac07e73824e7ccc94df208a0182570991fc4d4da5b5ae17`.
- npm, pip, Homebrew, Gradle, uv, pre-commit and Wootility download/build caches;
  Python bytecode; old unopened Chrome JS/Wasm compiled-code cache blobs; downloaded
  Claude installer. Sources, installed applications/environments, lock files,
  browser profile/sessions, current code-cache entries and open/linked files stay.
  Package tools refill caches; Python/Chrome regenerate compiled code normally.

All selected cache files are regular, singly linked, unused and unchanged at
removal. Active tool runtimes and uncertain files are retained. No app is quit.

## Protected work and receipt

Open PR292, PR297 and new PR298 inputs/source remain, including the M1 release
candidate, released v0.5.2 and shared K50 tables. M4 HU200 blend inputs remain.
The failed PR278 seed1 ZIP/upload/Lost and Found remains; no upload is assumed.
Shared Git, worktrees, Xcode, CrossOver and installed apps stay. Metadata checks
pass for **4,830 M4** and **7,856 M1** retained files. No scientific outcome,
retry/release decision, app publication or unattended schedule changes.

[Accepted exact-path private receipt](https://drive.google.com/file/d/1dS_sRfn8TrUQuR1Jf1zLmRwtKDtP4ab5/view?usp=drivesdk),
ID `1dS_sRfn8TrUQuR1Jf1zLmRwtKDtP4ab5`, **1,672,016 bytes**,
SHA256 `b582c20ca41bffd4a0197b771f1910a2d5d45a471c3b4146c31eecdc03dda325`,
parent `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. Gzip expands to a JSON receipt with
all source/member pins, durable journals, per-file cache plans and retained-file
checks; independent cloud metadata confirms its name/size/direct parent.
Local task records are `~/Local/storage-vacuum-20261015-evening/` on both Macs
(the complete accepted receipt is copied to both). No repeat payload audit.
