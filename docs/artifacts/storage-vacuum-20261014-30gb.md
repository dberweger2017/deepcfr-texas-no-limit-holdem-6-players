# Another 30 GB on M4, preserving live HU200 training — October 14

The owner requested 30 GB more on M4 without affecting live PR278.
This pass reclaimed **32.500 GB**, exceeding the target:
five unchanged archived HU100 originals, **16.544 GB**,
and three explicitly approved inactive policy caches,
**15.955 GB**. At **2026-10-14T16:20:35Z** the filesystem
reported **77,768,372,224 bytes /77.768 GB free**.
Free-space observations include concurrent training writes; per-path allocation
is the cleanup accounting and excludes other agents' reclamation.

## Acceptance, provenance and live-work protection

PR272 and its indexed PR260 parent are currently merged. Fresh native
uploaded/no-pending/no-conflicts/not-downloading/not-paused/excluded/trashed
acceptance and independent Drive ID/name/size/direct-parent confirmation accept
the four restoration archives. Existing model indices and complete local
member-readback receipts supply exact path/size/member/hash pins. Source paths,
byte counts, nanosecond mtimes and inodes match the prior recorded metadata.
Current ctime/mtime precede each original's specific verification/sealing
receipt: full/average archive verification for final models, byte-identical
parent-checkpoint verification and hashed parent-average evaluation input.
Thus the regenerated parent copies are proven against their indexed archived
PR260 equivalents without retrieving a checkpoint or repeating payload hashes.
The extra per-file source-history corroboration is in the private receipt.

Only paths below merged PR272's nonsynced results root were changed. Fresh open
file checks found no selected file/cache in use. The live PR278 worker uses
separate HU200 inputs under `~/Local/hu200-v055-confirmation-20261014/`; its work
root, models/tables, logs, outputs, locks and scientific processes were neither
changed nor signaled. Native trainer PID95151 was observed after removal; the
subsequent guard observation remained in frozen-train with no stop or breach.
Canonical K50 tables, PR265 HU200 5B/final averages, released v0.5.2 model/source,
Slumbot dependencies and shared Git remain. No cache construction, compression,
large payload read, archive download, repeated model/member hash audit or other
scientific computation was run on the live M4. No synced deletion, forced
offload, unpinning, M1 cleanup, application shutdown or schedule.

The owner specifically approved **“Delete those three rebuildable caches
(Recommended)”**, then gave standing permission for inactive rebuildable caches
and regenerable output with no data loss or active-PR impact. This is recorded
in AGENTS.md and the storage contract. The five scientific originals still use
the stricter merged/accepted-archive retention rule. Historical retained-path
statements are superseded only for these exact five originals/three caches.

## Removed originals and exact restoration

Download the linked archive into a new ignored nonsynced directory, verify its
whole SHA256, extract the exact member and verify the embedded manifest and
selected model sizes/SHA256s before reuse. Restoring evidence launches no science.

- **`~/Local/hu100-v2-frozen-5b-20261014/results/hu100-v2-frozen-5b/training/5000000000.average.gz`** — 3,196,291,551 logical /3,205,226,496 allocated bytes retired. [Archive `hu100-v2-frozen-5b-average-M4-20261014.zip`](https://drive.google.com/file/d/1p7LmF9xRGXroTLDqNiY4eRwPuMC2IRiX/view?usp=drivesdk), ID `1p7LmF9xRGXroTLDqNiY4eRwPuMC2IRiX`, 3,196,298,276 bytes, whole SHA256 `ea29e019781d390d6fb284c7e4675b73f197587ad28103656247f89304d7df2d`. Exact member `averages/5000000000.average.gz`, 3,196,291,551 bytes, SHA256 `c3a8edc4cc78098f07ac3e4ecbbb3dacae298a266b9fcd5fc788d0e2b6fcfc8e`; `ARCHIVE-MANIFEST.json` SHA256 `bc31a32e532d276fb95e501ff1e014c8fc419219b48c5a1c71c5005b923e6e67`.

- **`~/Local/hu100-v2-frozen-5b-20261014/results/hu100-v2-frozen-5b/training/5000000000.gz`** — 4,869,874,972 logical /4,882,898,944 allocated bytes retired. [Archive `hu100-v2-frozen-5b-M4-20261014.zip`](https://drive.google.com/file/d/1SKEvKCzuoWeMvBSYY8_iafuKQcb6pmT7/view?usp=drivesdk), ID `1SKEvKCzuoWeMvBSYY8_iafuKQcb6pmT7`, 11,540,258,992 bytes, whole SHA256 `1faf65de41c374921cd3aae0a7c06bc675ba28fa2cd4ca2d4a904ca6f1d24fed`. Exact member `research/training/5000000000.gz`, 4,869,874,972 bytes, SHA256 `401fe31263c763c5b805b0cff2f6af332d2dc7a3e9b7d8b99f1120b0216f58d6`; `ARCHIVE-MANIFEST.json` SHA256 `bae16b98314493a72c23e8a531952ee35fadc86a8adb96262262501084166e98`.

- **`~/Local/hu100-v2-frozen-5b-20261014/results/hu100-v2-frozen-5b/training/5000000000.current.gz`** — 2,589,617,212 logical /2,600,480,768 allocated bytes retired. [Archive `hu100-v2-frozen-5b-M4-20261014.zip`](https://drive.google.com/file/d/1SKEvKCzuoWeMvBSYY8_iafuKQcb6pmT7/view?usp=drivesdk), ID `1SKEvKCzuoWeMvBSYY8_iafuKQcb6pmT7`, 11,540,258,992 bytes, whole SHA256 `1faf65de41c374921cd3aae0a7c06bc675ba28fa2cd4ca2d4a904ca6f1d24fed`. Exact member `research/training/5000000000.current.gz`, 2,589,617,212 bytes, SHA256 `21c6a2ca5c755fb689119ac98f3a539ad05d83eb56cec702ed06ac0e72c5fed7`; `ARCHIVE-MANIFEST.json` SHA256 `bae16b98314493a72c23e8a531952ee35fadc86a8adb96262262501084166e98`.

- **`~/Local/hu100-v2-frozen-5b-20261014/results/hu100-v2-frozen-5b/parent/332894788.checkpoint.gz`** — 3,488,922,486 logical /3,490,177,024 allocated bytes retired. [Archive `hu100-k50-v2-evidence.zip`](https://drive.google.com/file/d/18H0Ve-4Uvzi4RLA0pFVZKGFhd-yQkkmv/view?usp=drivesdk), ID `18H0Ve-4Uvzi4RLA0pFVZKGFhd-yQkkmv`, 10,465,963,508 bytes, whole SHA256 `e21c2c525a37bd954a0a0a940289de287ee9f5da5e9e5e5e87a51baa0064b3ce`. Exact member `research/training/332894788.checkpoint.gz`, 3,488,922,486 bytes, SHA256 `202058df8aff994612b441d26ffc4e70475ee790cc06a799635e4389e219db2f`; `ARCHIVE-MANIFEST.json` SHA256 `c9d7e039c5febb4587917abb4aa7069bea6d2bb6a8c0f579191a3741db21ccb3`.

- **`~/Local/hu100-v2-frozen-5b-20261014/results/hu100-v2-frozen-5b/parent/332894788.average.gz`** — 2,363,005,114 logical /2,365,706,240 allocated bytes retired. [Archive `hu100-k50-v2-terminal-average.zip`](https://drive.google.com/file/d/1QNswSktLznNQq7NwBS9jyACeJFXmkJ6q/view?usp=drivesdk), ID `1QNswSktLznNQq7NwBS9jyACeJFXmkJ6q`, 2,363,016,534 bytes, whole SHA256 `f80a243a58542d92653323701956b8f919b35cd238d7a165cfdf4efe06df1ca2`. Exact member `averages/332894788.average.gz`, 2,363,005,114 bytes, SHA256 `96a179ceacc16d5050a3fa005800ad0678e6368a576da5e98b8a4542c03792fc`; `ARCHIVE-MANIFEST.json` SHA256 `357d31fced09b5b7deca257746a65e234dd498949119eb45f9f99cec5c3ddde1`.

```sh
shasum -a 256 ARCHIVE.zip
python -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extract(sys.argv[2],sys.argv[3])' ARCHIVE.zip EXACT_MEMBER NEW_RESTORE_ROOT
shasum -a 256 NEW_RESTORE_ROOT/EXACT_MEMBER
```

## Deleted policy caches and exact rebuild inputs

The three directories held derived compact-average arrays excluded from the
science archive. The full per-file sizes/SHA256s and manifests are retained in
the private receipt; no new cache payload audit was performed. Every selected
directory/file remained identical to the prepared metadata inventory, had no
symlink/external hardlink, and was unopened immediately before deletion.
Use scientific source `fd066d26e61cab1ecc2d7f9e131ce2beac296998`, restore the
pinned source average below and build a fresh nonsynced cache; do not overwrite
active inputs or rebuild during an unrelated running experiment.

- **`~/Local/hu100-v2-frozen-5b-20261014/results/hu100-v2-frozen-5b/policy-cache/0ed7f9512996bb3e9535a7258a5d0e4bc8c00f22734d511c642c0e5b95cbd3c0-hu100-native-reopening-ordered-history-card-v1-compact-average-cache-v1`** — 4,172,546,048 allocated bytes; source model SHA256 `0ed7f9512996bb3e9535a7258a5d0e4bc8c00f22734d511c642c0e5b95cbd3c0`, schema `hu100-native-reopening-ordered-history-card-v1`, 70,466,972 entries. Use retained local model `~/Local/hu100-k50-vs-v052-20261013/results/hu100-k50-vs-v052/release-v0.5.2/O4B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz`, 2,121,415,396 bytes, or the pinned [v0.5.2 release bundle](../releases/v0.5.2/PUBLICATION.md); verify source model SHA256 above. Tables: None for original released card-v1 schema.

- **`~/Local/hu100-v2-frozen-5b-20261014/results/hu100-v2-frozen-5b/policy-cache/96a179ceacc16d5050a3fa005800ad0678e6368a576da5e98b8a4542c03792fc-hu100-native-reopening-ordered-history-equity-k50-menu-v2-compact-average-cache-v1`** — 5,891,317,760 allocated bytes; source model SHA256 `96a179ceacc16d5050a3fa005800ad0678e6368a576da5e98b8a4542c03792fc`, schema `hu100-native-reopening-ordered-history-equity-k50-menu-v2`, 97,290,578 entries. Restore model from [ZIP `hu100-k50-v2-terminal-average.zip`](https://drive.google.com/file/d/1QNswSktLznNQq7NwBS9jyACeJFXmkJ6q/view?usp=drivesdk), whole SHA256 `f80a243a58542d92653323701956b8f919b35cd238d7a165cfdf4efe06df1ca2`, manifest SHA256 `357d31fced09b5b7deca257746a65e234dd498949119eb45f9f99cec5c3ddde1`, member `averages/332894788.average.gz` (2,363,005,114 bytes /SHA256 `96a179ceacc16d5050a3fa005800ad0678e6368a576da5e98b8a4542c03792fc`). Tables: PR163 retained canonical ~/Local/hu100-k50-v2/results/hu100-k50-v2/tables; use indexed SHA256s.

- **`~/Local/hu100-v2-frozen-5b-20261014/results/hu100-v2-frozen-5b/policy-cache/c3a8edc4cc78098f07ac3e4ecbbb3dacae298a266b9fcd5fc788d0e2b6fcfc8e-hu100-native-reopening-ordered-history-equity-k50-menu-v2-compact-average-cache-v1`** — 5,891,317,760 allocated bytes; source model SHA256 `c3a8edc4cc78098f07ac3e4ecbbb3dacae298a266b9fcd5fc788d0e2b6fcfc8e`, schema `hu100-native-reopening-ordered-history-equity-k50-menu-v2`, 97,290,578 entries. Restore model from [ZIP `hu100-v2-frozen-5b-average-M4-20261014.zip`](https://drive.google.com/file/d/1p7LmF9xRGXroTLDqNiY4eRwPuMC2IRiX/view?usp=drivesdk), whole SHA256 `ea29e019781d390d6fb284c7e4675b73f197587ad28103656247f89304d7df2d`, manifest SHA256 `bc31a32e532d276fb95e501ff1e014c8fc419219b48c5a1c71c5005b923e6e67`, member `averages/5000000000.average.gz` (3,196,291,551 bytes /SHA256 `c3a8edc4cc78098f07ac3e4ecbbb3dacae298a266b9fcd5fc788d0e2b6fcfc8e`). Tables: PR163 retained canonical ~/Local/hu100-k50-v2/results/hu100-k50-v2/tables; use indexed SHA256s.

```python
from pathlib import Path
from src.blueprint.average import AveragePolicy

# Use the exact model SHA/schema listed above and retained PR163 K50 tables
# for v2. The original released card-v1 schema uses card_buckets=None.
AveragePolicy(MODEL_PATH, EXPECTED_SHA256, expected_schema=SCHEMA,
              card_buckets=K50_TABLES_OR_NONE, sorted_rows=True,
              cache=Path(FRESH_NONSYNCED_CACHE))
```

[Full accepted private receipt](https://drive.google.com/file/d/1MHJdvblmb-eNeqKm2fiVWhtUaPCi9P2J/view?usp=drivesdk), ID `1MHJdvblmb-eNeqKm2fiVWhtUaPCi9P2J`, **157,663 bytes**, SHA256 `eb0d3fc94ca336847ea50da106ab1e9a7871a487e93b95888fcb75df91b4457c`. Independent metadata confirms exact receipt name/size and Research-Cloud parent `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. It preserves current native/cloud acceptance, source-history proof, exact removal journals, cache manifests/input pins/rebuild API and live-worker verification.
