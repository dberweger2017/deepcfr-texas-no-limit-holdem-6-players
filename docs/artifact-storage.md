# Git, Google Drive and local research storage

This is the common storage contract for new work. The maintainer's standing
instructions control artifact preservation on the project's machines;
[RESULTS_INDEX.md](../RESULTS_INDEX.md) remains
the canonical archive/retrieval index. A historical report's “retained” or
“upload pending” statement is a dated snapshot, not proof of current storage.

## What belongs where

**Git:** source, dependency pins, small configurations and deterministic test
fixtures, protocols, readable reports, modest plots, compact result summaries,
and archive/retrieval receipts. Keep enough context to understand the result
without downloading the full experiment. A filename such as `manifest.json`
does not make megabytes of raw data a compact receipt.

**Research-Cloud:** training checkpoints, inference exports, replay buffers,
bucket tables, raw hand/decision traces, bulky solver inputs/outputs, complete
logs, interrupted attempts, and generated research archives. The canonical Drive
root is [deepcfr-research-results](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s).
Use the existing per-PR folder name and record its actual Drive folder ID/URL;
do not create a second folder merely because its spelling differs from an example.

| Host | Archive location and upload route |
|---|---|
| M1/M4 | `~/Local/Research-Cloud/PR-<n>-<slug>/`, uploaded by native Drive desktop; both Macs see the same cloud folder. |
| cygnus-x1 (`ssh bb128`) | `/mnt/data2tb/research-cloud/PR-<n>-<slug>/`, copied directly to `gdrive:PR-<n>-<slug>/` with rclone. No Mac staging is required. |

**Ignored local working directories:** retrieve into `models/`, `results/` or
`planning/` outside synced folders. Keep active inputs where their campaign
expects them. Verify SHA256 before use. Released inference assets continue to
use their pinned GitHub release downloads; being named a release does not grant
permission to commit a model to Git.

## bb128 archive workflow

**Active October 16, 2026:** `rclone lsd gdrive:` successfully listed the existing
PR folders. [Activation observations](artifacts/bb128-storage-activation-20261016.md)
record rclone 1.60.1-DEV, matching existing Drive folder IDs, and the mounted
ext4 disk/fstab entry. A read-only check of the existing 50 MB smoke test also
found zero differences/one match; no new upload was created. The owner's `gdrive:` remote is already rooted at
`188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`; do not append another
`deepcfr-research-results` directory. Its per-PR names are exactly those under
the Macs' Research-Cloud. Access activation is separate from each archive's
upload acceptance.

The 1.8 TB Seagate HDD at `/mnt/data2tb` is persistent through fstab with
`nofail`. Its approximately 150–200 MB/s throughput suits archives and
checkpoints **at rest**. Active training, audits, inference/policy caches and
restored working inputs stay on NVMe. Check the mount before writing: because
`nofail` permits boot without it, an ordinary directory at that path is not
proof the disk is mounted. A missing disk, sign-in or network problem leaves
the closeout pending; retain originals and report the blocker in the PR.

1. Check the owning PR's current status and archive/dependency authorization.
   Build the ZIP into the existing HDD per-PR folder, with embedded
   `ARCHIVE-MANIFEST.json`. Preserve failures, partial attempts and source/input
   provenance. Record local archive bytes/SHA256 and manifest SHA256; read back
   every member against its size/SHA256 as before. Exclude credentials from
   the evidence set before building it.
2. Upload one way using immutable copies. Run **one large upload at a time**
   across agents, with one file transfer at a time, on the shared approximately
   3–4 MB/s home uplink. Do not let uploads compete with Slumbot play. Coordinate
   with existing transfers before starting; activation grants no new compute.

   ```sh
   mountpoint -q /mnt/data2tb || exit 1
   findmnt -T /mnt/data2tb -o TARGET,SOURCE,FSTYPE,OPTIONS
   rclone lsd gdrive:
   # Replace the placeholder with the exact existing per-PR folder name.
   research_pr_folder='PR-<n>-<slug>'
   rclone copy --immutable --transfers 1 \
     "/mnt/data2tb/research-cloud/$research_pr_folder/" \
     "gdrive:$research_pr_folder/"
   ```

   Never use `rclone sync`, `move`, `delete`, `purge`, `dedupe`, any
   `--delete-*` flag, or any operation that overwrites/removes existing Drive
   files. An immutable-copy conflict is a blocker: preserve both records and
   report it; use a distinct archive name for new evidence after review.
3. Accept the upload only when the following read-only check exits successfully
   with **zero differences** and compares Drive's MD5 with the local files.
   Require every local archive to be checked. Preserve stdout/stderr and exit
   status outside the folder being checked, avoiding self-referential logs.
   Do not substitute a listing,
   `--size-only`, or an unchecked/missing hash for this acceptance.

   ```sh
   rclone check --one-way \
     "/mnt/data2tb/research-cloud/$research_pr_folder/" \
     "gdrive:$research_pr_folder/"
   rclone lsjson "gdrive:$research_pr_folder/" --hash
   ```

   Record the exact file `ID`, name, size, MD5 and remote path from `lsjson`,
   the Drive folder ID/URL, local archive bytes/SHA256, manifest SHA256,
   member-readback result and check output in the receipt and RESULTS_INDEX
   (or link the compact receipt there). MD5 upload comparison complements the
   SHA256 archive/member provenance. It does not assert a remote SHA256 audit
   or archive-byte download.
4. Keep the HDD copy after upload as a local restore cache. Restore to a fresh
   ignored NVMe working directory, then verify the indexed SHA256s before use:

   ```sh
   rclone copy "gdrive:$research_pr_folder/ARCHIVE.zip" results/retrieved/NEW_ROOT/
   # While retained, the indexed HDD archive is another retrieval source.
   sha256sum results/retrieved/NEW_ROOT/ARCHIVE.zip
   ```

   Extract only after the whole hash matches; verify the required manifest and
   members too. Remove a local HDD cache copy only after its matching Drive
   copy passed `rclone check`, dependency review finds no active user, and a
   cleanup receipt records the exact path and restoration source. NVMe research
   originals still follow the merged-owner/upload-acceptance/dependency rules
   below. The HDD is a local cache, not a Drive-synced directory.

`~/.config/rclone/rclone.conf` contains the owner's Google OAuth token. Never
print, copy, archive, commit or transmit it; exclude it from every evidence set.
Agents never create or change Google credentials or sharing. The Macs retain
their existing synced-folder rules, including no deletion or forced offloading.

## Required closeout record

Each campaign's entry in RESULTS_INDEX must name the owning PR and include:

- Source revision, run identity, and whether the record is final or partial.
- Actual Drive archive ID/URL and folder ID/URL, archive name, byte count and SHA256.
- Embedded member manifest path/hash and member size/SHA256 readback result.
- For each required model/input: archive member path, byte count, SHA256 and a
  usable retrieval command, or an exact link to its already indexed record.
- Separate upload evidence for the route used: on Macs, current native
  uploaded/no-pending/no-conflicts status plus independent cloud
  ID/name/size/parent confirmation; on bb128, successful zero-difference
  `rclone check --one-way` with MD5 comparison plus `rclone lsjson` file IDs,
  names/sizes/hashes and folder mapping. State explicitly whether remote
  archive bytes were downloaded and verified; bb128 needs no Mac-native receipt.
- Failures, partial attempts and restore dependencies. Never present a successful
  final bundle as if earlier failures did not occur.

A cloud listing proves metadata, not member contents. A local ZIP readback proves
local archive integrity, not upload completion. Both claims must stay distinct.
For manual retrieval, download the indexed archive to a fresh ignored path,
verify its whole SHA256, extract the named members and verify their SHA256s.
Never treat a link alone as an accepted archive or overwrite an active input.

## Repository guard

`python -m scripts.check_repository_artifacts` checks the staged index;
`--revision HEAD` checks a commit. It rejects new model/archive/database formats,
raw JSONL traces, tracked files under local research roots, and any blob larger
than **1 MiB**, including large plain JSON. Case variants and interrupted suffixes
are checked. Small JSON summaries, CSV tables, report images and source remain
eligible. This is a maintenance rule, not a content classifier: authors still
must follow the storage contract for data that passes the mechanical check.

[The exact-path retention list](../configs/repository-artifacts.json) pins existing
legacy payloads by Git blob ID and byte count. It is **not** a Drive manifest or a
list of files approved for deletion. Existing payloads remain to avoid breaking
replay, tests or active research. Changed bytes or copies at new paths fail.
Do not regenerate the retention list to admit new experiment output. A necessary
fixture exception needs a specific reviewable reason and exact bytes; a release
model exception additionally requires explicit owner approval and an exact-path
ignore exception.

`.gitignore` prevents common accidental staging; CI also checks objects already
staged or force-added. Git blob IDs identify retained repository bytes and must
never be substituted for the archive/member SHA256s required for restoration.

## Removal and retirement

Before copying, archiving, moving or removing research evidence, read the owning
PR's **current** status. Protect every open PR root and its dependencies unless
the owner authorizes that specific action. Historical “merged” notes do not
replace the live check. Never delete inside the Macs' synced folders or force
offloading. bb128's ordinary HDD cache has the separate removal conditions above;
no remote Drive deletion is authorized.

For merged evidence, removal needs current Google Drive upload acceptance plus a
dependency review of active runs, source references, tests and other worktrees.
The owner clarified on October 7 that confirmed Drive uploads are trusted for
cleanup: use existing manifests and provenance without downloading archives or
repeating archive/member hash audits. Check selected originals against recorded
paths, sizes and modification history; retain changed or uncertain files. This
cleanup preference does not change new-archive creation receipts or SHA256 checks
when retrieving models for research use. Write a receipt naming each removed
path, exact restoration archive/member/hash, and update RESULTS_INDEX. Owner
standing authorization applies only after these checks. Unknown provenance means
retain and record the gap, not delete.

On October 14 the owner gave standing permission to remove inactive rebuildable
caches and other regenerable output without separate approval when there is no
data loss and no current active PR is affected. Derived policy caches and build
output can therefore be removed even if excluded from research archives, after
checking generating inputs, active dependencies and rebuild provenance. Record
exact paths, source/input pins and regeneration commands. This exception does
not cover unique scientific evidence or replace original-model archive rules.

Retiring code is separate from deleting evidence. First establish its callers,
fixtures, owning campaign and reproducible source revision. Keep mathematical,
rules, information-boundary and recovery regressions when replacing an approach.
No automatic archive upload, worktree cleanup, history rewrite or background
maintenance is scheduled by these instructions.
